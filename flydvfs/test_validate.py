"""Window validation tests. They do not need a GPU."""

import ctypes
import unittest

from flydvfs.nvml import UtilizationRates
from flydvfs.validate import SampleValidator

WINDOW_NS = 250_000_000
PEAK = 1e12


def _window(
    start: int,
    length: int = WINDOW_NS,
    sm_active: float = 8e8,
    sm_elapsed: float = 1e9,
    dram_bytes: float = 0.0,
) -> dict:
    return {
        "sm_active": sm_active,
        "sm_elapsed": sm_elapsed,
        "dram_bytes": dram_bytes,
        "start_ns": start,
        "end_ns": start + length,
    }


class ValidateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = SampleValidator(0.25, PEAK)

    def test_a_normal_window_is_usable(self):
        self.assertIsNone(self.validator.check(_window(0), None))
        self.assertIsNone(self.validator.check(_window(WINDOW_NS), 50))

    def test_missing_and_nonfinite_fields(self):
        sample = _window(0)
        del sample["dram_bytes"]
        self.assertEqual(self.validator.check(sample, None), "nonfinite")
        sample = _window(0)
        sample["sm_active"] = None
        self.assertEqual(self.validator.check(sample, None), "nonfinite")
        sample = _window(0)
        sample["sm_elapsed"] = float("nan")
        self.assertEqual(self.validator.check(sample, None), "nonfinite")
        sample = _window(0)
        sample["end_ns"] = float("inf")
        self.assertEqual(self.validator.check(sample, None), "nonfinite")

    def test_counters_and_sm_active_above_elapsed(self):
        self.assertEqual(self.validator.check(_window(0, sm_active=-1), None), "out_of_range")
        self.assertEqual(self.validator.check(_window(0, dram_bytes=-1), None), "out_of_range")
        self.assertEqual(self.validator.check(_window(0, sm_elapsed=0), None), "out_of_range")
        self.validator.session_restarted()
        elapsed = 1000.0
        at_limit = _window(0, sm_active=elapsed * 1.05, sm_elapsed=elapsed)
        self.assertIsNone(self.validator.check(at_limit, None))
        over = _window(WINDOW_NS, sm_active=elapsed * 1.05 + 1, sm_elapsed=elapsed)
        self.assertEqual(self.validator.check(over, None), "out_of_range")

    def test_window_length_band_is_wider_for_sysclk(self):
        self.assertIsNone(self.validator.check(_window(0, length=WINDOW_NS // 2), None))
        self.assertIsNone(self.validator.check(_window(WINDOW_NS, length=WINDOW_NS * 2), None))
        too_long = _window(WINDOW_NS * 4, length=WINDOW_NS * 2 + 1)
        self.assertEqual(self.validator.check(too_long, None), "window_length")
        self.validator.session_restarted()
        self.assertEqual(self.validator.check(_window(0, length=WINDOW_NS // 2 - 1), None), "window_length")

        self.validator.session_restarted()
        self.validator.set_trigger("sysclk")
        stretched = _window(0, length=WINDOW_NS * 3)
        self.assertIsNone(self.validator.check(stretched, None))
        self.assertEqual(
            self.validator.check(_window(WINDOW_NS * 3, length=WINDOW_NS * 4 + 1), None),
            "window_length",
        )

    def test_gpu_time_config_sets_the_window(self):
        validator = SampleValidator(0.08, PEAK)
        validator.set_trigger("gpu_time", WINDOW_NS)
        self.assertIsNone(validator.check(_window(0), None))
        validator.set_trigger("sysclk", 100000000)
        self.assertIsNone(validator.check(_window(WINDOW_NS, length=WINDOW_NS * 3), None))

    def test_timestamp_gap_and_session_restart(self):
        self.assertIsNone(self.validator.check(_window(0), None))
        exact = _window(WINDOW_NS + 2 * WINDOW_NS)
        self.assertIsNone(self.validator.check(exact, None))
        jumped = _window(exact["end_ns"] + 2 * WINDOW_NS + 1)
        self.assertEqual(self.validator.check(jumped, None), "timestamp_jump")
        backwards = _window(exact["end_ns"] - 1)
        self.assertEqual(self.validator.check(backwards, None), "timestamp_jump")

        self.validator.session_restarted()
        self.assertIsNone(self.validator.check(_window(10 * WINDOW_NS), None))

    def test_sm_active_against_nvml(self):
        quiet = _window(0, sm_active=5e7, sm_elapsed=1e9)
        self.assertEqual(self.validator.check(quiet, 90.1), "sm_vs_nvml")
        self.validator.session_restarted()
        self.assertIsNone(self.validator.check(quiet, 90), None)
        self.validator.session_restarted()
        self.assertIsNone(self.validator.check(_window(0, sm_active=1e8, sm_elapsed=1e9), 99), None)
        self.validator.session_restarted()
        self.assertIsNone(self.validator.check(quiet, None), None)

    def test_dram_without_sm(self):
        # 0.25 s * 1e12 B/s * 0.3 = 7.5e10 bytes.
        sample = _window(0, sm_active=2e8, sm_elapsed=1e9, dram_bytes=7.5e10)
        self.assertEqual(self.validator.check(sample, None), "dram_without_sm")
        self.validator.session_restarted()
        borderline_sm = _window(0, sm_active=3e8, sm_elapsed=1e9, dram_bytes=7.5e10)
        self.assertIsNone(self.validator.check(borderline_sm, None), None)
        self.validator.session_restarted()
        low_dram = _window(0, sm_active=2e8, sm_elapsed=1e9, dram_bytes=7.5e10 - 1)
        self.assertIsNone(self.validator.check(low_dram, None), None)

    def test_sm_collapse_needs_stable_elapsed_and_dram(self):
        self.assertIsNone(self.validator.check(_window(0, sm_active=1000, sm_elapsed=2000, dram_bytes=1000), None), None)
        collapsed = _window(WINDOW_NS, sm_active=199, sm_elapsed=2000, dram_bytes=1000)
        self.assertEqual(self.validator.check(collapsed, None), "sm_collapse")

        self.validator.session_restarted()
        self.assertIsNone(self.validator.check(_window(0, sm_active=1000, sm_elapsed=2000, dram_bytes=1000), None), None)
        held = _window(WINDOW_NS, sm_active=200, sm_elapsed=2000, dram_bytes=1000)
        self.assertIsNone(self.validator.check(held, None), None)

        self.validator.session_restarted()
        self.assertIsNone(self.validator.check(_window(0, sm_active=1000, sm_elapsed=2000, dram_bytes=1000), None), None)
        elapsed_moved = _window(WINDOW_NS, sm_active=199, sm_elapsed=2000 * 1.2, dram_bytes=1000)
        self.assertIsNone(self.validator.check(elapsed_moved, None), None)

        self.validator.session_restarted()
        self.assertIsNone(self.validator.check(_window(0, sm_active=1000, sm_elapsed=2000, dram_bytes=1000), None), None)
        dram_moved = _window(WINDOW_NS, sm_active=199, sm_elapsed=2000, dram_bytes=3000)
        self.assertIsNone(self.validator.check(dram_moved, None), None)

    def test_nonfinite_wins_over_a_later_reason(self):
        sample = _window(0, sm_active=float("nan"), sm_elapsed=-1)
        self.assertEqual(self.validator.check(sample, None), "nonfinite")


class UtilizationLayoutTest(unittest.TestCase):
    def test_gpu_field_precedes_memory(self):
        self.assertEqual(UtilizationRates.gpu.offset, 0)
        self.assertGreater(UtilizationRates.memory.offset, 0)
        self.assertEqual(UtilizationRates.memory.offset, ctypes.sizeof(ctypes.c_uint))


if __name__ == "__main__":
    unittest.main()

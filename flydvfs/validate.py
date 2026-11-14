"""Decide whether one PM-sampling window is safe to feed the controller.

A window that fails a check is logged and then ignored. The clock only moves
on a window that returns None. session_restarted() drops the timestamp and
the previous valid window, because a new CUPTI session does not continue the
old timeline.
"""

from __future__ import annotations

import math
from typing import Any


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _rate_within(current: float, previous: float, low: float, high: float) -> bool:
    if previous == 0:
        return current == 0
    ratio = current / previous
    return low <= ratio <= high


class SampleValidator:
    def __init__(self, window_s: float, dram_peak_bytes_per_s: float) -> None:
        if window_s <= 0 or dram_peak_bytes_per_s <= 0:
            raise ValueError("window and DRAM peak must be positive")
        self.window_s = window_s
        self.dram_peak_bytes_per_s = dram_peak_bytes_per_s
        self.trigger = "gpu_time"
        self._prev_end_ns: float | None = None
        self._prev_valid: dict[str, float] | None = None

    def set_trigger(self, trigger: str, interval: float | None = None) -> None:
        """Record the sampler trigger. GPU-time interval is the window, in ns."""
        self.trigger = trigger
        if trigger != "gpu_time" or interval is None:
            return
        value = _finite(interval)
        if value is not None and value > 0:
            self.window_s = value * 1e-9

    def session_restarted(self) -> None:
        self._prev_end_ns = None
        self._prev_valid = None

    def _window_ns(self) -> float:
        return self.window_s * 1e9

    def _length_limits_ns(self) -> tuple[float, float]:
        window_ns = self._window_ns()
        high = 4.0 if self.trigger == "sysclk" else 2.0
        return 0.5 * window_ns, high * window_ns

    def check(self, sample: dict, nvml_util_percent: float | None) -> str | None:
        fields = (
            _finite(sample.get("sm_active")),
            _finite(sample.get("sm_elapsed")),
            _finite(sample.get("dram_bytes")),
            _finite(sample.get("start_ns")),
            _finite(sample.get("end_ns")),
        )
        if any(field is None for field in fields):
            return "nonfinite"
        sm_active, sm_elapsed, dram_bytes, start_ns, end_ns = fields
        assert sm_active is not None and sm_elapsed is not None
        assert dram_bytes is not None and start_ns is not None and end_ns is not None

        prev_end = self._prev_end_ns
        self._prev_end_ns = end_ns

        if sm_active < 0 or dram_bytes < 0 or sm_elapsed <= 0 or sm_active > 1.05 * sm_elapsed:
            return "out_of_range"

        length_ns = end_ns - start_ns
        low_ns, high_ns = self._length_limits_ns()
        if length_ns < low_ns or length_ns > high_ns:
            return "window_length"

        if prev_end is not None:
            gap_ns = start_ns - prev_end
            if gap_ns < 0 or gap_ns > 2 * self._window_ns():
                return "timestamp_jump"

        sm_fraction = sm_active / sm_elapsed
        util = _finite(nvml_util_percent)
        if util is not None and util > 90 and sm_fraction < 0.1:
            return "sm_vs_nvml"

        window_s = length_ns * 1e-9
        dram_fraction = dram_bytes / (window_s * self.dram_peak_bytes_per_s)
        if dram_fraction >= 0.3 and sm_fraction < 0.3:
            return "dram_without_sm"

        if self._prev_valid is not None:
            previous = self._prev_valid
            elapsed_rate = sm_elapsed / window_s
            prev_elapsed_rate = previous["sm_elapsed"] / previous["window_s"]
            dram_rate = dram_bytes / window_s
            prev_dram_rate = previous["dram_bytes"] / previous["window_s"]
            elapsed_stable = _rate_within(elapsed_rate, prev_elapsed_rate, 0.9, 1.1)
            dram_stable = _rate_within(dram_rate, prev_dram_rate, 0.5, 2.0)
            if sm_active < 0.2 * previous["sm_active"] and elapsed_stable and dram_stable:
                return "sm_collapse"

        self._prev_valid = {
            "sm_active": sm_active,
            "sm_elapsed": sm_elapsed,
            "dram_bytes": dram_bytes,
            "window_s": window_s,
        }
        return None

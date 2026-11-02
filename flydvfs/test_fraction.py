"""Fraction tests. They do not need a GPU."""

import unittest

from flydvfs.fraction import compute_fraction, dram_peak_bytes_per_s, window_throughputs


class FractionTest(unittest.TestCase):
    def test_a100_pcie_peak_bandwidth(self):
        # 1512 MHz memory clock, 5120-bit bus, both edges: 1935 GB/s.
        peak = dram_peak_bytes_per_s(1512, 5120)
        self.assertAlmostEqual(peak, 1.93536e12, delta=1e9)

    def test_idle_window_does_not_change_the_clock(self):
        self.assertIsNone(compute_fraction(0.0, 0.0))
        self.assertIsNone(compute_fraction(0.01, 0.01))

    def test_compute_window_is_high_and_memory_window_is_near_the_middle(self):
        # Numbers from one-pass PM sampling on an A100: FMA, then a streaming copy.
        window_s = 0.0725
        peak = dram_peak_bytes_per_s(1512, 5120)
        fma = window_throughputs(8702024007, 11031797890, 1035264, window_s, peak)
        copy = window_throughputs(10861164833, 11027092388, 111865012352, window_s, peak)
        self.assertIsNotNone(fma)
        self.assertIsNotNone(copy)
        self.assertGreater(compute_fraction(*fma), 0.90)
        self.assertGreater(compute_fraction(*copy), 0.30)
        self.assertLess(compute_fraction(*copy), 0.60)


if __name__ == "__main__":
    unittest.main()

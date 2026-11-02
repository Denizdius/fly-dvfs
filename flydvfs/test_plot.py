"""Trace and figure tests. They do not need a GPU."""

import tempfile
import unittest
from pathlib import Path

from flydvfs.plot import plot_run, write_trace


class PlotTest(unittest.TestCase):
    def test_trace_and_figures(self):
        rows = [
            (0.0, 50.0, 1410, 1512, 1410),
            (0.1, 280.0, 1350, 1512, 1410),
            (0.2, 220.0, 810, 1512, 810),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.csv"
            write_trace(trace, rows)
            text = trace.read_text().splitlines()
            self.assertEqual(text[0], "t_s,power_w,sm_mhz,mem_mhz,cap_mhz")
            self.assertEqual(len(text), 4)
            power, frequency = plot_run(root, rows)
            self.assertGreater(power.stat().st_size, 1000)
            self.assertGreater(frequency.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()

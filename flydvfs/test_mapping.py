"""Mapping tests. They do not need a GPU."""

import unittest

from flydvfs.mapping import map_compute_fraction_to_sm_mhz, span_position


class MappingTest(unittest.TestCase):
    def test_levels(self):
        self.assertIsNone(span_position(0.95))
        self.assertIsNone(span_position(0.90))
        self.assertEqual(span_position(0.75), 0.75)
        self.assertEqual(span_position(0.60), 0.75)
        self.assertEqual(span_position(0.45), 0.50)
        self.assertEqual(span_position(0.30), 0.50)
        self.assertAlmostEqual(span_position(0.29), 1.0 / 3.0)

    def test_levels_use_this_devices_span(self):
        # Catalog 210..1410. One third, one half, and three quarters of that span.
        self.assertIsNone(map_compute_fraction_to_sm_mhz(0.95, 210, 1410))
        self.assertEqual(map_compute_fraction_to_sm_mhz(0.75, 210, 1410), 1110)
        self.assertEqual(map_compute_fraction_to_sm_mhz(0.45, 210, 1410), 810)
        self.assertEqual(map_compute_fraction_to_sm_mhz(0.20, 210, 1410), 610)
        self.assertIsNone(map_compute_fraction_to_sm_mhz(0.90, 345, 1980))

    def test_snaps_to_supported_clock(self):
        supported = [210, 900, 1410]
        self.assertEqual(map_compute_fraction_to_sm_mhz(0.75, 210, 1410, supported), 900)
        self.assertIsNone(map_compute_fraction_to_sm_mhz(0.95, 210, 1410, supported))


if __name__ == "__main__":
    unittest.main()

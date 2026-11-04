"""Mapping tests. They do not need a GPU."""

import unittest

from flydvfs.mapping import map_compute_fraction_to_sm_mhz, span_position


class MappingTest(unittest.TestCase):
    def test_levels(self):
        self.assertIsNone(span_position(0.95))
        self.assertIsNone(span_position(0.80))
        self.assertEqual(span_position(0.79), 0.90)
        self.assertEqual(span_position(0.70), 0.90)
        self.assertEqual(span_position(0.69), 0.75)
        self.assertEqual(span_position(0.50), 0.75)
        self.assertEqual(span_position(0.49), 0.50)
        self.assertEqual(span_position(0.30), 0.50)
        self.assertEqual(span_position(0.29), 0.30)

    def test_levels_use_this_devices_span(self):
        # Catalog 210..1410: 1290, 1110, 810, 570 MHz.
        self.assertIsNone(map_compute_fraction_to_sm_mhz(0.85, 210, 1410))
        self.assertEqual(map_compute_fraction_to_sm_mhz(0.75, 210, 1410), 1290)
        self.assertEqual(map_compute_fraction_to_sm_mhz(0.60, 210, 1410), 1110)
        self.assertEqual(map_compute_fraction_to_sm_mhz(0.40, 210, 1410), 810)
        self.assertEqual(map_compute_fraction_to_sm_mhz(0.20, 210, 1410), 570)
        # Catalog 210..3150: 2856, 2415, 1680, 1092 MHz.
        self.assertEqual(map_compute_fraction_to_sm_mhz(0.75, 210, 3150), 2856)
        self.assertEqual(map_compute_fraction_to_sm_mhz(0.60, 210, 3150), 2415)
        self.assertEqual(map_compute_fraction_to_sm_mhz(0.40, 210, 3150), 1680)
        self.assertEqual(map_compute_fraction_to_sm_mhz(0.20, 210, 3150), 1092)

    def test_snaps_to_supported_clock(self):
        supported = [210, 900, 1410]
        self.assertEqual(map_compute_fraction_to_sm_mhz(0.60, 210, 1410, supported), 900)
        self.assertIsNone(map_compute_fraction_to_sm_mhz(0.85, 210, 1410, supported))


if __name__ == "__main__":
    unittest.main()

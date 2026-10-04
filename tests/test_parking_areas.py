"""
Unit tests for ParkingAreaManager (Saved Parking Area Configurations).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.parking_config import ParkingAreaManager


class TestParkingAreaManager(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.manager = ParkingAreaManager(self.temp_dir.name)

        self.valid_area = {
            "parking_area_id": "area_north_01",
            "name": "North Lot",
            "reference_image": "north.jpg",
            "spaces": [
                {
                    "space_id": "P1",
                    "polygon": [[10, 10], [50, 10], [50, 80], [10, 80]],
                },
                {
                    "space_id": "P2",
                    "polygon": [[60, 10], [100, 10], [100, 80], [60, 80]],
                },
            ],
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_valid_area(self):
        """Test 1: Valid parking area creation and persistence."""
        saved = self.manager.create_area(self.valid_area)
        self.assertEqual(saved["parking_area_id"], "area_north_01")
        self.assertEqual(saved["name"], "North Lot")
        self.assertEqual(len(saved["spaces"]), 2)

        # File exists on disk
        filepath = Path(self.temp_dir.name) / "area_north_01.json"
        self.assertTrue(filepath.exists())

    def test_load_saved_area(self):
        """Test 2: Loading a saved parking area by ID."""
        self.manager.create_area(self.valid_area)
        loaded = self.manager.get_area("area_north_01")

        self.assertEqual(loaded["parking_area_id"], "area_north_01")
        self.assertEqual(loaded["name"], "North Lot")
        self.assertEqual(len(loaded["spaces"]), 2)
        self.assertEqual(loaded["spaces"][0]["space_id"], "P1")
        self.assertEqual(len(loaded["spaces"][0]["polygon"]), 4)

    def test_list_saved_areas(self):
        """Test 3: Listing multiple saved parking areas."""
        self.manager.create_area(self.valid_area)

        second_area = {
            "parking_area_id": "area_south_02",
            "name": "South Lot",
            "reference_image": "south.jpg",
            "spaces": [
                {"space_id": "S1", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}
            ],
        }
        self.manager.create_area(second_area)

        area_list = self.manager.list_areas()
        self.assertEqual(len(area_list), 2)
        ids = [a["parking_area_id"] for a in area_list]
        self.assertIn("area_north_01", ids)
        self.assertIn("area_south_02", ids)

    def test_unknown_area_id_raises_file_not_found(self):
        """Test 4: Requesting non-existent area ID raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.manager.get_area("non_existent_area_999")

    def test_duplicate_area_id_rejected(self):
        """Test 5: Creating area with existing ID without overwrite raises ValueError."""
        self.manager.create_area(self.valid_area)
        with self.assertRaises(ValueError):
            self.manager.create_area(self.valid_area, overwrite=False)

    def test_duplicate_space_id_within_area_rejected(self):
        """Test 6: Reject areas containing duplicate space IDs."""
        invalid_area = {
            "parking_area_id": "bad_area",
            "name": "Bad Area",
            "spaces": [
                {"space_id": "P1", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
                {"space_id": "P1", "polygon": [[20, 0], [30, 0], [30, 10], [20, 10]]},
            ],
        }
        with self.assertRaises(ValueError) as ctx:
            self.manager.create_area(invalid_area)
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_invalid_polygon_point_count_rejected(self):
        """Test 7: Reject polygons with != 4 points."""
        invalid_area = {
            "parking_area_id": "bad_poly_area",
            "name": "Bad Poly",
            "spaces": [
                {"space_id": "P1", "polygon": [[0, 0], [10, 0], [10, 10]]}  # Only 3 points
            ],
        }
        with self.assertRaises(ValueError):
            self.manager.create_area(invalid_area)

    def test_empty_id_or_name_rejected(self):
        """Test 8: Reject empty parking_area_id or name."""
        no_id = {"parking_area_id": "", "name": "Lot", "spaces": [{"space_id": "P1", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]}
        with self.assertRaises(ValueError):
            self.manager.create_area(no_id)

        no_name = {"parking_area_id": "id_1", "name": "   ", "spaces": [{"space_id": "P1", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]}
        with self.assertRaises(ValueError):
            self.manager.create_area(no_name)


if __name__ == "__main__":
    unittest.main()

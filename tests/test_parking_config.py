"""
Unit tests for ParkingConfig loader and validator.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.parking_config import ParkingConfig


class TestParkingConfig(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_json(self, filename: str, data: dict) -> Path:
        filepath = self.temp_dir_path / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return filepath

    def test_load_valid_configuration(self):
        """Test 1: Verify a valid JSON file loads correctly."""
        valid_data = {
            "image": "data/images/1.jpg",
            "spaces": [
                {
                    "id": "P1",
                    "polygon": [[100, 100], [200, 100], [200, 250], [100, 250]],
                }
            ],
        }
        filepath = self._write_json("valid.json", valid_data)
        config = ParkingConfig(filepath)
        loaded = config.load()

        self.assertEqual(loaded["image"], "data/images/1.jpg")
        self.assertEqual(len(loaded["spaces"]), 1)
        self.assertEqual(config.get_image_path(), "data/images/1.jpg")

    def test_load_multiple_parking_spaces(self):
        """Test 2: Verify multiple spaces are returned properly."""
        valid_data = {
            "image": "data/images/1.jpg",
            "spaces": [
                {
                    "id": "P1",
                    "polygon": [[10, 10], [50, 10], [50, 80], [10, 80]],
                },
                {
                    "id": "P2",
                    "polygon": [[60, 10], [100, 10], [100, 80], [60, 80]],
                },
                {
                    "id": "P3",
                    "polygon": [[110, 10], [150, 10], [150, 80], [110, 80]],
                },
            ],
        }
        filepath = self._write_json("multiple.json", valid_data)
        config = ParkingConfig(filepath)
        spaces = config.get_spaces()

        self.assertEqual(len(spaces), 3)

    def test_preserve_parking_space_ids(self):
        """Test 3: Verify IDs such as P1, P2, P3 are preserved in order."""
        expected_ids = ["P1", "P2", "P3", "P4"]
        valid_data = {
            "image": "data/images/1.jpg",
            "spaces": [
                {"id": sid, "polygon": [[0, 0], [10, 0], [10, 20], [0, 20]]}
                for sid in expected_ids
            ],
        }
        filepath = self._write_json("ids.json", valid_data)
        config = ParkingConfig(filepath)
        spaces = config.get_spaces()

        loaded_ids = [s["id"] for s in spaces]
        self.assertEqual(loaded_ids, expected_ids)

    def test_polygon_structure_four_points(self):
        """Test 4: Verify every space contains a polygon with exactly 4 coordinate points."""
        valid_data = {
            "image": "data/images/1.jpg",
            "spaces": [
                {
                    "id": "P1",
                    "polygon": [[100, 150], [200, 150], [220, 300], [120, 300]],
                }
            ],
        }
        filepath = self._write_json("poly.json", valid_data)
        config = ParkingConfig(filepath)
        spaces = config.get_spaces()

        for space in spaces:
            self.assertIn("polygon", space)
            polygon = space["polygon"]
            self.assertEqual(len(polygon), 4)
            for pt in polygon:
                self.assertEqual(len(pt), 2)
                self.assertIsInstance(pt[0], (int, float))
                self.assertIsInstance(pt[1], (int, float))

    def test_reject_nonexistent_file(self):
        """Test 5a: Verify non-existent file raises FileNotFoundError."""
        config = ParkingConfig(self.temp_dir_path / "does_not_exist.json")
        with self.assertRaises(FileNotFoundError):
            config.load()

    def test_reject_malformed_json(self):
        """Test 5b: Verify invalid JSON syntax raises ValueError."""
        filepath = self.temp_dir_path / "bad.json"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("{ bad json content ")

        config = ParkingConfig(filepath)
        with self.assertRaises(ValueError):
            config.load()

    def test_reject_invalid_polygon_point_count(self):
        """Test 5c: Verify polygon with fewer or more than 4 points is rejected."""
        bad_data = {
            "image": "data/images/1.jpg",
            "spaces": [
                {
                    "id": "P1",
                    "polygon": [[100, 100], [200, 100], [200, 200]],  # Only 3 points
                }
            ],
        }
        filepath = self._write_json("triangle.json", bad_data)
        config = ParkingConfig(filepath)
        with self.assertRaises(ValueError):
            config.load()

    def test_reject_invalid_coordinate_types(self):
        """Test 5d: Verify non-numeric coordinate values are rejected."""
        bad_data = {
            "image": "data/images/1.jpg",
            "spaces": [
                {
                    "id": "P1",
                    "polygon": [["a", "b"], [200, 100], [200, 200], [100, 200]],
                }
            ],
        }
        filepath = self._write_json("non_numeric.json", bad_data)
        config = ParkingConfig(filepath)
        with self.assertRaises(ValueError):
            config.load()

    def test_save_and_reload(self):
        """Test 6: Verify saving and reloading configuration via ParkingConfig."""
        filepath = self.temp_dir_path / "saved_config.json"
        config = ParkingConfig(filepath)
        spaces = [
            {
                "id": "P1",
                "polygon": [[10, 10], [50, 10], [50, 50], [10, 50]],
            }
        ]
        config.save("data/images/1.jpg", spaces)

        # Reload with new instance
        reloader = ParkingConfig(filepath)
        loaded_spaces = reloader.get_spaces()
        self.assertEqual(len(loaded_spaces), 1)
        self.assertEqual(loaded_spaces[0]["id"], "P1")
        self.assertEqual(reloader.get_image_path(), "data/images/1.jpg")


if __name__ == "__main__":
    unittest.main()

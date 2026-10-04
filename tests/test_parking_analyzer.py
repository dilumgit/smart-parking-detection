"""
Unit tests for Parking Occupancy Analyzer module.
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.parking_analyzer import ParkingAnalyzer


class TestParkingAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = ParkingAnalyzer()
        # Define a 100x100 square polygon: [0, 0] to [100, 100]
        self.sample_space = {
            "id": "P1",
            "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
        }

    def test_empty_parking_space(self):
        """Test 1: Parking space with no vehicle detections is empty."""
        spaces = [self.sample_space]
        detections = []
        results = self.analyzer.analyze(detections, spaces)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "P1")
        self.assertFalse(results[0]["occupied"])
        self.assertEqual(results[0]["vehicle_count"], 0)
        self.assertEqual(results[0]["vehicles"], [])

    def test_vehicle_inside_parking_space(self):
        """Test 2: Vehicle whose center is inside the polygon marks space as occupied."""
        spaces = [self.sample_space]
        # Bbox [20, 20, 60, 60] -> Center (40, 40) is inside [0, 0, 100, 100]
        detections = [
            {"class": "car", "confidence": 0.92, "bbox": [20, 20, 60, 60]}
        ]
        results = self.analyzer.analyze(detections, spaces)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["occupied"])
        self.assertEqual(results[0]["vehicle_count"], 1)
        self.assertEqual(len(results[0]["vehicles"]), 1)

    def test_vehicle_outside_parking_space(self):
        """Test 3: Vehicle whose center is outside the polygon leaves space as empty."""
        spaces = [self.sample_space]
        # Bbox [200, 200, 300, 300] -> Center (250, 250) is outside [0, 0, 100, 100]
        detections = [
            {"class": "car", "confidence": 0.88, "bbox": [200, 200, 300, 300]}
        ]
        results = self.analyzer.analyze(detections, spaces)

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["occupied"])
        self.assertEqual(results[0]["vehicle_count"], 0)

    def test_multiple_parking_spaces(self):
        """Test 4: Multiple parking spaces are analyzed independently."""
        spaces = [
            {"id": "P1", "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]},
            {"id": "P2", "polygon": [[120, 0], [220, 0], [220, 100], [120, 100]]},
            {"id": "P3", "polygon": [[240, 0], [340, 0], [340, 100], [240, 100]]},
        ]
        # Car only in P2 (center 170, 50)
        detections = [
            {"class": "car", "confidence": 0.85, "bbox": [150, 30, 190, 70]}
        ]
        results = self.analyzer.analyze(detections, spaces)

        self.assertEqual(len(results), 3)
        self.assertFalse(results[0]["occupied"])  # P1 empty
        self.assertTrue(results[1]["occupied"])   # P2 occupied
        self.assertFalse(results[2]["occupied"])  # P3 empty

    def test_multiple_vehicles_in_same_space(self):
        """Test 5: Multiple vehicle centers in same space marks occupied with vehicle_count=2."""
        spaces = [self.sample_space]
        detections = [
            {"class": "motorcycle", "confidence": 0.90, "bbox": [10, 10, 30, 30]},
            {"class": "motorcycle", "confidence": 0.85, "bbox": [60, 60, 80, 80]},
        ]
        results = self.analyzer.analyze(detections, spaces)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["occupied"])
        self.assertEqual(results[0]["vehicle_count"], 2)

    def test_vehicle_class_preservation(self):
        """Test 6: Vehicle metadata and class names are preserved in detection results."""
        spaces = [self.sample_space]
        detections = [
            {
                "class_name": "truck",
                "class": "truck",
                "confidence": 0.94,
                "bbox": [10, 10, 70, 70],
            }
        ]
        results = self.analyzer.analyze(detections, spaces)

        self.assertEqual(len(results[0]["vehicles"]), 1)
        vehicle = results[0]["vehicles"][0]
        self.assertEqual(vehicle.get("class") or vehicle.get("class_name"), "truck")
        self.assertEqual(vehicle["confidence"], 0.94)

    def test_occupancy_rate_calculation(self):
        """Test 7: Occupancy percentage calculation is accurate."""
        # 1 occupied out of 4 spaces -> 25.0%
        mock_results = [
            {"id": "P1", "occupied": True},
            {"id": "P2", "occupied": False},
            {"id": "P3", "occupied": False},
            {"id": "P4", "occupied": False},
        ]
        rate = self.analyzer.calculate_occupancy_rate(mock_results)
        self.assertEqual(rate, 25.0)

        # 3 occupied out of 4 spaces -> 75.0%
        mock_results[1]["occupied"] = True
        mock_results[2]["occupied"] = True
        rate = self.analyzer.calculate_occupancy_rate(mock_results)
        self.assertEqual(rate, 75.0)

    def test_no_parking_spaces(self):
        """Test 8: Empty parking space list handles cleanly without division by zero."""
        results = self.analyzer.analyze([], [])
        self.assertEqual(results, [])

        rate = self.analyzer.calculate_occupancy_rate([])
        self.assertEqual(rate, 0.0)

        summary = self.analyzer.get_summary([])
        self.assertEqual(summary["total_spaces"], 0)
        self.assertEqual(summary["occupied_spaces"], 0)
        self.assertEqual(summary["occupancy_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()

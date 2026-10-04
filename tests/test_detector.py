"""
Unit tests for YOLO Vehicle Detector module using standard library unittest.
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.detector import VehicleDetector, VEHICLE_CLASSES


class TestVehicleDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = VehicleDetector(model_path="yolo11n.pt", conf_threshold=0.4)

    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.conf_threshold, 0.4)
        self.assertIn("car", self.detector.vehicle_classes)
        self.assertIn("truck", self.detector.vehicle_classes)
        self.assertIn("bus", self.detector.vehicle_classes)
        self.assertIn("motorcycle", self.detector.vehicle_classes)

    def test_detector_invalid_image_path(self):
        with self.assertRaises(FileNotFoundError):
            self.detector.detect("non_existent_image_path.jpg")

    def test_detector_numpy_input(self):
        # Blank 640x640 RGB image should yield 0 vehicle detections
        blank_image = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = self.detector.detect(blank_image)
        self.assertIsInstance(detections, list)
        self.assertEqual(len(detections), 0)

    def test_detector_output_structure(self):
        sample_path = PROJECT_ROOT / "data" / "images" / "parking.jpg"
        if sample_path.exists():
            detections = self.detector.detect(str(sample_path))
            self.assertIsInstance(detections, list)
            for det in detections:
                self.assertIn("class_name", det)
                self.assertIn("confidence", det)
                self.assertIn("bbox", det)
                self.assertIn(det["class_name"], VEHICLE_CLASSES)
                self.assertTrue(0.0 <= det["confidence"] <= 1.0)
                self.assertEqual(len(det["bbox"]), 4)
                x1, y1, x2, y2 = det["bbox"]
                self.assertLessEqual(x1, x2)
                self.assertLessEqual(y1, y2)


if __name__ == "__main__":
    unittest.main()

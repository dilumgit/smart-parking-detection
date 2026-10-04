"""
Unit tests for OpenCV Visualizer module.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.visualizer import draw_detections, CLASS_COLORS


class TestVisualizer(unittest.TestCase):

    def setUp(self):
        # Create a blank 480x640x3 BGR image
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_draw_detections_returns_image(self):
        """Test 1: Given valid image and detection, draw_detections returns a valid image."""
        detections = [
            {
                "class": "car",
                "confidence": 0.94,
                "bbox": [100, 100, 300, 300],
            }
        ]
        result = draw_detections(self.image, detections)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, self.image.dtype)
        # Ensure that drawing altered pixels on the annotated image
        self.assertTrue(np.any(result > 0))

    def test_draw_detections_empty(self):
        """Test 2: Empty detections list returns an image without error."""
        result = draw_detections(self.image, [])
        self.assertIsNotNone(result)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.image.shape)
        # Result should match original blank image
        np.testing.assert_array_equal(result, self.image)

    def test_draw_detections_multiple_classes(self):
        """Test 3: Detections containing multiple vehicle classes (car, motorcycle, bus, truck)."""
        detections = [
            {"class": "car", "confidence": 0.95, "bbox": [50, 50, 150, 150]},
            {"class": "motorcycle", "confidence": 0.88, "bbox": [160, 50, 260, 150]},
            {"class": "bus", "confidence": 0.91, "bbox": [270, 50, 400, 200]},
            {"class": "truck", "confidence": 0.85, "bbox": [410, 50, 550, 220]},
        ]
        result = draw_detections(self.image, detections)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.image.shape)
        self.assertTrue(np.any(result > 0))

    def test_draw_detections_preserves_input(self):
        """Test 4: Input image and input detections dictionary are not modified in-place."""
        original_copy = self.image.copy()
        detections = [
            {"class": "car", "confidence": 0.90, "bbox": [100, 100, 200, 200]}
        ]
        detections_copy = [dict(d) for d in detections]

        _ = draw_detections(self.image, detections)

        # Original image remains unaltered
        np.testing.assert_array_equal(self.image, original_copy)
        # Detection dicts remain unaltered
        self.assertEqual(detections, detections_copy)

    def test_draw_detections_none_image_raises(self):
        """Test 5: Passing None as image raises ValueError."""
        with self.assertRaises(ValueError):
            draw_detections(None, [])


if __name__ == "__main__":
    unittest.main()

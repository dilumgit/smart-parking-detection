"""
Unit and integration tests for Streamlit frontend helper functions, API client, and canvas coordinate handling.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.app import (
    check_api_health,
    fetch_saved_areas,
    render_spaces_preview,
    save_parking_area_api,
    send_analyze_request,
)


class TestStreamlitAppHelpers(unittest.TestCase):

    def setUp(self):
        self.base_url = "http://127.0.0.1:8000"
        self.sample_spaces = [
            {
                "space_id": "P1",
                "polygon": [[10, 10], [50, 10], [50, 50], [10, 50]],
            },
            {
                "space_id": "P2",
                "polygon": [[60, 10], [100, 10], [100, 50], [60, 50]],
            },
        ]

    def test_render_spaces_preview_output_shape_and_type(self):
        """Test 1: render_spaces_preview renders polygons and returns valid RGB image."""
        blank_image = np.zeros((200, 300, 3), dtype=np.uint8)
        rendered = render_spaces_preview(
            image_bgr=blank_image,
            spaces=self.sample_spaces,
            current_points=[(120, 20), (130, 30)],
        )

        self.assertIsInstance(rendered, np.ndarray)
        self.assertEqual(rendered.shape, (200, 300, 3))
        self.assertEqual(rendered.dtype, np.uint8)

    def test_check_api_health_success(self):
        """Test 2: check_api_health returns True on HTTP 200."""
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"status": "healthy", "service": "smart-parking-ai"}
            mock_get.return_value = mock_resp

            is_healthy, data, err = check_api_health(self.base_url)
            self.assertTrue(is_healthy)
            self.assertEqual(data["status"], "healthy")
            self.assertIsNone(err)

    def test_check_api_health_failure(self):
        """Test 3: check_api_health handles connection failure gracefully."""
        import requests
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError):
            is_healthy, data, err = check_api_health(self.base_url)
            self.assertFalse(is_healthy)
            self.assertIsNone(data)
            self.assertIn("refused", err.lower())

    def test_fetch_saved_areas_success(self):
        """Test 4: fetch_saved_areas parses API response on HTTP 200."""
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = [
                {
                    "parking_area_id": "parking_area_001",
                    "name": "Main Lot",
                    "reference_image": "1.jpg",
                    "total_spaces": 4,
                }
            ]
            mock_get.return_value = mock_resp

            areas, err = fetch_saved_areas(self.base_url)
            self.assertIsNone(err)
            self.assertEqual(len(areas), 1)
            self.assertEqual(areas[0]["parking_area_id"], "parking_area_001")

    def test_save_parking_area_api_success(self):
        """Test 5: save_parking_area_api sends POST request and returns saved area."""
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 201
            mock_resp.json.return_value = {
                "parking_area_id": "parking_area_002",
                "name": "West Wing",
                "reference_image": "1.jpg",
                "total_spaces": 2,
                "spaces": self.sample_spaces,
            }
            mock_post.return_value = mock_resp

            payload = {
                "parking_area_id": "parking_area_002",
                "name": "West Wing",
                "reference_image": "1.jpg",
                "spaces": self.sample_spaces,
            }
            saved, err = save_parking_area_api(self.base_url, payload)
            self.assertIsNone(err)
            self.assertIsNotNone(saved)
            self.assertEqual(saved["parking_area_id"], "parking_area_002")

    def test_save_parking_area_api_error_handling(self):
        """Test 6: save_parking_area_api handles HTTP 400 error cleanly."""
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 400
            mock_resp.json.return_value = {"detail": "Parking area with ID 'parking_area_001' already exists."}
            mock_post.return_value = mock_resp

            payload = {"parking_area_id": "parking_area_001", "name": "Dup", "spaces": []}
            saved, err = save_parking_area_api(self.base_url, payload)
            self.assertIsNone(saved)
            self.assertIn("already exists", err)

    def test_send_analyze_request_success(self):
        """Test 7: send_analyze_request sends image and optional area ID."""
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "success": True,
                "filename": "1.jpg",
                "summary": {
                    "total_spaces": 4,
                    "occupied_spaces": 1,
                    "empty_spaces": 3,
                    "occupancy_rate": 25.0,
                },
                "parking_spaces": [],
            }
            mock_post.return_value = mock_resp

            res, err = send_analyze_request(
                base_url=self.base_url,
                file_bytes=b"mock-image-bytes",
                filename="1.jpg",
                parking_area_id="parking_area_001",
            )
            self.assertIsNone(err)
            self.assertTrue(res["success"])
            self.assertEqual(res["summary"]["total_spaces"], 4)

    def test_component_html_template_exists(self):
        """Test 8: Ensure the custom image_annotator component HTML file is present and valid."""
        component_path = PROJECT_ROOT / "streamlit_app" / "components" / "image_annotator" / "index.html"
        self.assertTrue(component_path.exists())
        content = component_path.read_text(encoding="utf-8")
        self.assertIn("streamlit:setComponentValue", content)
        self.assertIn("streamlit:componentReady", content)
        self.assertIn("streamlit:render", content)
        self.assertIn("canvas", content.lower())

    def test_polygon_completion_and_state_transitions(self):
        """Test 9: Simulate sequential 4-click point entry, polygon closure, and undo logic."""
        setup_spaces = []
        setup_points = []

        # 4 clicks for Space P1
        clicks_p1 = [[45, 123], [110, 237], [213, 212], [138, 105]]
        for pt in clicks_p1:
            setup_points.append(pt)
            if len(setup_points) == 4:
                space_id = f"P{len(setup_spaces) + 1}"
                setup_spaces.append({"space_id": space_id, "polygon": list(setup_points)})
                setup_points = []

        self.assertEqual(len(setup_spaces), 1)
        self.assertEqual(setup_spaces[0]["space_id"], "P1")
        self.assertEqual(len(setup_spaces[0]["polygon"]), 4)
        self.assertEqual(len(setup_points), 0)

        # 2 clicks in progress for P2
        setup_points.append([140, 104])
        setup_points.append([212, 214])
        self.assertEqual(len(setup_points), 2)

        # Undo last point
        setup_points.pop()
        self.assertEqual(len(setup_points), 1)
        self.assertEqual(setup_points[0], [140, 104])

        # Delete last space
        deleted = setup_spaces.pop()
        self.assertEqual(deleted["space_id"], "P1")
        self.assertEqual(len(setup_spaces), 0)


if __name__ == "__main__":
    unittest.main()

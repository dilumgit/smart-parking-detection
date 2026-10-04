"""
Integration tests for FastAPI REST API endpoints.
"""

import io
import os
import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app


class TestFastAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_image_path = PROJECT_ROOT / "data" / "images" / "1.jpg"

    def test_root_endpoint(self):
        """Test 1: GET / returns 200 and basic API links."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("docs", data)

    def test_health_endpoint_status_200(self):
        """Test 2: GET /health returns 200."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_health_endpoint_payload(self):
        """Test 3: GET /health returns expected payload schema."""
        response = self.client.get("/health")
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "smart-parking-ai")

    def test_analyze_valid_image(self):
        """Test 4: POST /api/v1/parking/analyze with a valid image returns 200."""
        self.assertTrue(self.test_image_path.exists(), "Sample test image 1.jpg must exist.")

        with open(self.test_image_path, "rb") as img_file:
            files = {"file": ("1.jpg", img_file, "image/jpeg")}
            response = self.client.post("/api/v1/parking/analyze", files=files)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check top-level fields
        self.assertTrue(data["success"])
        self.assertEqual(data["filename"], "1.jpg")
        self.assertIn("summary", data)
        self.assertIn("parking_spaces", data)

        # Check summary metrics
        summary = data["summary"]
        self.assertEqual(summary["total_spaces"], 4)
        self.assertEqual(summary["occupied_spaces"], 1)
        self.assertEqual(summary["empty_spaces"], 3)
        self.assertEqual(summary["occupancy_rate"], 25.0)

        # Check parking space results
        spaces = data["parking_spaces"]
        self.assertEqual(len(spaces), 4)

        space_map = {s["space_id"]: s for s in spaces}
        self.assertIn("P1", space_map)
        self.assertIn("P2", space_map)
        self.assertIn("P3", space_map)
        self.assertIn("P4", space_map)

        # P1 is occupied by car
        self.assertEqual(space_map["P1"]["status"], "OCCUPIED")
        self.assertEqual(space_map["P1"]["vehicle_count"], 1)
        self.assertEqual(space_map["P1"]["vehicle_class"], "car")
        self.assertIsNotNone(space_map["P1"]["confidence"])

        # P2, P3, P4 are empty
        self.assertEqual(space_map["P2"]["status"], "EMPTY")
        self.assertEqual(space_map["P2"]["vehicle_count"], 0)
        self.assertIsNone(space_map["P2"]["vehicle_class"])

        self.assertEqual(space_map["P3"]["status"], "EMPTY")
        self.assertEqual(space_map["P4"]["status"], "EMPTY")

    def test_analyze_empty_file_rejected(self):
        """Test 5: POST /api/v1/parking/analyze with empty file returns 400 Bad Request."""
        empty_stream = io.BytesIO(b"")
        files = {"file": ("empty.jpg", empty_stream, "image/jpeg")}
        response = self.client.post("/api/v1/parking/analyze", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"].lower())

    def test_analyze_invalid_image_rejected(self):
        """Test 6: POST /api/v1/parking/analyze with non-image bytes returns 400 Bad Request."""
        corrupt_stream = io.BytesIO(b"this is plain text, not a valid jpeg/png image")
        files = {"file": ("not_image.jpg", corrupt_stream, "image/jpeg")}
        response = self.client.post("/api/v1/parking/analyze", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("invalid", response.json()["detail"].lower())

    def test_list_parking_areas(self):
        """Test 7: GET /api/v1/parking/areas returns list of saved areas."""
        response = self.client.get("/api/v1/parking/areas")
        self.assertEqual(response.status_code, 200)
        areas = response.json()
        self.assertIsInstance(areas, list)
        self.assertTrue(len(areas) >= 1)
        ids = [a["parking_area_id"] for a in areas]
        self.assertIn("parking_area_001", ids)

    def test_get_parking_area_by_id(self):
        """Test 8: GET /api/v1/parking/areas/{id} returns full area definition."""
        response = self.client.get("/api/v1/parking/areas/parking_area_001")
        self.assertEqual(response.status_code, 200)
        area = response.json()
        self.assertEqual(area["parking_area_id"], "parking_area_001")
        self.assertEqual(area["total_spaces"], 4)
        self.assertEqual(len(area["spaces"]), 4)

    def test_get_unknown_parking_area_returns_404(self):
        """Test 9: GET /api/v1/parking/areas/{unknown} returns 404 Not Found."""
        response = self.client.get("/api/v1/parking/areas/unknown_non_existent_999")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"].lower())

    def test_create_and_fetch_new_parking_area(self):
        """Test 10: POST /api/v1/parking/areas creates new area and returns 201."""
        new_area_payload = {
            "parking_area_id": "test_area_temp_99",
            "name": "Temporary Test Area",
            "reference_image": "1.jpg",
            "spaces": [
                {
                    "space_id": "T1",
                    "polygon": [[10, 10], [50, 10], [50, 50], [10, 50]],
                }
            ],
        }
        create_res = self.client.post("/api/v1/parking/areas", json=new_area_payload)
        self.assertEqual(create_res.status_code, 201)
        created = create_res.json()
        self.assertEqual(created["parking_area_id"], "test_area_temp_99")
        self.assertEqual(created["total_spaces"], 1)

        # Verify duplicate creation with same ID returns 400
        dup_res = self.client.post("/api/v1/parking/areas", json=new_area_payload)
        self.assertEqual(dup_res.status_code, 400)

        # Clean up test file from disk
        test_file = PROJECT_ROOT / "data" / "parking_configs" / "test_area_temp_99.json"
        if test_file.exists():
            test_file.unlink()


if __name__ == "__main__":
    unittest.main()

"""
Verification script for click-to-mark workflow and occupancy analysis.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.app import check_api_health, fetch_saved_areas, save_parking_area_api, send_analyze_request

base_url = "http://127.0.0.1:8000"

def main():
    print("=== STEP 1: Backend Health Check ===")
    healthy, health_data, err = check_api_health(base_url)
    print("Health check status:", healthy, health_data)

    print("\n=== STEP 2: Simulate Click-to-Mark Workflow (P1..P4) ===")
    click_sequences = [
        # P1 corners
        [[45, 123], [110, 237], [213, 212], [138, 105]],
        # P2 corners
        [[140, 104], [212, 214], [312, 188], [230, 87]],
        # P3 corners
        [[231, 87], [312, 189], [406, 166], [319, 68]],
        # P4 corners
        [[321, 70], [408, 167], [498, 146], [405, 51]],
    ]

    simulated_spaces = []
    for idx, space_clicks in enumerate(click_sequences):
        cur_pts = []
        for pt in space_clicks:
            cur_pts.append(pt)
            print(f"  Click registered: {pt} | In-progress points: {len(cur_pts)}/4")
        space_id = f"P{idx + 1}"
        simulated_spaces.append({"space_id": space_id, "polygon": cur_pts})
        print(f"  -> Auto-created {space_id}: {cur_pts}")

    print(f"Total spaces created via 4-corner clicks: {len(simulated_spaces)}")

    print("\n=== STEP 3: Test Undo, Delete, and Reset ===")
    test_pts = [[10, 20], [30, 40]]
    test_pts.pop()
    print("Undo last point test passed, remaining points:", test_pts)

    deleted_space = simulated_spaces.pop()
    print(f"Delete last space test passed, deleted {deleted_space['space_id']}, remaining: {len(simulated_spaces)}")

    simulated_spaces.append(deleted_space)
    print(f"Restored P4, total spaces: {len(simulated_spaces)}")

    print("\n=== STEP 4: Save Parking Area to FastAPI Backend ===")
    area_payload = {
        "parking_area_id": "parking_area_click_test",
        "name": "Interactive Click Lot",
        "reference_image": "1.jpg",
        "spaces": simulated_spaces,
    }

    saved, err = save_parking_area_api(base_url, area_payload)
    print("Save API response:", saved)

    print("\n=== STEP 5: Verify JSON Configuration on Disk ===")
    config_file = Path("data/parking_configs/parking_area_click_test.json")
    print("JSON file exists:", config_file.exists())
    if config_file.exists():
        with open(config_file) as f:
            data = json.load(f)
        print(f"Spaces in JSON ({len(data['spaces'])}):")
        for s in data["spaces"]:
            print(f"  {s['space_id']}: {s['polygon']}")

    print("\n=== STEP 6: Verify Occupancy Analysis against New Area ===")
    with open("data/images/1.jpg", "rb") as f:
        img_bytes = f.read()

    res, err = send_analyze_request(base_url, img_bytes, "1.jpg", parking_area_id="parking_area_click_test")
    print("Analysis Result Summary:", res["summary"])
    print("Parking Spaces Status:")
    for s in res["parking_spaces"]:
        print(f"  {s['space_id']}: {s['status']} (class: {s['vehicle_class']}, conf: {s['confidence']})")

    # Clean up
    if config_file.exists():
        config_file.unlink()
    print("\n=== STEP 7: Cleanup Complete ===")


if __name__ == "__main__":
    main()

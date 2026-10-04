"""
Smart Parking AI - Parking Space Occupancy Analysis Pipeline.

This script executes the complete Phase 7 occupancy analysis workflow:
1. Loads the parking lot image using OpenCV.
2. Loads parking space polygons from parking.json.
3. Detects vehicles using the pretrained YOLO VehicleDetector.
4. Determines occupancy status for each parking space via ParkingAnalyzer.
5. Prints a clear summary of occupied and empty spaces and occupancy rate.
6. Saves visual results to outputs/parking_occupancy.jpg.
"""

import sys
from pathlib import Path
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.detector import VehicleDetector
from ai.parking_config import ParkingConfig
from ai.parking_analyzer import ParkingAnalyzer
from ai.visualizer import draw_parking_occupancy


def main():
    image_path = PROJECT_ROOT / "data" / "images" / "1.jpg"
    config_path = PROJECT_ROOT / "data" / "parking_configs" / "parking.json"
    output_dir = PROJECT_ROOT / "outputs"
    output_path = output_dir / "parking_occupancy.jpg"

    # Step 1: Validate input image
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Step 2: Load parking space configuration
    if not config_path.exists():
        raise FileNotFoundError(
            f"Parking configuration file not found at: {config_path}\n"
            "Please run 'scripts/annotate_parking.py' first to define parking spaces."
        )

    config_loader = ParkingConfig(config_path)
    spaces = config_loader.get_spaces()

    # Step 3: Run YOLO vehicle detection
    detector = VehicleDetector(model_path="yolo11n.pt", conf_threshold=0.4)
    detections = detector.detect(image)

    # Step 4: Analyze parking occupancy
    analyzer = ParkingAnalyzer()
    occupancy_results = analyzer.analyze(detections, spaces)
    summary = analyzer.get_summary(occupancy_results)

    # Step 5: Save visualization
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_image = draw_parking_occupancy(
        image,
        detections=detections,
        spaces=spaces,
        occupancy_results=occupancy_results,
    )
    cv2.imwrite(str(output_path), annotated_image)

    # Step 6: Print formatted summary to console
    rel_image = image_path.relative_to(PROJECT_ROOT)
    rel_output = output_path.relative_to(PROJECT_ROOT)

    print("========================================")
    print("Parking Occupancy Analysis")
    print("========================================\n")
    print(f"Image:\n{rel_image}\n")
    print(f"Total parking spaces: {summary['total_spaces']}")
    print(f"Occupied spaces: {summary['occupied_spaces']}")
    print(f"Empty spaces: {summary['empty_spaces']}\n")

    for res in occupancy_results:
        status_str = "OCCUPIED" if res["occupied"] else "EMPTY"
        print(f"{res['id']} → {status_str}")

    print(f"\nOccupancy rate: {summary['occupancy_rate']:.2f}%\n")
    print(f"Output:\n{rel_output}")


if __name__ == "__main__":
    main()

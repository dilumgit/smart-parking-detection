"""
Smart Parking AI - Phase 4 OpenCV Image Processing & Visualization Pipeline.

This script executes the complete image processing workflow:
1. Loads an image from disk using OpenCV.
2. Runs vehicle detection using VehicleDetector (YOLO).
3. Receives structured vehicle detections.
4. Visualizes bounding boxes, classes, and confidence using draw_detections.
5. Saves the annotated output to outputs/annotated_parking.jpg.
6. Prints a clean detection summary to the console.
"""

import os
import sys
from pathlib import Path
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.detector import VehicleDetector
from ai.visualizer import draw_detections


def main():
    image_path = PROJECT_ROOT / "data" / "images" / "parking.jpg"
    output_dir = PROJECT_ROOT / "outputs"
    output_path = output_dir / "annotated_parking.jpg"

    # Step 1: Validate file existence and load with OpenCV
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Step 2: Initialize Detector and run YOLO inference
    detector = VehicleDetector(model_path="yolo11n.pt", conf_threshold=0.4)
    detections = detector.detect(image)

    # Step 3: Draw detections onto the image using OpenCV
    annotated_image = draw_detections(image, detections)

    # Step 4: Ensure output folder exists and save result
    output_dir.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), annotated_image)
    if not success:
        raise RuntimeError(f"Failed to write annotated image to: {output_path}")

    # Step 5: Print detection summary
    rel_image_path = image_path.relative_to(PROJECT_ROOT)
    rel_output_path = output_path.relative_to(PROJECT_ROOT)

    print("====================================")
    print("Vehicle Detection Result")
    print("====================================\n")
    print(f"Image:\n{rel_image_path}\n")
    print(f"Vehicles detected: {len(detections)}\n")

    for idx, det in enumerate(detections, start=1):
        class_name = det.get("class") or det.get("class_name")
        conf = det.get("confidence", 0.0)
        print(f"{idx}. {class_name:<11} confidence={conf:.2f}")

    print(f"\nAnnotated image:\n{rel_output_path}")


if __name__ == "__main__":
    main()

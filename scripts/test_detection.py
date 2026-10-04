"""
Smart Parking AI - Test YOLO Vehicle Detection Script.

This script loads a pretrained YOLO model, performs vehicle detection
on a sample parking lot image, displays the detections in the console,
and saves the annotated output image.
"""

import os
import sys
from pathlib import Path
import cv2

# Add workspace root to Python path to ensure clean imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.detector import VehicleDetector


def main():
    print("========================================")
    print(" Smart Parking AI - Vehicle Detection   ")
    print("========================================\n")

    model_path = "yolo11n.pt"
    image_path = PROJECT_ROOT / "data" / "images" / "parking.jpg"
    output_dir = PROJECT_ROOT / "outputs"
    output_path = output_dir / "detection_result.jpg"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate image exists
    if not image_path.exists():
        print(f"Error: Test image not found at {image_path}")
        print("Please place a test image at 'data/images/parking.jpg'.")
        sys.exit(1)

    print(f"Loading YOLO model ({model_path})...")
    try:
        detector = VehicleDetector(model_path=model_path, conf_threshold=0.4)
    except Exception as exc:
        print(f"Error loading model: {exc}")
        sys.exit(1)

    print(f"\nImage: {image_path.relative_to(PROJECT_ROOT)}")

    # Run detection
    try:
        detections = detector.detect(str(image_path))
    except Exception as exc:
        print(f"Error during detection: {exc}")
        sys.exit(1)

    print("\nDetected vehicles:")
    if detections:
        for idx, det in enumerate(detections, start=1):
            class_name = det["class_name"]
            confidence = det["confidence"]
            bbox = det["bbox"]
            print(f"{idx}. {class_name:<12} confidence={confidence:.2f}  bbox={bbox}")
    else:
        print("No vehicles detected with confidence above threshold.")

    print(f"\nTotal vehicles: {len(detections)}")

    # Generate and save annotated image
    try:
        annotated_image = detector.annotate_image(str(image_path), detections)
        cv2.imwrite(str(output_path), annotated_image)
        print(f"\nAnnotated image saved to:")
        print(f"{output_path.relative_to(PROJECT_ROOT)}")
    except Exception as exc:
        print(f"Error generating/saving annotated image: {exc}")
        sys.exit(1)

    print("\nDetection test completed successfully.")


if __name__ == "__main__":
    main()

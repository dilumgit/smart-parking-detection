"""
Smart Parking AI - YOLO Vehicle Detection Module.

This module provides the VehicleDetector class for identifying vehicles
(cars, motorcycles, buses, trucks) in images using pretrained YOLO models.
"""

import os
from typing import Any, Dict, List, Set, Union
import cv2
import numpy as np
from ultralytics import YOLO

# Common vehicle classes from the COCO dataset
VEHICLE_CLASSES: Set[str] = {
    "car",
    "motorcycle",
    "bus",
    "truck",
}


class VehicleDetector:
    """
    Vehicle detector using pretrained YOLO models.
    Filters detections to target vehicle classes and returns structured bounding boxes.
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        vehicle_classes: Set[str] = VEHICLE_CLASSES,
        conf_threshold: float = 0.5,
    ) -> None:
        """
        Initialize the VehicleDetector.

        Args:
            model_path: Path to model weights or model name (e.g. 'yolo11n.pt', 'yolov8n.pt').
            vehicle_classes: Set of vehicle class names to detect.
            conf_threshold: Default confidence threshold (between 0.0 and 1.0).
        """
        self.model_path = model_path
        self.vehicle_classes = set(vehicle_classes)
        self.conf_threshold = conf_threshold
        try:
            self.model = YOLO(self.model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO model from '{self.model_path}': {exc}"
            ) from exc

    def detect(
        self,
        image: Union[str, np.ndarray],
        conf_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect vehicles in an image file or numpy array.

        Args:
            image: File path to image or numpy BGR image array.
            conf_threshold: Optional confidence override for this detection run.

        Returns:
            List[Dict[str, Any]]: List of vehicle detections, each containing:
                - class_name (str): Vehicle class name (e.g. 'car')
                - class (str): Vehicle class name alias
                - confidence (float): Confidence score between 0.0 and 1.0
                - bbox (List[int]): Bounding box [x1, y1, x2, y2]
        """
        threshold = self.conf_threshold if conf_threshold is None else conf_threshold

        # Validate file path input
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")

        # Run inference
        results = self.model(image, verbose=False)
        detections: List[Dict[str, Any]] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                conf = float(box.conf[0].item())
                if conf < threshold:
                    continue

                cls_id = int(box.cls[0].item())
                class_name = self.model.names.get(cls_id, "")

                if class_name in self.vehicle_classes:
                    xyxy = box.xyxy[0].tolist()
                    bbox = [int(round(coord)) for coord in xyxy]

                    detections.append({
                        "class_name": class_name,
                        "class": class_name,
                        "confidence": round(conf, 2),
                        "bbox": bbox,
                    })

        return detections

    def annotate_image(
        self,
        image: Union[str, np.ndarray],
        detections: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Draw vehicle bounding boxes, class names, and confidence scores on an image.

        Args:
            image: Image path or numpy BGR image array.
            detections: List of detection dictionaries from `detect()`.

        Returns:
            np.ndarray: Annotated BGR image.
        """
        from ai.visualizer import draw_detections

        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            loaded_img = cv2.imread(image)
            if loaded_img is None:
                raise ValueError(f"Could not decode image at path: {image}")
        else:
            loaded_img = image

        return draw_detections(loaded_img, detections)


# Backward compatibility alias
Detector = VehicleDetector

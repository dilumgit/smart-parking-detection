"""
Smart Parking AI - OpenCV Visualization Module.

This module provides OpenCV-based drawing and annotation utilities
for visualizing vehicle detections and parking space occupancy on images.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

# Distinct BGR color palette for vehicle classes
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "car": (0, 220, 0),         # Vibrant Green
    "truck": (255, 120, 0),      # Cyan/Blue
    "bus": (0, 140, 255),        # Orange
    "motorcycle": (200, 0, 220), # Magenta
}
DEFAULT_COLOR: Tuple[int, int, int] = (0, 220, 0)

# Colors for parking space status (BGR)
COLOR_OCCUPIED: Tuple[int, int, int] = (0, 0, 225)    # Red / Coral
COLOR_EMPTY: Tuple[int, int, int] = (0, 200, 0)       # Green


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    box_thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 1,
) -> np.ndarray:
    """
    Draw vehicle bounding boxes, class labels, and confidence scores onto an image.

    Args:
        image: Numpy ndarray (BGR image from cv2.imread).
        detections: List of detection dictionaries, each containing:
            - 'bbox' or 'bounding_box': [x1, y1, x2, y2]
            - 'class' or 'class_name': vehicle class name (e.g. 'car')
            - 'confidence' or 'score': float confidence score (0.0 to 1.0)
        box_thickness: Line thickness for the bounding box.
        font_scale: Scale for cv2.putText.
        font_thickness: Line thickness for text.

    Returns:
        np.ndarray: A copy of the input image with rendered annotations.
    """
    if image is None:
        raise ValueError("Input image cannot be None.")

    annotated = image.copy()
    if not detections:
        return annotated

    img_h, img_w = annotated.shape[:2]

    for det in detections:
        bbox = det.get("bbox") or det.get("bounding_box")
        if bbox is None or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        class_name = str(det.get("class") or det.get("class_name") or "vehicle")
        confidence = det.get("confidence") if "confidence" in det else det.get("score", 0.0)
        conf_val = float(confidence) if confidence is not None else 0.0

        color = CLASS_COLORS.get(class_name.lower(), DEFAULT_COLOR)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

        # Label: "car 0.94"
        label = f"{class_name} {conf_val:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX

        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        if y1 - text_h - baseline - 4 >= 0:
            lbl_y1 = y1 - text_h - baseline - 4
            lbl_y2 = y1
        else:
            lbl_y1 = y1
            lbl_y2 = y1 + text_h + baseline + 4

        lbl_x1 = x1
        lbl_x2 = min(x1 + text_w + 6, img_w)

        cv2.rectangle(annotated, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), color, -1)
        text_y = lbl_y2 - baseline - 2
        cv2.putText(
            annotated,
            label,
            (lbl_x1 + 3, text_y),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return annotated


def draw_parking_occupancy(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    spaces: List[Dict[str, Any]],
    occupancy_results: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Render parking space polygons, occupancy statuses (OCCUPIED / EMPTY),
    and detected vehicle bounding boxes.

    Args:
        image: Original input BGR image.
        detections: List of detected vehicles.
        spaces: List of parking space polygon dicts with 'id' and 'polygon'.
        occupancy_results: List of occupancy dicts with 'id', 'occupied', 'vehicle_count'.

    Returns:
        np.ndarray: Annotated output image.
    """
    if image is None:
        raise ValueError("Input image cannot be None.")

    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]

    # Map occupancy status by space id
    occ_map = {res["id"]: res.get("occupied", False) for res in occupancy_results}

    # 1. Draw parking space polygons and status labels
    for space in spaces:
        space_id = space["id"]
        is_occupied = occ_map.get(space_id, False)
        color = COLOR_OCCUPIED if is_occupied else COLOR_EMPTY
        status_text = "OCCUPIED" if is_occupied else "EMPTY"

        pts = np.array(space["polygon"], dtype=np.int32).reshape((-1, 1, 2))

        # Draw transparent fill overlay
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [pts], color=color)
        alpha = 0.20 if is_occupied else 0.12
        cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)

        # Draw polygon border
        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

        # Calculate centroid for label placement
        M = cv2.moments(pts)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = space["polygon"][0]

        # Draw space ID and status badge
        badge_text = f"{space_id}: {status_text}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        font_thickness = 1
        (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, font_thickness)

        bx1 = max(0, cX - tw // 2 - 4)
        by1 = max(0, cY - th // 2 - 4)
        bx2 = min(img_w, bx1 + tw + 8)
        by2 = min(img_h, by1 + th + 8)

        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 1)
        cv2.putText(
            annotated,
            badge_text,
            (bx1 + 4, by2 - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    # 2. Draw vehicle detections (bounding boxes and confidence labels)
    annotated = draw_detections(annotated, detections)

    return annotated

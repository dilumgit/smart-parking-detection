"""
Smart Parking AI - Interactive Parking Space Annotation Tool.

This script allows users to manually define parking spaces on an image
by clicking the 4 corner points of each space using OpenCV GUI.

Controls:
    - Left Click: Select a corner point of a parking space (4 points per space).
    - R: Reset/clear currently selected points for the in-progress space.
    - U: Undo the most recently completed parking space.
    - Q / ESC: Finish, save configuration to JSON, save visual output, and exit.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.parking_config import ParkingConfig


# Global state for annotation
class AnnotatorState:
    def __init__(self, original_image: np.ndarray, image_rel_path: str):
        self.original_image = original_image
        self.image_rel_path = image_rel_path
        self.spaces: List[Dict[str, Any]] = []
        self.current_points: List[Tuple[int, int]] = []
        self.window_name = "Smart Parking AI - Manual Annotation Tool"

    def redraw(self) -> np.ndarray:
        """Render completed polygons and in-progress points on a fresh copy of the image."""
        canvas = self.original_image.copy()
        img_h, img_w = canvas.shape[:2]

        # 1. Draw all completed parking spaces
        for space in self.spaces:
            space_id = space["id"]
            pts = np.array(space["polygon"], dtype=np.int32).reshape((-1, 1, 2))

            # Draw polygon boundary
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 220, 0), thickness=2)

            # Draw semi-transparent fill for better visibility
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color=(0, 220, 0))
            cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)

            # Compute centroid or top point for ID label
            M = cv2.moments(pts)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = space["polygon"][0]

            # Label badge
            label = space_id
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            lx1 = max(0, cX - tw // 2 - 4)
            ly1 = max(0, cY - th // 2 - 4)
            lx2 = min(img_w, lx1 + tw + 8)
            ly2 = min(img_h, ly1 + th + 8)

            cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                label,
                (lx1 + 4, ly2 - 4),
                font,
                font_scale,
                (0, 255, 255),
                thickness,
                lineType=cv2.LINE_AA,
            )

        # 2. Draw current in-progress points and connecting lines
        for i, pt in enumerate(self.current_points):
            # Point circle
            cv2.circle(canvas, pt, 4, (0, 0, 255), -1)
            cv2.circle(canvas, pt, 6, (255, 255, 255), 1)
            # Connecting line to previous point
            if i > 0:
                cv2.line(canvas, self.current_points[i - 1], pt, (0, 165, 255), 2)

        # 3. Draw instructions on the top bar
        info_bar = canvas.copy()
        cv2.rectangle(info_bar, (0, 0), (img_w, 28), (30, 30, 30), -1)
        cv2.addWeighted(info_bar, 0.8, canvas, 0.2, 0, canvas)

        next_id = f"P{len(self.spaces) + 1}"
        pts_count = len(self.current_points)
        status_text = f"Spaces: {len(self.spaces)} | Defining {next_id} (Point {pts_count}/4) | R: Reset | U: Undo | Q: Save & Exit"
        cv2.putText(
            canvas,
            status_text,
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )

        return canvas


def on_mouse_click(event: int, x: int, y: int, flags: int, param: Any) -> None:
    """OpenCV mouse callback handler."""
    state: AnnotatorState = param

    if event == cv2.EVENT_LBUTTONDOWN:
        state.current_points.append((int(x), int(y)))
        print(f"  Point {len(state.current_points)}/4 selected: ({x}, {y})")

        # When 4 points are selected, complete the parking space
        if len(state.current_points) == 4:
            space_id = f"P{len(state.spaces) + 1}"
            polygon = [[pt[0], pt[1]] for pt in state.current_points]
            state.spaces.append({"id": space_id, "polygon": polygon})
            print(f"-> Completed {space_id}: {polygon}\n")
            state.current_points.clear()

        # Update display
        updated_frame = state.redraw()
        cv2.imshow(state.window_name, updated_frame)


def main():
    print("==================================================")
    print(" Smart Parking AI - Manual Parking Annotation     ")
    print("==================================================")
    print("Instructions:")
    print("  - Left Click : Click 4 corners for each parking space (P1, P2, ...)")
    print("  - R          : Reset the in-progress 4-point selection")
    print("  - U          : Undo the last completed parking space")
    print("  - Q or ESC   : Save configuration and exit")
    print("==================================================\n")

    image_path = PROJECT_ROOT / "data" / "images" / "1.jpg"
    config_path = PROJECT_ROOT / "data" / "parking_configs" / "parking.json"
    output_dir = PROJECT_ROOT / "outputs"
    vis_output_path = output_dir / "parking_spaces.jpg"

    if not image_path.exists():
        print(f"Error: Target image not found at {image_path}")
        sys.exit(1)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Failed to load image from {image_path}")
        sys.exit(1)

    rel_image_path = str(image_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    state = AnnotatorState(image, rel_image_path)

    # Initialize window and mouse callback
    cv2.namedWindow(state.window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(state.window_name, on_mouse_click, state)

    # Initial render
    cv2.imshow(state.window_name, state.redraw())

    print(f"Annotating image: {rel_image_path}")
    print("Click 4 corners for each space...\n")

    while True:
        key = cv2.waitKey(20) & 0xFF

        # Q or ESC -> Save and exit
        if key in (ord("q"), ord("Q"), 27):
            print("\nSaving configuration...")
            break

        # R -> Reset in-progress points
        elif key in (ord("r"), ord("R")):
            if state.current_points:
                print("Reset current in-progress points.")
                state.current_points.clear()
                cv2.imshow(state.window_name, state.redraw())

        # U -> Undo last space
        elif key in (ord("u"), ord("U")):
            if state.spaces:
                removed = state.spaces.pop()
                print(f"Undid space {removed['id']}.")
                cv2.imshow(state.window_name, state.redraw())
            elif state.current_points:
                state.current_points.clear()
                print("Cleared in-progress points.")
                cv2.imshow(state.window_name, state.redraw())

    cv2.destroyAllWindows()

    # Save to JSON via ParkingConfig
    config_loader = ParkingConfig(config_path)
    config_loader.save(rel_image_path, state.spaces)
    print(f"Saved {len(state.spaces)} parking space(s) to:")
    print(f"  {config_path.relative_to(PROJECT_ROOT)}")

    # Save visual rendering to outputs/parking_spaces.jpg
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_output = state.redraw()
    cv2.imwrite(str(vis_output_path), annotated_output)
    print(f"Saved visualization to:")
    print(f"  {vis_output_path.relative_to(PROJECT_ROOT)}")
    print("\nAnnotation completed successfully.")


if __name__ == "__main__":
    main()

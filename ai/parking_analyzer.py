"""
Smart Parking AI - Parking Occupancy Analyzer Module.

This module determines the occupancy status (OCCUPIED vs EMPTY) of parking spaces
by checking whether detected vehicle bounding-box center points fall within
parking-space polygons using OpenCV geometric point-in-polygon tests.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


class ParkingAnalyzer:
    """
    Analyzes parking space occupancy based on vehicle detections and polygon definitions.
    """

    @staticmethod
    def calculate_center(bbox: List[Union[int, float]]) -> Tuple[float, float]:
        """
        Calculate the center point (x, y) of a bounding box [x1, y1, x2, y2].

        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2].

        Returns:
            Tuple[float, float]: Center coordinate (center_x, center_y).
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        return (center_x, center_y)

    def analyze(
        self,
        detections: List[Dict[str, Any]],
        spaces: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Evaluate occupancy for every parking space.

        Args:
            detections: List of vehicle detections with 'bbox', 'class'/'class_name', 'confidence'.
            spaces: List of parking space dictionaries with 'id' and 'polygon'.

        Returns:
            List[Dict[str, Any]]: Occupancy results for each space:
                - id (str): Parking space identifier
                - occupied (bool): True if at least one vehicle center is inside
                - vehicle_count (int): Number of vehicle centers inside the polygon
                - vehicles (List[Dict]): List of matching detection objects
        """
        # Precompute vehicle centers for all detections
        vehicle_centers: List[Tuple[Tuple[float, float], Dict[str, Any]]] = []
        for det in detections:
            bbox = det.get("bbox") or det.get("bounding_box")
            if bbox is not None and len(bbox) == 4:
                center = self.calculate_center(bbox)
                vehicle_centers.append((center, det))

        results: List[Dict[str, Any]] = []

        for space in spaces:
            space_id = space["id"]
            polygon_pts = np.array(space["polygon"], dtype=np.float32)

            matched_vehicles: List[Dict[str, Any]] = []

            for center, det in vehicle_centers:
                # cv2.pointPolygonTest returns >= 0 if the point is inside or on the contour
                dist = cv2.pointPolygonTest(polygon_pts, center, measureDist=False)
                if dist >= 0:
                    matched_vehicles.append(det)

            count = len(matched_vehicles)
            results.append({
                "id": space_id,
                "occupied": count > 0,
                "vehicle_count": count,
                "vehicles": matched_vehicles,
            })

        return results

    @staticmethod
    def calculate_occupancy_rate(results: List[Dict[str, Any]]) -> float:
        """
        Calculate the percentage of occupied parking spaces.

        Args:
            results: List of occupancy results from `analyze()`.

        Returns:
            float: Occupancy percentage between 0.0 and 100.0.
        """
        total = len(results)
        if total == 0:
            return 0.0
        occupied = sum(1 for r in results if r.get("occupied", False))
        return round((occupied / total) * 100.0, 2)

    def get_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get high-level summary metrics of the parking lot occupancy.

        Args:
            results: List of occupancy results from `analyze()`.

        Returns:
            Dict[str, Any]: Summary dictionary with counts and rate.
        """
        total = len(results)
        occupied = sum(1 for r in results if r.get("occupied", False))
        empty = total - occupied
        rate = self.calculate_occupancy_rate(results)

        return {
            "total_spaces": total,
            "occupied_spaces": occupied,
            "empty_spaces": empty,
            "occupancy_rate": rate,
            "spaces": results,
        }

"""AI Package."""
from ai.detector import VehicleDetector, VEHICLE_CLASSES
from ai.visualizer import draw_detections, draw_parking_occupancy, CLASS_COLORS
from ai.parking_config import ParkingConfig, ParkingAreaManager
from ai.parking_analyzer import ParkingAnalyzer

__all__ = [
    "VehicleDetector",
    "VEHICLE_CLASSES",
    "draw_detections",
    "draw_parking_occupancy",
    "CLASS_COLORS",
    "ParkingConfig",
    "ParkingAreaManager",
    "ParkingAnalyzer",
]

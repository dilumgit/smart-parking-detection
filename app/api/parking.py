"""
Smart Parking AI - Parking API Router.

Provides endpoints for:
- Analyzing parking space occupancy from uploaded images.
- Managing saved parking area configurations (creating, listing, and retrieving).
"""

import os
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from ai.detector import VehicleDetector
from ai.parking_config import ParkingConfig, ParkingAreaManager
from ai.parking_analyzer import ParkingAnalyzer
from app.schemas.parking import (
    ParkingAnalysisResponse,
    ParkingSummary,
    ParkingSpaceResult,
    ParkingAreaCreate,
    ParkingAreaResponse,
    ParkingAreaSummary,
    ParkingSpaceDefinition,
)

router = APIRouter(prefix="/api/v1/parking", tags=["parking"])

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "data" / "parking_configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "parking.json"

# Reusable detector and area manager instances
_detector: Optional[VehicleDetector] = None
_area_manager: Optional[ParkingAreaManager] = None


def get_detector() -> VehicleDetector:
    """Lazy-load and reuse the YOLO vehicle detector instance."""
    global _detector
    if _detector is None:
        _detector = VehicleDetector(model_path="yolo11n.pt", conf_threshold=0.4)
    return _detector


def get_area_manager() -> ParkingAreaManager:
    """Get or initialize the ParkingAreaManager instance."""
    global _area_manager
    if _area_manager is None:
        _area_manager = ParkingAreaManager(CONFIG_DIR)
    return _area_manager


# =====================================================================
# SAVED PARKING AREAS ENDPOINTS
# =====================================================================

@router.get(
    "/areas",
    response_model=List[ParkingAreaSummary],
    status_code=status.HTTP_200_OK,
    summary="List Saved Parking Areas",
    description="Returns a list of all configured and saved parking area summaries.",
)
async def list_parking_areas() -> List[ParkingAreaSummary]:
    """List all saved parking area configurations."""
    manager = get_area_manager()
    areas = manager.list_areas()
    return [
        ParkingAreaSummary(
            parking_area_id=a["parking_area_id"],
            name=a["name"],
            reference_image=a["reference_image"],
            total_spaces=a["total_spaces"],
        )
        for a in areas
    ]


@router.get(
    "/areas/{parking_area_id}",
    response_model=ParkingAreaResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Saved Parking Area by ID",
    description="Loads and returns the full polygon and space configuration for a specific parking area.",
)
async def get_parking_area(parking_area_id: str) -> ParkingAreaResponse:
    """Retrieve details and polygons of a specific parking area."""
    manager = get_area_manager()
    try:
        area_data = manager.get_area(parking_area_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Parking area '{parking_area_id}' not found.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load parking area: {str(exc)}",
        )

    spaces = [
        ParkingSpaceDefinition(
            space_id=s.get("space_id") or s.get("id"),
            polygon=s["polygon"],
        )
        for s in area_data.get("spaces", [])
    ]

    return ParkingAreaResponse(
        parking_area_id=area_data["parking_area_id"],
        name=area_data.get("name", area_data["parking_area_id"]),
        reference_image=str(area_data.get("reference_image") or area_data.get("image") or "1.jpg"),
        total_spaces=len(spaces),
        spaces=spaces,
    )


@router.post(
    "/areas",
    response_model=ParkingAreaResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Saved Parking Area",
    description="Validates and persists a new parking area configuration to disk.",
)
async def create_parking_area(area_in: ParkingAreaCreate) -> ParkingAreaResponse:
    """Create a new named parking area configuration."""
    manager = get_area_manager()
    area_dict = area_in.model_dump()

    try:
        saved = manager.create_area(area_dict)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save parking area: {str(exc)}",
        )

    spaces = [
        ParkingSpaceDefinition(space_id=s["space_id"], polygon=s["polygon"])
        for s in saved["spaces"]
    ]

    return ParkingAreaResponse(
        parking_area_id=saved["parking_area_id"],
        name=saved["name"],
        reference_image=saved["reference_image"],
        total_spaces=len(spaces),
        spaces=spaces,
    )


# =====================================================================
# OCCUPANCY ANALYSIS ENDPOINT
# =====================================================================

@router.post(
    "/analyze",
    response_model=ParkingAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Parking Occupancy",
    description=(
        "Uploads an image of a parking lot, detects vehicles using YOLO, "
        "and computes the occupancy status (OCCUPIED/EMPTY) for each defined parking space. "
        "Optionally accepts parking_area_id to analyze against a specific saved area layout."
    ),
)
async def analyze_parking_image(
    file: UploadFile = File(..., description="Image file of the parking lot (JPEG, PNG, etc.)"),
    parking_area_id: Optional[str] = Query(None, description="Optional ID of saved parking area to use"),
) -> ParkingAnalysisResponse:
    """
    Process an uploaded parking lot image and return detailed occupancy results.
    """
    # 1. Validate file presence
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided in upload request.",
        )

    # 2. Read file contents into memory safely
    try:
        contents = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(exc)}",
        )

    if not contents or len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # 3. Decode image from memory using OpenCV
    try:
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error decoding image file: {str(exc)}",
        )

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file or unsupported image format.",
        )

    # 4. Load parking spaces configuration (from specific area ID or default config)
    manager = get_area_manager()
    if parking_area_id:
        try:
            area_data = manager.get_area(parking_area_id)
            spaces = [
                {"id": s.get("space_id") or s.get("id"), "polygon": s["polygon"]}
                for s in area_data.get("spaces", [])
            ]
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Parking area '{parking_area_id}' not found.",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to load parking area '{parking_area_id}': {str(exc)}",
            )
    else:
        # Fallback to default parking.json or parking_area_001.json
        if DEFAULT_CONFIG_PATH.exists():
            try:
                parking_config = ParkingConfig(DEFAULT_CONFIG_PATH)
                spaces = parking_config.get_spaces()
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to load parking configuration: {str(exc)}",
                )
        else:
            try:
                area_data = manager.get_area("parking_area_001")
                spaces = [
                    {"id": s.get("space_id") or s.get("id"), "polygon": s["polygon"]}
                    for s in area_data.get("spaces", [])
                ]
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Parking space configuration not found. Please define parking spaces first.",
                )

    # 5. Run YOLO vehicle detection
    detector = get_detector()
    try:
        detections = detector.detect(image)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vehicle detection failed: {str(exc)}",
        )

    # 6. Analyze parking space occupancy
    analyzer = ParkingAnalyzer()
    occupancy_results = analyzer.analyze(detections, spaces)
    summary_data = analyzer.get_summary(occupancy_results)

    # 7. Build structured response
    parking_space_results = []
    for res in occupancy_results:
        is_occupied = res.get("occupied", False)
        vehicles = res.get("vehicles", [])

        vehicle_class = None
        confidence = None
        if is_occupied and vehicles:
            primary_vehicle = vehicles[0]
            vehicle_class = primary_vehicle.get("class") or primary_vehicle.get("class_name")
            confidence = primary_vehicle.get("confidence")

        parking_space_results.append(
            ParkingSpaceResult(
                space_id=res["id"],
                status="OCCUPIED" if is_occupied else "EMPTY",
                vehicle_count=res.get("vehicle_count", 0),
                vehicle_class=vehicle_class,
                confidence=confidence,
            )
        )

    return ParkingAnalysisResponse(
        success=True,
        filename=file.filename,
        summary=ParkingSummary(
            total_spaces=summary_data["total_spaces"],
            occupied_spaces=summary_data["occupied_spaces"],
            empty_spaces=summary_data["empty_spaces"],
            occupancy_rate=summary_data["occupancy_rate"],
        ),
        parking_spaces=parking_space_results,
    )

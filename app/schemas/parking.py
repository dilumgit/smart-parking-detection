"""
Smart Parking AI - Pydantic Schemas for Parking API.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for health check endpoint response."""
    status: str = Field(..., description="Service health status", examples=["healthy"])
    service: str = Field(..., description="Service identifier", examples=["smart-parking-ai"])


class ParkingSummary(BaseModel):
    """Summary metrics of parking lot occupancy."""
    total_spaces: int = Field(..., description="Total defined parking spaces", examples=[4])
    occupied_spaces: int = Field(..., description="Number of occupied spaces", examples=[1])
    empty_spaces: int = Field(..., description="Number of empty spaces", examples=[3])
    occupancy_rate: float = Field(..., description="Percentage of occupied spaces", examples=[25.0])


class ParkingSpaceResult(BaseModel):
    """Occupancy status and details for an individual parking space."""
    space_id: str = Field(..., description="Parking space identifier", examples=["P1"])
    status: str = Field(..., description="Occupancy status: OCCUPIED or EMPTY", examples=["OCCUPIED"])
    vehicle_count: int = Field(..., description="Number of vehicles detected in this space", examples=[1])
    vehicle_class: Optional[str] = Field(None, description="Primary detected vehicle class", examples=["car"])
    confidence: Optional[float] = Field(None, description="Detection confidence score", examples=[0.47])


class ParkingAnalysisResponse(BaseModel):
    """Complete response model for parking occupancy analysis endpoint."""
    success: bool = Field(True, description="Indicates whether the request was successful")
    filename: str = Field(..., description="Name of the analyzed image file", examples=["1.jpg"])
    summary: ParkingSummary = Field(..., description="Overall occupancy summary metrics")
    parking_spaces: List[ParkingSpaceResult] = Field(..., description="List of all parking space results")


# --- Saved Parking Area Schemas ---

class ParkingSpaceDefinition(BaseModel):
    """Definition of an individual parking space polygon."""
    space_id: str = Field(..., description="Unique space identifier within the area", examples=["P1"])
    polygon: List[List[Union[int, float]]] = Field(
        ...,
        description="4-point polygon corner coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]",
        examples=[[[45, 123], [110, 237], [213, 212], [138, 105]]],
    )


class ParkingAreaCreate(BaseModel):
    """Request schema for creating a new saved parking area."""
    parking_area_id: str = Field(..., description="Unique parking area identifier", examples=["parking_area_001"])
    name: str = Field(..., description="Human-readable name of the parking area", examples=["Main Parking Lot"])
    reference_image: Optional[str] = Field("1.jpg", description="Reference parking image filename or path", examples=["1.jpg"])
    spaces: List[ParkingSpaceDefinition] = Field(..., description="List of defined parking space polygons")


class ParkingAreaResponse(BaseModel):
    """Response schema for a complete parking area configuration."""
    parking_area_id: str = Field(..., description="Unique parking area identifier", examples=["parking_area_001"])
    name: str = Field(..., description="Human-readable name of the parking area", examples=["Main Parking Lot"])
    reference_image: str = Field(..., description="Reference parking image filename", examples=["1.jpg"])
    total_spaces: int = Field(..., description="Total number of spaces defined", examples=[4])
    spaces: List[ParkingSpaceDefinition] = Field(..., description="List of defined parking space polygons")


class ParkingAreaSummary(BaseModel):
    """Summary schema for listing saved parking areas."""
    parking_area_id: str = Field(..., description="Unique parking area identifier", examples=["parking_area_001"])
    name: str = Field(..., description="Human-readable name of the parking area", examples=["Main Parking Lot"])
    reference_image: str = Field(..., description="Reference parking image filename", examples=["1.jpg"])
    total_spaces: int = Field(..., description="Total number of spaces defined", examples=[4])

"""
Smart Parking AI - Health Check Router.
"""

from fastapi import APIRouter
from app.schemas.parking import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Returns the operational status of the Smart Parking AI service.",
)
async def health_check() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(status="healthy", service="smart-parking-ai")

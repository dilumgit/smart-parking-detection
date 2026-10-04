"""
Smart Parking AI - FastAPI Application Entrypoint.

Provides a REST API for vehicle detection, parking spot mapping,
and occupancy analysis.
"""

from fastapi import FastAPI
from app.api.health import router as health_router
from app.api.parking import router as parking_router

app = FastAPI(
    title="Smart Parking Detection AI",
    description="API for real-time vehicle detection and parking occupancy analysis using YOLO and OpenCV.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Register API routers
app.include_router(health_router)
app.include_router(parking_router)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint welcoming clients and directing to API docs."""
    return {
        "message": "Smart Parking Detection AI API is running",
        "docs": "/docs",
        "health": "/health",
        "analyze": "/api/v1/parking/analyze",
    }

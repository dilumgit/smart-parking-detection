"""App API Routers Package."""
from app.api.health import router as health_router
from app.api.parking import router as parking_router

__all__ = ["health_router", "parking_router"]

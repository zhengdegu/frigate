"""Cross-camera tracking API endpoints."""

import logging
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Cross-Camera Tracking"])


def create_cross_camera_router(cross_camera_tracker) -> APIRouter:
    """Create router with reference to the cross-camera tracker instance."""

    @router.get("/api/cross_camera/tracks")
    def get_global_tracks() -> dict[str, Any]:
        """Return all active global tracks."""
        if cross_camera_tracker is None:
            return {"enabled": False, "tracks": {}}
        return {
            "enabled": True,
            "tracks": cross_camera_tracker.get_global_tracks(),
        }

    @router.get("/api/cross_camera/tracks/{global_id}")
    def get_global_track(global_id: str) -> dict[str, Any]:
        """Return a single global track by ID."""
        if cross_camera_tracker is None:
            return {"error": "Cross-camera tracking not enabled"}
        tracks = cross_camera_tracker.get_global_tracks()
        if global_id in tracks:
            return tracks[global_id]
        return {"error": f"Track {global_id} not found"}

    @router.get("/api/cross_camera/stats")
    def get_stats() -> dict[str, Any]:
        """Return cross-camera tracking statistics."""
        if cross_camera_tracker is None:
            return {"enabled": False}
        tracks = cross_camera_tracker.get_global_tracks()
        plates = sum(1 for t in tracks.values() if t.get("plate"))
        multi_cam = sum(
            1 for t in tracks.values() if len(t.get("sightings", [])) > 1
        )
        return {
            "enabled": True,
            "total_global_tracks": len(tracks),
            "tracks_with_plate": plates,
            "tracks_across_cameras": multi_cam,
        }

    @router.post("/api/cross_camera/cleanup")
    def cleanup_expired() -> dict[str, Any]:
        """Manually trigger cleanup of expired tracks."""
        if cross_camera_tracker is None:
            return {"error": "Cross-camera tracking not enabled"}
        removed = cross_camera_tracker.cleanup_expired()
        return {"removed": removed}

    return router

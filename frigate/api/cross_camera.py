"""Cross-camera tracking API endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Cross-Camera Tracking"])


def _get_tracker(request: Request) -> Optional[object]:
    """Get the CrossCameraTracker from the FastAPI app state."""
    processor = getattr(request.app, "detected_frames_processor", None)
    if processor is None:
        return None
    return getattr(processor, "cross_camera_tracker", None)


@router.get("/api/cross_camera/tracks")
def get_cross_camera_tracks(request: Request):
    """Return all active global tracks."""
    tracker = _get_tracker(request)
    if tracker is None:
        return {"success": False, "message": "Cross-camera tracking not enabled"}

    tracks = tracker.get_global_tracks()
    return {"success": True, "tracks": tracks, "count": len(tracks)}


@router.get("/api/cross_camera/tracks/{global_id}")
def get_cross_camera_track(request: Request, global_id: str):
    """Return a single global track by ID."""
    tracker = _get_tracker(request)
    if tracker is None:
        return {"success": False, "message": "Cross-camera tracking not enabled"}

    tracks = tracker.get_global_tracks()
    if global_id not in tracks:
        return {"success": False, "message": f"Track {global_id} not found"}

    return {"success": True, "track": tracks[global_id]}


@router.get("/api/cross_camera/stats")
def get_cross_camera_stats(request: Request):
    """Return cross-camera tracking statistics."""
    tracker = _get_tracker(request)
    if tracker is None:
        return {"success": False, "message": "Cross-camera tracking not enabled"}

    tracks = tracker.get_global_tracks()
    plates_known = sum(1 for t in tracks.values() if t.get("plate"))
    cameras_seen = set()
    for t in tracks.values():
        for s in t.get("sightings", []):
            cameras_seen.add(s.get("camera"))

    return {
        "success": True,
        "total_global_tracks": len(tracks),
        "tracks_with_plate": plates_known,
        "tracks_without_plate": len(tracks) - plates_known,
        "cameras_involved": sorted(cameras_seen),
    }


@router.delete("/api/cross_camera/expired")
def cleanup_expired_tracks(request: Request):
    """Manually trigger cleanup of expired global tracks."""
    tracker = _get_tracker(request)
    if tracker is None:
        return {"success": False, "message": "Cross-camera tracking not enabled"}

    removed = tracker.cleanup_expired()
    return {"success": True, "removed": removed}

"""Cross-camera tracking API endpoints."""

import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Cross-Camera Tracking"])


@router.get("/cross_camera/tracks")
def get_cross_camera_tracks():
    """Return all active global tracks."""
    from frigate.app import FrigateApp

    app = FrigateApp.current
    if app is None or not hasattr(app, "detected_frames_processor"):
        return {"success": False, "message": "Frigate not ready"}

    tracker = getattr(app.detected_frames_processor, "cross_camera_tracker", None)
    if tracker is None:
        return {"success": False, "message": "Cross-camera tracking not enabled"}

    tracks = tracker.get_global_tracks()
    return {"success": True, "tracks": tracks, "count": len(tracks)}


@router.get("/cross_camera/tracks/{global_id}")
def get_cross_camera_track(global_id: str):
    """Return a single global track by ID."""
    from frigate.app import FrigateApp

    app = FrigateApp.current
    if app is None or not hasattr(app, "detected_frames_processor"):
        return {"success": False, "message": "Frigate not ready"}

    tracker = getattr(app.detected_frames_processor, "cross_camera_tracker", None)
    if tracker is None:
        return {"success": False, "message": "Cross-camera tracking not enabled"}

    tracks = tracker.get_global_tracks()
    if global_id not in tracks:
        return {"success": False, "message": f"Track {global_id} not found"}

    return {"success": True, "track": tracks[global_id]}


@router.delete("/cross_camera/expired")
def cleanup_expired_tracks():
    """Manually trigger cleanup of expired global tracks."""
    from frigate.app import FrigateApp

    app = FrigateApp.current
    if app is None or not hasattr(app, "detected_frames_processor"):
        return {"success": False, "message": "Frigate not ready"}

    tracker = getattr(app.detected_frames_processor, "cross_camera_tracker", None)
    if tracker is None:
        return {"success": False, "message": "Cross-camera tracking not enabled"}

    removed = tracker.cleanup_expired()
    return {"success": True, "removed": removed}

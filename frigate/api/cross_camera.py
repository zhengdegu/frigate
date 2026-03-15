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
@router.get("/api/cross_camera/alerts")
def get_cross_camera_alerts(request: Request):
    """Return alerts enriched with cross-camera tracking paths.

    Links review alert detections to global tracks via local_track_id,
    producing a unified view of alert + movement path across cameras.
    """
    tracker = _get_tracker(request)
    if tracker is None:
        return {"success": False, "message": "Cross-camera tracking not enabled"}

    from frigate.models import ReviewSegment

    tracks = tracker.get_global_tracks()

    # Build local_track_id -> global_id index
    local_to_global: dict[str, str] = {}
    for gid, track in tracks.items():
        for sighting in track.get("sightings", []):
            lid = sighting.get("local_track_id")
            if lid:
                local_to_global[lid] = gid

    # Get recent alerts
    alerts = (
        ReviewSegment.select()
        .where(ReviewSegment.severity == "alert")
        .order_by(ReviewSegment.start_time.desc())
        .limit(50)
    )

    results = []
    seen_globals = set()

    for alert in alerts:
        alert_data = alert.data if isinstance(alert.data, dict) else {}
        detection_ids = alert_data.get("detections", [])

        # Find matching global tracks
        matched_global_ids = set()
        for det_id in detection_ids:
            if det_id in local_to_global:
                matched_global_ids.add(local_to_global[det_id])

        for gid in matched_global_ids:
            if gid in seen_globals:
                continue
            seen_globals.add(gid)

            track = tracks[gid]
            sightings = track.get("sightings", [])

            # Build camera path (deduplicated, ordered)
            camera_path = []
            for s in sightings:
                cam = s.get("camera")
                if not camera_path or camera_path[-1]["camera"] != cam:
                    camera_path.append({
                        "camera": cam,
                        "enter_time": s.get("first_seen"),
                        "exit_time": s.get("last_seen"),
                    })
                else:
                    camera_path[-1]["exit_time"] = s.get("last_seen")

            results.append({
                "alert_id": alert.id,
                "alert_camera": alert.camera,
                "alert_start": alert.start_time,
                "alert_severity": alert.severity,
                "alert_thumb": alert.thumb_path,
                "global_id": gid,
                "label": track.get("label", "unknown"),
                "upper_color": track.get("upper_color"),
                "lower_color": track.get("lower_color"),
                "body_ratio": track.get("body_ratio"),
                "face_name": track.get("face_name"),
                "plate": track.get("plate"),
                "color": track.get("color"),
                "vehicle_type": track.get("vehicle_type"),
                "camera_path": camera_path,
                "cameras_visited": len(set(s["camera"] for s in sightings)),
                "total_sightings": len(sightings),
                "first_seen": track.get("created_at"),
                "last_seen": track.get("updated_at"),
            })

    # Sort by cameras_visited desc (multi-camera first)
    results.sort(key=lambda x: (-x["cameras_visited"], -x["last_seen"]))

    return {
        "success": True,
        "alerts": results,
        "count": len(results),
    }


@router.delete("/api/cross_camera/expired")
def cleanup_expired_tracks(request: Request):
    """Manually trigger cleanup of expired global tracks."""
    tracker = _get_tracker(request)
    if tracker is None:
        return {"success": False, "message": "Cross-camera tracking not enabled"}

    removed = tracker.cleanup_expired()
    return {"success": True, "removed": removed}

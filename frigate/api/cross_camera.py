"""Cross-camera tracking API endpoints."""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

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


@router.get("/api/cross_camera/search")
def search_cross_camera_tracks(
    request: Request,
    upper_color: Optional[str] = None,
    lower_color: Optional[str] = None,
    label: Optional[str] = None,
    plate: Optional[str] = None,
    face_name: Optional[str] = None,
    vehicle_type: Optional[str] = None,
    color: Optional[str] = None,
):
    """Search global tracks by attribute filters (AND logic)."""
    tracker = _get_tracker(request)
    if tracker is None:
        return {"success": False, "message": "Cross-camera tracking not enabled"}

    tracks = tracker.get_global_tracks()

    filters = {
        "upper_color": upper_color,
        "lower_color": lower_color,
        "label": label,
        "plate": plate,
        "face_name": face_name,
        "vehicle_type": vehicle_type,
        "color": color,
    }
    # Remove None values
    active_filters = {k: v.lower() for k, v in filters.items() if v is not None}

    if not active_filters:
        return {"success": True, "tracks": tracks, "count": len(tracks)}

    matched = {}
    for gid, track in tracks.items():
        match = True
        for key, value in active_filters.items():
            track_value = track.get(key)
            if track_value is None or str(track_value).lower() != value:
                match = False
                break
        if match:
            matched[gid] = track

    return {"success": True, "tracks": matched, "count": len(matched)}


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

    from frigate.models import ReviewSegment, Event

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

        # Method 1: direct match detection ID -> local_track_id
        matched_global_ids = set()
        for det_id in detection_ids:
            if det_id in local_to_global:
                matched_global_ids.add(local_to_global[det_id])

        # Method 2: bridge via events table
        # When video restarts, tracker gets new IDs that don't match
        # old alert detections. Use events table to find current
        # active events on the same camera, then match to tracker.
        if not matched_global_ids:
            try:
                camera_events = (
                    Event.select(Event.id)
                    .where(Event.camera == alert.camera)
                    .limit(500)
                )
                for ev in camera_events:
                    if ev.id in local_to_global:
                        matched_global_ids.add(local_to_global[ev.id])
            except Exception:
                pass

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


@router.get("/api/cross_camera/map_data")
def get_cross_camera_map_data(request: Request):
    """Return camera locations and active track paths for map display.

    Each track includes an ordered list of camera coordinates representing
    the movement path, suitable for Google Maps route rendering.
    """
    tracker = _get_tracker(request)
    if tracker is None:
        return {"success": False, "message": "Cross-camera tracking not enabled"}

    from frigate.config import FrigateConfig

    config: FrigateConfig = request.app.frigate_config
    tracks = tracker.get_global_tracks()

    # Build camera location map
    cameras = {}
    for cam_name, cam_config in config.cameras.items():
        lat = getattr(cam_config, "latitude", None)
        lng = getattr(cam_config, "longitude", None)
        if lat is not None and lng is not None:
            cameras[cam_name] = {
                "name": cam_name,
                "latitude": lat,
                "longitude": lng,
            }

    # Build track paths with coordinates
    track_paths = []
    for gid, track in tracks.items():
        sightings = track.get("sightings", [])
        if len(sightings) < 1:
            continue

        # Build ordered waypoints (deduplicated consecutive cameras)
        waypoints = []
        for s in sightings:
            cam = s.get("camera")
            if cam not in cameras:
                continue
            loc = cameras[cam]
            if not waypoints or waypoints[-1]["camera"] != cam:
                waypoints.append({
                    "camera": cam,
                    "latitude": loc["latitude"],
                    "longitude": loc["longitude"],
                    "enter_time": s.get("first_seen"),
                    "exit_time": s.get("last_seen"),
                })
            else:
                waypoints[-1]["exit_time"] = s.get("last_seen")

        if len(waypoints) < 1:
            continue

        track_paths.append({
            "global_id": gid,
            "label": track.get("label", "unknown"),
            "upper_color": track.get("upper_color"),
            "lower_color": track.get("lower_color"),
            "face_name": track.get("face_name"),
            "plate": track.get("plate"),
            "color": track.get("color"),
            "vehicle_type": track.get("vehicle_type"),
            "waypoints": waypoints,
            "cameras_visited": len(set(w["camera"] for w in waypoints)),
            "first_seen": track.get("created_at"),
            "last_seen": track.get("updated_at"),
        })

    # Sort by cameras_visited desc
    track_paths.sort(key=lambda x: (-x["cameras_visited"], -x["last_seen"]))

    return {
        "success": True,
        "cameras": cameras,
        "tracks": track_paths,
        "count": len(track_paths),
    }


@router.get("/api/cross_camera/dwell_analysis")
def get_dwell_analysis(request: Request, threshold: int = 300):
    """Analyze dwell time per camera for each global track.

    Returns how long each global track stayed at each camera,
    and flags anomalies where dwell time exceeds the threshold.

    Query params:
        threshold: seconds to consider a dwell anomalous (default 300)
    """
    tracker = _get_tracker(request)
    if tracker is None:
        return {"success": False, "message": "Cross-camera tracking not enabled"}

    tracks = tracker.get_global_tracks()

    analysis = []
    anomalies = []

    for gid, track in tracks.items():
        sightings = track.get("sightings", [])
        if not sightings:
            continue

        camera_dwell = []
        total_duration = 0.0

        for s in sightings:
            first_seen = s.get("first_seen")
            last_seen = s.get("last_seen")
            if first_seen is None or last_seen is None:
                continue
            duration = round(last_seen - first_seen, 2)
            is_anomaly = duration > threshold
            camera_dwell.append({
                "camera": s.get("camera"),
                "duration": duration,
                "is_anomaly": is_anomaly,
            })
            total_duration += duration

        entry = {
            "global_id": gid,
            "label": track.get("label", "unknown"),
            "total_duration": round(total_duration, 2),
            "camera_dwell": camera_dwell,
        }
        analysis.append(entry)

        if any(d["is_anomaly"] for d in camera_dwell):
            anomalies.append(entry)

    return {
        "success": True,
        "analysis": analysis,
        "anomalies": anomalies,
        "threshold": threshold,
    }


@router.get("/api/cross_camera/heatmap")
def get_cross_camera_heatmap(request: Request):
    """Return heatmap data: per-camera detection counts and path flows between cameras."""
    tracker = _get_tracker(request)
    if tracker is None:
        return {"success": False, "message": "Cross-camera tracking not enabled"}

    tracks = tracker.get_global_tracks()

    # camera_counts: {camera: {label: count}}
    camera_counts: dict[str, dict[str, int]] = {}
    # path_flows: {(from, to, label): count}
    path_flows: dict[tuple[str, str, str], int] = {}

    for track in tracks.values():
        label = track.get("label", "unknown")
        sightings = track.get("sightings", [])

        prev_camera = None
        for s in sightings:
            cam = s.get("camera")
            if cam is None:
                continue

            # Count detection per camera
            if cam not in camera_counts:
                camera_counts[cam] = {}
            camera_counts[cam][label] = camera_counts[cam].get(label, 0) + 1

            # Count path flow between consecutive different cameras
            if prev_camera is not None and prev_camera != cam:
                key = (prev_camera, cam, label)
                path_flows[key] = path_flows.get(key, 0) + 1

            prev_camera = cam

    # Format camera_counts with totals
    formatted_counts: dict[str, dict[str, int]] = {}
    for cam, labels in camera_counts.items():
        entry = dict(labels)
        entry["total"] = sum(labels.values())
        formatted_counts[cam] = entry

    # Format path_flows as list
    formatted_flows = [
        {"from": f, "to": t, "count": c, "label": lb}
        for (f, t, lb), c in path_flows.items()
    ]
    formatted_flows.sort(key=lambda x: -x["count"])

    # Busiest camera
    busiest_camera = None
    if formatted_counts:
        busiest_camera = max(formatted_counts, key=lambda c: formatted_counts[c]["total"])

    # Busiest path
    busiest_path = None
    if formatted_flows:
        top = formatted_flows[0]
        busiest_path = {"from": top["from"], "to": top["to"], "count": top["count"]}

    return {
        "success": True,
        "camera_counts": formatted_counts,
        "path_flows": formatted_flows,
        "busiest_camera": busiest_camera,
        "busiest_path": busiest_path,
    }


@router.delete("/api/cross_camera/expired")
def cleanup_expired_tracks(request: Request):
    """Manually trigger cleanup of expired global tracks."""
    tracker = _get_tracker(request)
    if tracker is None:
        return {"success": False, "message": "Cross-camera tracking not enabled"}

    removed = tracker.cleanup_expired()
    return {"success": True, "removed": removed}


# ---------------------------------------------------------------------------
# Watchlist (face / plate watch list)
# ---------------------------------------------------------------------------

WATCHLIST_PATH = "/config/cross_camera_watchlist.json"


class WatchlistItem(BaseModel):
    type: str  # "face" or "plate"
    value: str
    note: str = ""


def _load_watchlist() -> list[dict]:
    """Load watchlist from JSON file."""
    if not os.path.exists(WATCHLIST_PATH):
        return []
    try:
        with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_watchlist(items: list[dict]) -> None:
    """Save watchlist to JSON file."""
    os.makedirs(os.path.dirname(WATCHLIST_PATH), exist_ok=True)
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


@router.post("/api/cross_camera/watchlist")
def add_watchlist_item(request: Request, body: WatchlistItem):
    """Add a face or plate to the watchlist."""
    if body.type not in ("face", "plate"):
        return {"success": False, "message": "type must be 'face' or 'plate'"}

    items = _load_watchlist()
    new_item = {
        "id": str(uuid.uuid4()),
        "type": body.type,
        "value": body.value,
        "note": body.note,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    items.append(new_item)
    _save_watchlist(items)

    return {"success": True, "item": new_item}


@router.get("/api/cross_camera/watchlist")
def get_watchlist(request: Request):
    """Return the full watchlist."""
    items = _load_watchlist()
    return {"success": True, "items": items, "count": len(items)}


@router.delete("/api/cross_camera/watchlist/{item_id}")
def delete_watchlist_item(request: Request, item_id: str):
    """Delete a watchlist item by ID."""
    items = _load_watchlist()
    new_items = [i for i in items if i.get("id") != item_id]

    if len(new_items) == len(items):
        return {"success": False, "message": f"Item {item_id} not found"}

    _save_watchlist(new_items)
    return {"success": True, "message": f"Item {item_id} deleted"}


@router.get("/api/cross_camera/watchlist/matches")
def get_watchlist_matches(request: Request):
    """Check current global tracks for matches against the watchlist."""
    tracker = _get_tracker(request)
    if tracker is None:
        return {"success": False, "message": "Cross-camera tracking not enabled"}

    items = _load_watchlist()
    if not items:
        return {"success": True, "matches": [], "count": 0}

    # Build lookup sets
    face_watchlist: dict[str, dict] = {}
    plate_watchlist: dict[str, dict] = {}
    for item in items:
        if item["type"] == "face":
            face_watchlist[item["value"].lower()] = item
        elif item["type"] == "plate":
            plate_watchlist[item["value"].upper()] = item

    tracks = tracker.get_global_tracks()
    matches = []

    for gid, track in tracks.items():
        # Check face match
        face_name = track.get("face_name")
        if face_name and face_name.lower() in face_watchlist:
            wl_item = face_watchlist[face_name.lower()]
            matches.append({
                "watchlist_item": wl_item,
                "global_id": gid,
                "match_field": "face_name",
                "match_value": face_name,
                "track": track,
            })

        # Check plate match
        plate = track.get("plate")
        if plate and plate.upper() in plate_watchlist:
            wl_item = plate_watchlist[plate.upper()]
            matches.append({
                "watchlist_item": wl_item,
                "global_id": gid,
                "match_field": "plate",
                "match_value": plate,
                "track": track,
            })

    return {"success": True, "matches": matches, "count": len(matches)}

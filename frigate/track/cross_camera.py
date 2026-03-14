"""Cross-camera vehicle tracking engine.

Uses a tiered matching strategy:
  1. License plate exact/fuzzy match (strongest signal)
  2. Vehicle color + type + appearance (fallback when plate unavailable)
  3. Spatio-temporal constraints from camera topology

All matching is gated by the camera topology so only plausible
camera transitions are considered.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VehicleSignature:
    """Compact representation of a tracked vehicle for cross-camera matching."""

    camera: str
    local_track_id: str
    plate: Optional[str] = None
    plate_score: float = 0.0
    color: str = "unknown"
    vehicle_type: str = "unknown"  # sedan / suv / truck / van / bus / unknown
    color_histogram: Optional[np.ndarray] = None
    first_seen: float = 0.0
    last_seen: float = 0.0
    box: list[int] = field(default_factory=list)
    direction: Optional[str] = None  # entering / leaving


@dataclass
class GlobalTrack:
    """A vehicle identity that persists across cameras."""

    global_id: str
    plate: Optional[str] = None
    plate_score: float = 0.0
    color: str = "unknown"
    vehicle_type: str = "unknown"
    color_histogram: Optional[np.ndarray] = None
    sightings: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0


# ---------------------------------------------------------------------------
# Color extraction
# ---------------------------------------------------------------------------

# HSV hue ranges mapped to color names (OpenCV hue is 0-179)
_HUE_MAP = {
    0: "red", 1: "orange", 2: "yellow", 3: "green", 4: "green",
    5: "cyan", 6: "blue", 7: "blue", 8: "purple", 9: "pink",
    10: "red", 11: "red",
}


def extract_vehicle_color(crop_bgr: np.ndarray) -> tuple[str, np.ndarray]:
    """Return (color_name, hsv_histogram) from a vehicle crop.

    The top 1/3 of the crop (windshield area) is excluded to focus on
    the body panels which carry the actual paint color.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown", np.zeros(12, dtype=np.float32)

    h, w = crop_bgr.shape[:2]
    # use bottom 2/3 = body panels
    body = crop_bgr[h // 3:, :, :]
    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)

    # compute hue histogram (12 bins)
    hist = cv2.calcHist([hsv], [0], None, [12], [0, 180])
    hist = hist.flatten().astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total

    # check achromatic (black / white / gray / silver)
    mean_sat = float(np.mean(hsv[:, :, 1]))
    mean_val = float(np.mean(hsv[:, :, 2]))

    if mean_sat < 40:
        if mean_val < 60:
            return "black", hist
        if mean_val > 180:
            return "white", hist
        if mean_val > 120:
            return "silver", hist
        return "gray", hist

    dominant_bin = int(np.argmax(hist))
    color_name = _HUE_MAP.get(dominant_bin, "unknown")
    return color_name, hist


# ---------------------------------------------------------------------------
# Vehicle type estimation (lightweight, no extra model)
# ---------------------------------------------------------------------------

def estimate_vehicle_type(box: list[int]) -> str:
    """Rough vehicle type from bounding box aspect ratio.

    This is a cheap heuristic; a dedicated classifier would be better
    but this gives a useful signal at zero cost.
    """
    if len(box) < 4:
        return "unknown"
    w = box[2] - box[0]
    h = box[3] - box[1]
    if w == 0 or h == 0:
        return "unknown"
    ratio = w / h
    # trucks / buses are wider and taller
    if ratio > 2.2:
        return "bus"
    if ratio > 1.8:
        return "truck"
    if h > w * 0.85:
        return "suv"
    return "sedan"


# ---------------------------------------------------------------------------
# Plate matching
# ---------------------------------------------------------------------------

def plate_similarity(a: Optional[str], b: Optional[str]) -> float:
    """Score 0-1 for how similar two plate strings are.

    Returns 0 if either plate is None/empty.
    """
    if not a or not b:
        return 0.0
    a, b = a.upper().strip(), b.upper().strip()
    if a == b:
        return 1.0
    # allow 1-char OCR error for same-length plates
    if len(a) == len(b):
        diffs = sum(ca != cb for ca, cb in zip(a, b))
        if diffs == 1:
            return 0.85
        if diffs == 2:
            return 0.5
    # partial suffix match (last 4+ chars)
    min_len = min(len(a), len(b))
    if min_len >= 4 and a[-4:] == b[-4:]:
        return 0.6
    return 0.0


# ---------------------------------------------------------------------------
# Color histogram similarity
# ---------------------------------------------------------------------------

def color_histogram_similarity(h1: Optional[np.ndarray], h2: Optional[np.ndarray]) -> float:
    """Bhattacharyya-based similarity between two hue histograms (0-1, higher=better)."""
    if h1 is None or h2 is None:
        return 0.0
    if h1.sum() == 0 or h2.sum() == 0:
        return 0.0
    # cv2.compareHist HISTCMP_BHATTACHARYYA returns distance 0-1
    dist = cv2.compareHist(
        h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA
    )
    return max(0.0, 1.0 - dist)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class CrossCameraTracker:
    """Manages cross-camera vehicle identity association.

    Thread-safe: all public methods acquire ``_lock``.
    """

    def __init__(self, config) -> None:
        from frigate.config.cross_camera import CrossCameraConfig

        self.config: CrossCameraConfig = config
        self._lock = threading.Lock()

        # global_id counter
        self._next_id = 1

        # plate -> global_id (exact plate index for fast lookup)
        self._plate_index: dict[str, str] = {}

        # global_id -> GlobalTrack
        self._global_tracks: dict[str, GlobalTrack] = {}

        # local_track_id -> global_id (for quick reverse lookup)
        self._local_to_global: dict[str, str] = {}

        # camera topology: (src, dst) -> (min_s, max_s)
        self._topology: dict[tuple[str, str], tuple[float, float]] = {}
        for link in self.config.topology:
            self._topology[(link.source, link.target)] = (
                link.min_seconds,
                link.max_seconds,
            )
            if link.bidirectional:
                self._topology[(link.target, link.source)] = (
                    link.min_seconds,
                    link.max_seconds,
                )

        logger.info(
            f"CrossCameraTracker initialized: "
            f"{len(self._topology)} links, "
            f"tracked_objects={self.config.tracked_objects}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_track_update(
        self,
        camera: str,
        local_track_id: str,
        label: str,
        frame_time: float,
        box: list[int],
        plate: Optional[str] = None,
        plate_score: float = 0.0,
        crop_bgr: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Called when a tracked object is updated.

        Returns the global_id if a cross-camera association is made,
        or None if the label is not tracked cross-camera.
        """
        if label not in self.config.tracked_objects:
            return None

        with self._lock:
            # already associated?
            if local_track_id in self._local_to_global:
                gid = self._local_to_global[local_track_id]
                self._touch_global(gid, camera, local_track_id, frame_time, box, plate, plate_score, crop_bgr)
                return gid

            # build signature
            sig = self._build_signature(
                camera, local_track_id, frame_time, box, plate, plate_score, crop_bgr
            )

            # try match
            gid = self._match(sig)
            if gid is None:
                gid = self._create_global(sig)

            self._local_to_global[local_track_id] = gid
            return gid

    def on_track_end(self, camera: str, local_track_id: str) -> None:
        """Called when a local track disappears from a camera."""
        with self._lock:
            # keep the global track alive for gallery_max_age
            # just remove the local mapping
            if local_track_id in self._local_to_global:
                gid = self._local_to_global.pop(local_track_id)
                gt = self._global_tracks.get(gid)
                if gt:
                    gt.updated_at = time.time()

    def cleanup_expired(self) -> int:
        """Remove global tracks older than gallery_max_age. Returns count removed."""
        now = time.time()
        max_age = self.config.gallery_max_age
        removed = 0
        with self._lock:
            expired = [
                gid
                for gid, gt in self._global_tracks.items()
                if (now - gt.updated_at) > max_age
            ]
            for gid in expired:
                gt = self._global_tracks.pop(gid)
                if gt.plate and gt.plate in self._plate_index:
                    if self._plate_index[gt.plate] == gid:
                        del self._plate_index[gt.plate]
                removed += 1
        return removed

    def get_global_tracks(self) -> dict[str, dict]:
        """Return a snapshot of all active global tracks (for API)."""
        with self._lock:
            return {
                gid: {
                    "global_id": gid,
                    "plate": gt.plate,
                    "color": gt.color,
                    "vehicle_type": gt.vehicle_type,
                    "sightings": gt.sightings.copy(),
                    "created_at": gt.created_at,
                    "updated_at": gt.updated_at,
                }
                for gid, gt in self._global_tracks.items()
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_signature(
        self,
        camera: str,
        local_track_id: str,
        frame_time: float,
        box: list[int],
        plate: Optional[str],
        plate_score: float,
        crop_bgr: Optional[np.ndarray],
    ) -> VehicleSignature:
        color_name = "unknown"
        color_hist = None
        if crop_bgr is not None:
            color_name, color_hist = extract_vehicle_color(crop_bgr)

        vtype = estimate_vehicle_type(box)

        return VehicleSignature(
            camera=camera,
            local_track_id=local_track_id,
            plate=plate,
            plate_score=plate_score,
            color=color_name,
            vehicle_type=vtype,
            color_histogram=color_hist,
            first_seen=frame_time,
            last_seen=frame_time,
            box=box,
        )

    def _match(self, sig: VehicleSignature) -> Optional[str]:
        """Try to find an existing global track that matches *sig*."""

        # --- Strategy 1: plate exact/fuzzy match ---
        if sig.plate:
            # exact
            if sig.plate.upper().strip() in self._plate_index:
                gid = self._plate_index[sig.plate.upper().strip()]
                logger.debug(
                    f"Cross-camera plate match: {sig.plate} -> {gid} "
                    f"(cam={sig.camera})"
                )
                return gid

            # fuzzy
            best_gid = None
            best_score = 0.0
            for p, gid in self._plate_index.items():
                s = plate_similarity(sig.plate, p)
                if s > best_score:
                    best_score = s
                    best_gid = gid
            if best_score >= self.config.plate_match_threshold and best_gid:
                logger.debug(
                    f"Cross-camera fuzzy plate match: {sig.plate} ~ "
                    f"{self._global_tracks[best_gid].plate} "
                    f"(score={best_score:.2f}, cam={sig.camera})"
                )
                return best_gid

        # --- Strategy 2: color + type + appearance with topology constraint ---
        candidates = self._get_topology_candidates(sig)
        if not candidates:
          return None

        best_gid = None
        best_combined = 0.0

        for gid, gt in candidates:
            # color name match (binary)
            color_name_score = 1.0 if (
                sig.color != "unknown"
                and gt.color != "unknown"
                and sig.color == gt.color
            ) else 0.0

            # color histogram similarity (continuous)
            hist_score = color_histogram_similarity(sig.color_histogram, gt.color_histogram)

            # blend name match and histogram
            color_score = max(color_name_score, hist_score)

            # vehicle type match
            type_score = 1.0 if (
                sig.vehicle_type != "unknown"
                and gt.vehicle_type != "unknown"
                and sig.vehicle_type == gt.vehicle_type
            ) else 0.0

            # combined
            combined = (
                self.config.color_weight * color_score
                + self.config.type_weight * type_score
                + self.config.appearance_weight * hist_score
            )

            if combined > best_combined:
                best_combined = combined
                best_gid = gid

        if best_combined >= self.config.match_threshold and best_gid:
            logger.debug(
                f"Cross-camera appearance match: "
                f"color={sig.color}, type={sig.vehicle_type} -> {best_gid} "
                f"(score={best_combined:.2f}, cam={sig.camera})"
            )
            return best_gid

        return None

    def _get_topology_candidates(
        self, sig: VehicleSignature
    ) -> list[tuple[str, GlobalTrack]]:
        """Return global tracks that could plausibly be the same vehicle
        based on camera topology and timing."""
        now = sig.last_seen
        results = []

        for gid, gt in self._global_tracks.items():
            # skip if same camera and still active
            last_sighting = gt.sightings[-1] if gt.sightings else None
            if not last_sighting:
                continue

            last_cam = last_sighting["camera"]
            last_time = last_sighting["last_seen"]

            # same camera → not a cross-camera match
            if last_cam == sig.camera:
                continue

            # check topology
            link = self._topology.get((last_cam, sig.camera))
            if link is None:
                # no topology defined between these cameras
                # if topology is empty, allow all (open matching)
                if self._topology:
                    continue

            elapsed = now - last_time
            if elapsed < 0:
                continue

            if link:
                min_s, max_s = link
                if elapsed < min_s or elapsed > max_s:
                    continue

            # passed topology check
            results.append((gid, gt))

        return results

    def _create_global(self, sig: VehicleSignature) -> str:
        gid = f"xc-{self._next_id}"
        self._next_id += 1

        gt = GlobalTrack(
            global_id=gid,
            plate=sig.plate.upper().strip() if sig.plate else None,
            plate_score=sig.plate_score,
            color=sig.color,
            vehicle_type=sig.vehicle_type,
            color_histogram=sig.color_histogram,
            created_at=sig.first_seen,
            updated_at=sig.last_seen,
            sightings=[
                {
                    "camera": sig.camera,
                    "local_track_id": sig.local_track_id,
                    "first_seen": sig.first_seen,
                    "last_seen": sig.last_seen,
                    "box": sig.box,
                }
            ],
        )
        self._global_tracks[gid] = gt

        if gt.plate:
            self._plate_index[gt.plate] = gid

        logger.debug(
            f"New global track {gid}: plate={gt.plate}, "
            f"color={gt.color}, type={gt.vehicle_type}, cam={sig.camera}"
        )
        return gid

    def _touch_global(
        self,
        gid: str,
        camera: str,
        local_track_id: str,
        frame_time: float,
        box: list[int],
        plate: Optional[str],
        plate_score: float,
        crop_bgr: Optional[np.ndarray],
    ) -> None:
        """Update an existing global track with new data."""
        gt = self._global_tracks.get(gid)
        if not gt:
            return

        gt.updated_at = frame_time

        # update plate if we got a better one
        if plate and plate_score > gt.plate_score:
            old_plate = gt.plate
            gt.plate = plate.upper().strip()
            gt.plate_score = plate_score
            # update index
            if old_plate and old_plate in self._plate_index:
                del self._plate_index[old_plate]
            self._plate_index[gt.plate] = gid

        # update color if we have a crop
        if crop_bgr is not None:
            color_name, color_hist = extract_vehicle_color(crop_bgr)
            if color_name != "unknown":
                gt.color = color_name
                gt.color_histogram = color_hist

        # update vehicle type
        vtype = estimate_vehicle_type(box)
        if vtype != "unknown":
            gt.vehicle_type = vtype

        # update or append sighting
        if gt.sightings and gt.sightings[-1]["camera"] == camera:
            gt.sightings[-1]["last_seen"] = frame_time
            gt.sightings[-1]["box"] = box
        else:
            gt.sightings.append(
                {
                    "camera": camera,
                    "local_track_id": local_track_id,
                    "first_seen": frame_time,
                    "last_seen": frame_time,
                    "box": box,
                }
            )

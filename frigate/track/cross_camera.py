"""Cross-camera vehicle tracking engine.

Tiered matching strategy:
  1. License plate exact/fuzzy match (strongest signal)
  2. Vehicle color + type + appearance histogram (fallback)
  3. Spatio-temporal constraints from camera topology
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from frigate.config.cross_camera import CrossCameraConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class VehicleSignature:
    """Compact representation of a tracked vehicle."""

    camera: str
    local_track_id: str
    label: str = "car"
    plate: Optional[str] = None
    plate_score: float = 0.0
    color: str = "unknown"
    vehicle_type: str = "unknown"
    color_histogram: Optional[np.ndarray] = None
    first_seen: float = 0.0
    last_seen: float = 0.0
    box: list[int] = field(default_factory=list)


@dataclass
class PersonSignature:
    """Compact representation of a tracked person."""

    camera: str
    local_track_id: str
    label: str = "person"
    upper_color: str = "unknown"
    lower_color: str = "unknown"
    upper_histogram: Optional[np.ndarray] = None
    lower_histogram: Optional[np.ndarray] = None
    body_ratio: float = 0.0  # height / width
    face_embedding: Optional[np.ndarray] = None
    face_name: Optional[str] = None  # recognized face name from Frigate
    first_seen: float = 0.0
    last_seen: float = 0.0
    box: list[int] = field(default_factory=list)


@dataclass
class GlobalTrack:
    """An identity that persists across cameras (vehicle or person)."""

    global_id: str
    label: str = "car"
    # vehicle fields
    plate: Optional[str] = None
    plate_score: float = 0.0
    color: str = "unknown"
    vehicle_type: str = "unknown"
    color_histogram: Optional[np.ndarray] = None
    # person fields
    upper_color: str = "unknown"
    lower_color: str = "unknown"
    upper_histogram: Optional[np.ndarray] = None
    lower_histogram: Optional[np.ndarray] = None
    body_ratio: float = 0.0
    face_embedding: Optional[np.ndarray] = None
    face_name: Optional[str] = None
    # common
    sightings: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0


# ---------------------------------------------------------------------------
# Color extraction (pure OpenCV, no extra model)
# ---------------------------------------------------------------------------

_HUE_MAP = {
    0: "red",
    1: "orange",
    2: "yellow",
    3: "green",
    4: "green",
    5: "cyan",
    6: "blue",
    7: "blue",
    8: "blue",
    9: "purple",
    10: "pink",
    11: "red",
}


def extract_vehicle_color(crop_bgr: np.ndarray) -> tuple[str, np.ndarray]:
    """Return (color_name, hsv_histogram) from a vehicle crop.

    Excludes top 1/3 (windshield) to focus on body panels.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown", np.zeros(12, dtype=np.float32)

    h, w = crop_bgr.shape[:2]
    body = crop_bgr[h // 3 :, :, :]
    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv], [0], None, [12], [0, 180])
    hist = hist.flatten().astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total

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
    return _HUE_MAP.get(dominant_bin, "unknown"), hist


# ---------------------------------------------------------------------------
# Vehicle type estimation (from bbox aspect ratio, no extra model)
# ---------------------------------------------------------------------------


def estimate_vehicle_type(box: list[int]) -> str:
    """Rough vehicle type from bounding box aspect ratio."""
    if len(box) < 4:
        return "unknown"
    w = box[2] - box[0]
    h = box[3] - box[1]
    if w == 0 or h == 0:
        return "unknown"
    ratio = w / h
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
    """Score 0-1 for how similar two plate strings are."""
    if not a or not b:
        return 0.0
    a, b = a.upper().strip(), b.upper().strip()
    if a == b:
        return 1.0
    if len(a) == len(b):
        diffs = sum(ca != cb for ca, cb in zip(a, b))
        if diffs == 1:
            return 0.85
        if diffs == 2:
            return 0.5
    min_len = min(len(a), len(b))
    if min_len >= 4 and a[-4:] == b[-4:]:
        return 0.6
    return 0.0


# ---------------------------------------------------------------------------
# Color histogram similarity
# ---------------------------------------------------------------------------


def color_histogram_similarity(
    h1: Optional[np.ndarray], h2: Optional[np.ndarray]
) -> float:
    """Bhattacharyya-based similarity (0-1, higher=better)."""
    if h1 is None or h2 is None:
        return 0.0
    if h1.sum() == 0 or h2.sum() == 0:
        return 0.0
    dist = cv2.compareHist(
        h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA
    )
    return max(0.0, 1.0 - dist)


# ---------------------------------------------------------------------------
# Person: clothing color extraction (upper/lower body, pure OpenCV)
# ---------------------------------------------------------------------------


def extract_person_colors(
    crop_bgr: np.ndarray,
) -> tuple[str, str, np.ndarray, np.ndarray]:
    """Extract upper and lower body dominant colors from a person crop.

    Splits the crop at ~40% from top (upper body) and bottom 60% (lower body).
    Returns (upper_color, lower_color, upper_histogram, lower_histogram).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        z = np.zeros(12, dtype=np.float32)
        return "unknown", "unknown", z, z

    h, w = crop_bgr.shape[:2]
    if h < 10 or w < 5:
        z = np.zeros(12, dtype=np.float32)
        return "unknown", "unknown", z, z

    split = int(h * 0.4)
    upper = crop_bgr[:split, :, :]
    lower = crop_bgr[split:, :, :]

    upper_color, upper_hist = _region_color(upper)
    lower_color, lower_hist = _region_color(lower)
    return upper_color, lower_color, upper_hist, lower_hist


def _region_color(region_bgr: np.ndarray) -> tuple[str, np.ndarray]:
    """Get dominant color name and histogram from a BGR region."""
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten().astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total

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
    return _HUE_MAP.get(dominant_bin, "unknown"), hist


# ---------------------------------------------------------------------------
# Person: body ratio estimation
# ---------------------------------------------------------------------------


def estimate_body_ratio(box: list[int]) -> float:
    """Height/width ratio of a person bounding box. Taller people have higher ratios."""
    if len(box) < 4:
        return 0.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    if w == 0:
        return 0.0
    return h / w


# ---------------------------------------------------------------------------
# Person: face embedding similarity
# ---------------------------------------------------------------------------


def face_embedding_similarity(
    e1: Optional[np.ndarray], e2: Optional[np.ndarray]
) -> float:
    """Cosine similarity between two face embeddings (0-1, higher=better)."""
    if e1 is None or e2 is None:
        return 0.0
    e1 = e1.flatten().astype(np.float64)
    e2 = e2.flatten().astype(np.float64)
    norm1 = np.linalg.norm(e1)
    norm2 = np.linalg.norm(e2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(e1, e2) / (norm1 * norm2))


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class CrossCameraTracker:
    """Manages cross-camera vehicle identity association. Thread-safe."""

    def __init__(self, config: CrossCameraConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._next_id = 1
        self._plate_index: dict[str, str] = {}
        self._global_tracks: dict[str, GlobalTrack] = {}
        self._local_to_global: dict[str, str] = {}
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
            f"CrossCameraTracker initialized: {len(self._topology)} topology links, "
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
        face_embedding: Optional[np.ndarray] = None,
        face_name: Optional[str] = None,
    ) -> Optional[str]:
        """Called on every tracked object update. Returns global_id or None."""
        if label not in self.config.tracked_objects:
            return None

        with self._lock:
            if local_track_id in self._local_to_global:
                gid = self._local_to_global[local_track_id]
                self._touch_global(
                    gid, camera, local_track_id, label, frame_time, box,
                    plate, plate_score, crop_bgr, face_embedding, face_name,
                )
                return gid

            if label == "person":
                sig = self._build_person_signature(
                    camera, local_track_id, frame_time, box,
                    crop_bgr, face_embedding, face_name,
                )
            else:
                sig = self._build_signature(
                    camera, local_track_id, frame_time, box,
                    plate, plate_score, crop_bgr,
                )

            gid = self._match(sig)
            if gid is None:
                gid = self._create_global(sig)

            self._local_to_global[local_track_id] = gid
            return gid

    def on_track_end(self, camera: str, local_track_id: str) -> None:
        """Called when a local track disappears."""
        with self._lock:
            if local_track_id in self._local_to_global:
                gid = self._local_to_global.pop(local_track_id)
                gt = self._global_tracks.get(gid)
                if gt:
                    gt.updated_at = time.time()

    def cleanup_expired(self) -> int:
        """Remove global tracks older than gallery_max_age."""
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
                if gt.plate and self._plate_index.get(gt.plate) == gid:
                    del self._plate_index[gt.plate]
                removed += 1
        return removed

    def get_global_tracks(self) -> dict[str, dict]:
        """Snapshot of all active global tracks (for API / MQTT)."""
        with self._lock:
            result = {}
            for gid, gt in self._global_tracks.items():
                entry = {
                    "global_id": gid,
                    "label": gt.label,
                    "sightings": list(gt.sightings),
                    "created_at": gt.created_at,
                    "updated_at": gt.updated_at,
                }
                if gt.label == "person":
                    entry["upper_color"] = gt.upper_color
                    entry["lower_color"] = gt.lower_color
                    entry["body_ratio"] = round(gt.body_ratio, 2)
                    entry["face_name"] = gt.face_name
                else:
                    entry["plate"] = gt.plate
                    entry["color"] = gt.color
                    entry["vehicle_type"] = gt.vehicle_type
                result[gid] = entry
            return result

    # ------------------------------------------------------------------
    # Internal: build signature
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
        if crop_bgr is not None and crop_bgr.size > 0:
            color_name, color_hist = extract_vehicle_color(crop_bgr)

        return VehicleSignature(
            camera=camera,
            local_track_id=local_track_id,
            plate=plate,
            plate_score=plate_score,
            color=color_name,
            vehicle_type=estimate_vehicle_type(box),
            color_histogram=color_hist,
            first_seen=frame_time,
            last_seen=frame_time,
            box=box,
        )

    def _build_person_signature(
        self,
        camera: str,
        local_track_id: str,
        frame_time: float,
        box: list[int],
        crop_bgr: Optional[np.ndarray],
        face_embedding: Optional[np.ndarray],
        face_name: Optional[str],
    ) -> PersonSignature:
        upper_color = "unknown"
        lower_color = "unknown"
        upper_hist = None
        lower_hist = None
        if crop_bgr is not None and crop_bgr.size > 0:
            upper_color, lower_color, upper_hist, lower_hist = extract_person_colors(crop_bgr)

        return PersonSignature(
            camera=camera,
            local_track_id=local_track_id,
            upper_color=upper_color,
            lower_color=lower_color,
            upper_histogram=upper_hist,
            lower_histogram=lower_hist,
            body_ratio=estimate_body_ratio(box),
            face_embedding=face_embedding,
            face_name=face_name,
            first_seen=frame_time,
            last_seen=frame_time,
            box=box,
        )

    # ---------------------------------------------------------------
    # Internal: matching
    # ------------------------------------------------------------------

    def _match(self, sig) -> Optional[str]:
        """Try to find an existing global track matching sig (vehicle or person)."""
        if isinstance(sig, PersonSignature):
            return self._match_person(sig)
        return self._match_vehicle(sig)

    def _match_vehicle(self, sig: VehicleSignature) -> Optional[str]:
        """Vehicle matching: plate first, then color+type+appearance."""

        # --- Strategy 1: plate match ---
        if sig.plate:
            plate_key = sig.plate.upper().strip()

            # exact
            if plate_key in self._plate_index:
                gid = self._plate_index[plate_key]
                logger.debug(
                    f"Cross-cam plate exact match: {plate_key} -> {gid} (cam={sig.camera})"
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
                    f"Cross-cam plate fuzzy match: {sig.plate} ~ "
                    f"{self._global_tracks[best_gid].plate} "
                    f"(score={best_score:.2f}, cam={sig.camera})"
                )
                return best_gid

        # --- Strategy 2: color + type + appearance with topology ---
        candidates = self._get_topology_candidates(sig, label="car")
        if not candidates:
            return None

        best_gid = None
        best_combined = 0.0

        for gid, gt in candidates:
            color_name_score = 1.0 if (
                sig.color != "unknown"
                and gt.color != "unknown"
                and sig.color == gt.color
            ) else 0.0

            hist_score = color_histogram_similarity(
                sig.color_histogram, gt.color_histogram
            )
            # If both color names are known but different, override histogram
            if (sig.color != "unknown" and gt.color != "unknown"
                    and sig.color != gt.color):
                color_score = 0.0
                hist_score = 0.0
            else:
                color_score = max(color_name_score, hist_score)

            type_score = 1.0 if (
                sig.vehicle_type != "unknown"
                and gt.vehicle_type != "unknown"
                and sig.vehicle_type == gt.vehicle_type
            ) else 0.0

            combined = (
                self.config.color_weight * color_score
                + self.config.type_weight * type_score
                + self.config.appearance_weight * hist_score
            )

            if combined > best_combined:
                best_combined = combined
                best_gid = gid

        if best_combined >= self.config.match_threshold and best_gid:
            gt = self._global_tracks[best_gid]
            logger.debug(
                f"Cross-cam vehicle appearance match: color={sig.color} type={sig.vehicle_type} "
                f"-> {best_gid} (plate={gt.plate}, score={best_combined:.2f}, cam={sig.camera})"
            )
            return best_gid

        return None

    def _match_person(self, sig: PersonSignature) -> Optional[str]:
        """Person matching: face first, then clothing color + body ratio."""

        # --- Strategy 1: recognized face name (strongest) ---
        if sig.face_name:
            for gid, gt in self._global_tracks.items():
                if gt.label == "person" and gt.face_name and gt.face_name == sig.face_name:
                    # same camera = not cross-camera
                    last = gt.sightings[-1] if gt.sightings else None
                    if last and last["camera"] != sig.camera:
                        logger.debug(
                            f"Cross-cam face name match: {sig.face_name} -> {gid} (cam={sig.camera})"
                        )
                        return gid

        # --- Strategy 2: face embedding similarity ---
        if sig.face_embedding is not None:
            best_gid = None
            best_face_score = 0.0
            for gid, gt in self._global_tracks.items():
                if gt.label != "person" or gt.face_embedding is None:
                    continue
                last = gt.sightings[-1] if gt.sightings else None
                if last and last["camera"] == sig.camera:
                    continue
                score = face_embedding_similarity(sig.face_embedding, gt.face_embedding)
                if score > best_face_score:
                    best_face_score = score
                    best_gid = gid
            if best_face_score > 0.7 and best_gid:
                logger.debug(
                    f"Cross-cam face embedding match: score={best_face_score:.2f} "
                    f"-> {best_gid} (cam={sig.camera})"
                )
                return best_gid

        # --- Strategy 3: clothing color + body ratio with topology ---
        candidates = self._get_topology_candidates(sig, label="person")
        if not candidates:
            return None

        best_gid = None
        best_combined = 0.0

        for gid, gt in candidates:
            # upper body color
            upper_name_score = 1.0 if (
                sig.upper_color != "unknown"
                and gt.upper_color != "unknown"
                and sig.upper_color == gt.upper_color
            ) else 0.0
            upper_hist_score = color_histogram_similarity(
                sig.upper_histogram, gt.upper_histogram
            )
            upper_score = max(upper_name_score, upper_hist_score)

            # lower body color
            lower_name_score = 1.0 if (
                sig.lower_color != "unknown"
                and gt.lower_color != "unknown"
                and sig.lower_color == gt.lower_color
            ) else 0.0
            lower_hist_score = color_histogram_similarity(
                sig.lower_histogram, gt.lower_histogram
            )
            lower_score = max(lower_name_score, lower_hist_score)

            # body ratio similarity (closer ratio = more similar)
            ratio_score = 0.0
            if sig.body_ratio > 0 and gt.body_ratio > 0:
                diff = abs(sig.body_ratio - gt.body_ratio)
                ratio_score = max(0.0, 1.0 - diff / 2.0)

            # face embedding bonus (if available but below hard threshold)
            face_score = 0.0
            if sig.face_embedding is not None and gt.face_embedding is not None:
                face_score = face_embedding_similarity(sig.face_embedding, gt.face_embedding)

            combined = (
                self.config.person_upper_color_weight * upper_score
                + self.config.person_lower_color_weight * lower_score
                + self.config.person_body_ratio_weight * ratio_score
                + self.config.person_face_weight * face_score
            )

            if combined > best_combined:
                best_combined = combined
                best_gid = gid

        if best_combined >= self.config.person_match_threshold and best_gid:
            gt = self._global_tracks[best_gid]
            logger.debug(
                f"Cross-cam person appearance match: upper={sig.upper_color} lower={sig.lower_color} "
                f"-> {best_gid} (face={gt.face_name}, score={best_combined:.2f}, cam={sig.camera})"
            )
            return best_gid

        return None

    def _get_topology_candidates(
        self, sig, label: str = "car"
    ) -> list[tuple[str, GlobalTrack]]:
        """Global tracks that could plausibly be the same object."""
        results = []
        for gid, gt in self._global_tracks.items():
            if gt.label != label:
                continue
            if not gt.sightings:
                continue

            last = gt.sightings[-1]
            last_cam = last["camera"]
            last_time = last["last_seen"]

            if last_cam == sig.camera:
                continue

            elapsed = sig.last_seen - last_time
            if elapsed < 0:
                continue

            link = self._topology.get((last_cam, sig.camera))
            if self._topology:
                if link is None:
                    continue
                if elapsed < link[0] or elapsed > link[1]:
                    continue
            else:
                if elapsed > self.config.gallery_max_age:
                    continue

            results.append((gid, gt))
        return results

    # ------------------------------------------------------------------
    # Internal: create / update
    # ------------------------------------------------------------------


    def _create_global(self, sig) -> str:
        gid = f"xc-{self._next_id}"
        self._next_id += 1

        sighting = {
            "camera": sig.camera,
            "local_track_id": sig.local_track_id,
            "first_seen": sig.first_seen,
            "last_seen": sig.last_seen,
            "box": sig.box,
        }

        if isinstance(sig, PersonSignature):
            gt = GlobalTrack(
                global_id=gid,
                label="person",
                upper_color=sig.upper_color,
                lower_color=sig.lower_color,
                upper_histogram=sig.upper_histogram,
                lower_histogram=sig.lower_histogram,
                body_ratio=sig.body_ratio,
                face_embedding=sig.face_embedding,
                face_name=sig.face_name,
                created_at=sig.first_seen,
                updated_at=sig.last_seen,
                sightings=[sighting],
            )
            logger.debug(
                f"New global person {gid}: upper={gt.upper_color} lower={gt.lower_color} "
                f"face={gt.face_name} cam={sig.camera}"
            )
        else:
            gt = GlobalTrack(
                global_id=gid,
                label="car",
                plate=sig.plate.upper().strip() if sig.plate else None,
                plate_score=sig.plate_score,
                color=sig.color,
                vehicle_type=sig.vehicle_type,
                color_histogram=sig.color_histogram,
                created_at=sig.first_seen,
                updated_at=sig.last_seen,
                sightings=[sighting],
            )
            if gt.plate:
                self._plate_index[gt.plate] = gid
            logger.debug(
                f"New global vehicle {gid}: plate={gt.plate} color={gt.color} "
                f"type={gt.vehicle_type} cam={sig.camera}"
            )

        self._global_tracks[gid] = gt
        return gid

    def _touch_global(
        self,
        gid: str,
        camera: str,
        local_track_id: str,
        label: str,
        frame_time: float,
        box: list[int],
        plate: Optional[str],
        plate_score: float,
        crop_bgr: Optional[np.ndarray],
        face_embedding: Optional[np.ndarray] = None,
        face_name: Optional[str] = None,
    ) -> None:
        """Update an existing global track with fresh data."""
        gt = self._global_tracks.get(gid)
        if not gt:
            return

        gt.updated_at = frame_time

        if label == "person":
            # update clothing colors
            if crop_bgr is not None and crop_bgr.size > 0:
                uc, lc, uh, lh = extract_person_colors(crop_bgr)
                if uc != "unknown":
                    gt.upper_color = uc
                    gt.upper_histogram = uh
                if lc != "unknown":
                    gt.lower_color = lc
                    gt.lower_histogram = lh

            # update body ratio
            br = estimate_body_ratio(box)
            if br > 0:
                gt.body_ratio = br

            # update face
            if face_embedding is not None:
                gt.face_embedding = face_embedding
            if face_name:
                gt.face_name = face_name
        else:
            # vehicle updates
            if plate and plate_score > gt.plate_score:
                old_plate = gt.plate
                gt.plate = plate.upper().strip()
                gt.plate_score = plate_score
                if old_plate and old_plate in self._plate_index:
                    del self._plate_index[old_plate]
                self._plate_index[gt.plate] = gid

            if crop_bgr is not None and crop_bgr.size > 0:
                color_name, color_hist = extract_vehicle_color(crop_bgr)
                if color_name != "unknown":
                    gt.color = color_name
                    gt.color_histogram = color_hist

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

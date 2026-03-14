"""Tests for cross-camera vehicle tracking."""

import numpy as np
import pytest

from frigate.config.cross_camera import CameraLink, CrossCameraConfig
from frigate.track.cross_camera import (
    CrossCameraTracker,
    color_histogram_similarity,
    estimate_vehicle_type,
    extract_vehicle_color,
    plate_similarity,
)


# ---------------------------------------------------------------------------
# plate_similarity
# ---------------------------------------------------------------------------


class TestPlateSimilarity:
    def test_exact_match(self):
        assert plate_similarity("京A12345", "京A12345") == 1.0

    def test_case_insensitive(self):
        assert plate_similarity("ABC123", "abc123") == 1.0

    def test_one_char_diff(self):
        score = plate_similarity("京A12345", "京A12346")
        assert score == 0.85

    def test_two_char_diff(self):
        score = plate_similarity("京A12345", "京A12356")
        assert score == 0.5

    def test_suffix_match(self):
        score = plate_similarity("京A12345", "沪B12345")
        # last 4 chars "2345" match
        assert score == 0.6

    def test_no_match(self):
        assert plate_similarity("京A12345", "沪BXXXXX") == 0.0

    def test_none_input(self):
        assert plate_similarity(None, "京A12345") == 0.0
        assert plate_similarity("京A12345", None) == 0.0
        assert plate_similarity(None, None) == 0.0

    def test_empty_input(self):
        assert plate_similarity("", "京A12345") == 0.0
        assert plate_similarity("京A12345", "") == 0.0


# ---------------------------------------------------------------------------
# estimate_vehicle_type
# ---------------------------------------------------------------------------


class TestEstimateVehicleType:
    def test_sedan(self):
        # typical sedan: wider than tall, ratio ~1.5
        assert estimate_vehicle_type([0, 0, 300, 200]) == "sedan"

    def test_suv(self):
        # SUV: taller relative to width, h > w * 0.85
        assert estimate_vehicle_type([0, 0, 200, 250]) == "suv"

    def test_truck(self):
        # truck: very wide, ratio > 1.8
        assert estimate_vehicle_type([0, 0, 400, 200]) == "truck"

    def test_bus(self):
        # bus: extremely wide, ratio > 2.2
        assert estimate_vehicle_type([0, 0, 500, 200]) == "bus"

    def test_empty_box(self):
        assert estimate_vehicle_type([]) == "unknown"

    def test_zero_size(self):
        assert estimate_vehicle_type([0, 0, 0, 0]) == "unknown"


# ---------------------------------------------------------------------------
# extract_vehicle_color
# ---------------------------------------------------------------------------


class TestExtractVehicleColor:
    def test_white_vehicle(self):
        # create a bright, low-saturation image (white)
        crop = np.full((90, 120, 3), [240, 240, 240], dtype=np.uint8)
        color, hist = extract_vehicle_color(crop)
        assert color == "white"
        assert hist is not None
        assert len(hist) == 12

    def test_black_vehicle(self):
        crop = np.full((90, 120, 3), [20, 20, 20], dtype=np.uint8)
        color, hist = extract_vehicle_color(crop)
        assert color == "black"

    def test_red_vehicle(self):
        # BGR red = (0, 0, 255)
        crop = np.full((90, 120, 3), [0, 0, 255], dtype=np.uint8)
        color, hist = extract_vehicle_color(crop)
        assert color == "red"

    def test_blue_vehicle(self):
        # BGR blue = (255, 0, 0)
        crop = np.full((90, 120, 3), [255, 0, 0], dtype=np.uint8)
        color, hist = extract_vehicle_color(crop)
        assert color == "blue"

    def test_empty_crop(self):
        crop = np.array([], dtype=np.uint8)
        color, hist = extract_vehicle_color(crop)
        assert color == "unknown"

    def test_none_crop(self):
        color, hist = extract_vehicle_color(None)
        assert color == "unknown"


# ---------------------------------------------------------------------------
# color_histogram_similarity
# ---------------------------------------------------------------------------


class TestColorHistogramSimilarity:
    def test_identical(self):
        h = np.array([0.5, 0.3, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        assert color_histogram_similarity(h, h) == pytest.approx(1.0)

    def test_different(self):
        h1 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        h2 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
        score = color_histogram_similarity(h1, h2)
        assert score < 0.5

    def test_none_input(self):
        h = np.zeros(12, dtype=np.float32)
        assert color_histogram_similarity(None, h) == 0.0
        assert color_histogram_similarity(h, None) == 0.0

    def test_zero_histograms(self):
        h = np.zeros(12, dtype=np.float32)
        assert color_histogram_similarity(h, h) == 0.0


# ---------------------------------------------------------------------------
# CrossCameraTracker
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> CrossCameraConfig:
    defaults = {
        "enabled": True,
        "tracked_objects": ["car"],
        "plate_match_threshold": 0.8,
        "color_weight": 0.4,
        "type_weight": 0.3,
        "appearance_weight": 0.3,
        "match_threshold": 0.55,
        "gallery_max_age": 300,
        "topology": [],
    }
    defaults.update(overrides)
    return CrossCameraConfig(**defaults)


def _white_crop():
    return np.full((90, 120, 3), [240, 240, 240], dtype=np.uint8)


def _red_crop():
    return np.full((90, 120, 3), [0, 0, 255], dtype=np.uint8)


class TestCrossCameraTracker:
    def test_new_track_created(self):
        tracker = CrossCameraTracker(_make_config())
        gid = tracker.on_track_update(
            camera="cam1",
            local_track_id="t1",
            label="car",
            frame_time=1000.0,
            box=[100, 100, 400, 300],
            plate="京A12345",
            plate_score=0.95,
        )
        assert gid is not None
        assert gid.startswith("xc-")
        tracks = tracker.get_global_tracks()
        assert len(tracks) == 1
        assert tracks[gid]["plate"] == "京A12345"

    def test_plate_exact_match_across_cameras(self):
        tracker = CrossCameraTracker(_make_config())

        gid1 = tracker.on_track_update(
            camera="cam1",
            local_track_id="t1",
            label="car",
            frame_time=1000.0,
            box=[100, 100, 400, 300],
            plate="京A12345",
            plate_score=0.95,
        )
        tracker.on_track_end("cam1", "t1")

        gid2 = tracker.on_track_update(
            camera="cam2",
            local_track_id="t2",
            label="car",
            frame_time=1010.0,
            box=[200, 150, 500, 350],
            plate="京A12345",
            plate_score=0.90,
        )

        assert gid1 == gid2
        tracks = tracker.get_global_tracks()
        assert len(tracks) == 1
        assert len(tracks[gid1]["sightings"]) == 2

    def test_plate_fuzzy_match(self):
        tracker = CrossCameraTracker(_make_config())

        gid1 = tracker.on_track_update(
            camera="cam1",
            local_track_id="t1",
            label="car",
            frame_time=1000.0,
            box=[100, 100, 400, 300],
            plate="京A12345",
            plate_score=0.95,
        )
        tracker.on_track_end("cam1", "t1")

        # 1 char OCR error
        gid2 = tracker.on_track_update(
            camera="cam2",
            local_track_id="t2",
            label="car",
            frame_time=1010.0,
            box=[200, 150, 500, 350],
            plate="京A12346",
            plate_score=0.80,
        )

        assert gid1 == gid2

    def test_no_plate_color_match(self):
        """Without plate, same color + type should match (no topology = open)."""
        tracker = CrossCameraTracker(_make_config())

        gid1 = tracker.on_track_update(
            camera="cam1",
            local_track_id="t1",
            label="car",
            frame_time=1000.0,
            box=[100, 100, 400, 300],
            crop_bgr=_white_crop(),
        )
        tracker.on_track_end("cam1", "t1")

        gid2 = tracker.on_track_update(
            camera="cam2",
            local_track_id="t2",
            label="car",
            frame_time=1010.0,
            box=[100, 100, 400, 300],
            crop_bgr=_white_crop(),
        )

        assert gid1 == gid2

    def test_no_plate_different_color_no_match(self):
        """Different colors should not match."""
        tracker = CrossCameraTracker(_make_config())

        gid1 = tracker.on_track_update(
            camera="cam1",
            local_track_id="t1",
            label="car",
            frame_time=1000.0,
            box=[100, 100, 400, 300],
            crop_bgr=_white_crop(),
        )
        tracker.on_track_end("cam1", "t1")

        gid2 = tracker.on_track_update(
            camera="cam2",
            local_track_id="t2",
            label="car",
            frame_time=1010.0,
            box=[100, 100, 400, 300],
            crop_bgr=_red_crop(),
        )

        assert gid1 != gid2
        assert len(tracker.get_global_tracks()) == 2

    def test_topology_constraint(self):
        """Topology should block matches outside time window."""
        config = _make_config(
            topology=[
                CameraLink(
                    source="cam1",
                    target="cam2",
                    min_seconds=5,
                    max_seconds=20,
                    bidirectional=True,
                )
            ]
        )
        tracker = CrossCameraTracker(config)

        gid1 = tracker.on_track_update(
            camera="cam1",
            local_track_id="t1",
            label="car",
            frame_time=1000.0,
            box=[100, 100, 400, 300],
            plate="京A12345",
            plate_score=0.95,
        )
        tracker.on_track_end("cam1", "t1")

        # too fast (2 seconds, min is 5) — but plate match bypasses topology
        gid2 = tracker.on_track_update(
            camera="cam2",
            local_track_id="t2",
            label="car",
            frame_time=1002.0,
            box=[200, 150, 500, 350],
            plate="京A12345",
            plate_score=0.90,
        )
        # plate match always works regardless of topology
        assert gid1 == gid2

    def test_topology_blocks_appearance_match(self):
        """Appearance match should be blocked by topology time window."""
        config = _make_config(
            topology=[
                CameraLink(
                    source="cam1",
                    target="cam2",
                    min_seconds=5,
                    max_seconds=20,
                    bidirectional=True,
                )
            ]
        )
        tracker = CrossCameraTracker(config)

        gid1 = tracker.on_track_update(
            camera="cam1",
            local_track_id="t1",
            label="car",
            frame_time=1000.0,
            box=[100, 100, 400, 300],
            crop_bgr=_white_crop(),
        )
        tracker.on_track_end("cam1", "t1")

        # 2 seconds — too fast for topology (min 5s)
        gid2 = tracker.on_track_update(
            camera="cam2",
            local_track_id="t2",
            label="car",
            frame_time=1002.0,
            box=[100, 100, 400, 300],
            crop_bgr=_white_crop(),
        )

        # should NOT match because topology blocks it
        assert gid1 != gid2

    def test_non_tracked_label_ignored(self):
        tracker = CrossCameraTracker(_make_config(tracked_objects=["car"]))
        gid = tracker.on_track_update(
            camera="cam1",
            local_track_id="t1",
            label="person",
            frame_time=1000.0,
            box=[100, 100, 200, 400],
        )
        assert gid is None

    def test_cleanup_expired(self):
        tracker = CrossCameraTracker(_make_config(gallery_max_age=0.01))
        tracker.on_track_update(
            camera="cam1",
            local_track_id="t1",
            label="car",
            frame_time=1000.0,
            box=[100, 100, 400, 300],
            plate="京A12345",
            plate_score=0.95,
        )
        tracker.on_track_end("cam1", "t1")

        import time
        time.sleep(0.02)

        removed = tracker.cleanup_expired()
        assert removed == 1
        assert len(tracker.get_global_tracks()) == 0

    def test_plate_upgrade(self):
        """Better plate score should update the global track."""
        tracker = CrossCameraTracker(_make_config())

        tracker.on_track_update(
            camera="cam1",
            local_track_id="t1",
            label="car",
            frame_time=1000.0,
            box=[100, 100, 400, 300],
            plate="京A12345",
            plate_score=0.70,
        )

        # same track, better plate score
        tracker.on_track_update(
            camera="cam1",
            local_track_id="t1",
            label="car",
            frame_time=1001.0,
            box=[100, 100, 400, 300],
            plate="京A12345",
            plate_score=0.98,
        )

        tracks = tracker.get_global_tracks()
        assert len(tracks) == 1
        track = list(tracks.values())[0]
        assert track["plate"] == "京A12345"

    def test_same_camera_no_cross_match(self):
        """Objects on the same camera should not cross-match via appearance."""
        tracker = CrossCameraTracker(_make_config())

        gid1 = tracker.on_track_update(
            camera="cam1",
            local_track_id="t1",
            label="car",
            frame_time=1000.0,
            box=[100, 100, 400, 300],
            crop_bgr=_white_crop(),
        )

        gid2 = tracker.on_track_update(
            camera="cam1",
            local_track_id="t2",
            label="car",
            frame_time=1001.0,
            box=[100, 100, 400, 300],
            crop_bgr=_white_crop(),
        )

        # different local tracks on same camera = different global tracks
        assert gid1 != gid2

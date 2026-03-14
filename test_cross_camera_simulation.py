#!/usr/bin/env python3
"""
Cross-Camera Tracking Simulation Test
======================================
Standalone test that validates the full cross-camera tracking engine
without starting Frigate. Uses simulated camera feeds with fake
vehicles and persons.

Run: python3 test_cross_camera_simulation.py
"""

import sys
import time
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Inline the core logic (no Frigate imports needed)
# ---------------------------------------------------------------------------

_HUE_MAP = {
    0: "red", 1: "orange", 2: "yellow", 3: "green", 4: "green",
    5: "cyan", 6: "blue", 7: "blue", 8: "blue", 9: "purple",
    10: "pink", 11: "red",
}


def _region_color(region_bgr):
    if region_bgr is None or region_bgr.size == 0:
        return "unknown", np.zeros(12, dtype=np.float32)
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten().astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    mean_sat = float(np.mean(hsv[:, :, 1]))
    mean_val = float(np.mean(hsv[:, :, 2]))
    if mean_sat < 40:
        if mean_val < 60: return "black", hist
        if mean_val > 180: return "white", hist
        if mean_val > 120: return "silver", hist
        return "gray", hist
    return _HUE_MAP.get(int(np.argmax(hist)), "unknown"), hist


def extract_vehicle_color(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown", np.zeros(12, dtype=np.float32)
    h, w = crop_bgr.shape[:2]
    body = crop_bgr[h // 3:, :, :]
    return _region_color(body)


def extract_person_colors(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        z = np.zeros(12, dtype=np.float32)
        return "unknown", "unknown", z, z
    h, w = crop_bgr.shape[:2]
    mid = h // 2
    uc, uh = _region_color(crop_bgr[:mid, :, :])
    lc, lh = _region_color(crop_bgr[mid:, :, :])
    return uc, lc, uh, lh


def estimate_vehicle_type(box):
    if len(box) < 4: return "unknown"
    w = box[2] - box[0]
    h = box[3] - box[1]
    if w == 0 or h == 0: return "unknown"
    ratio = w / h
    if ratio > 2.2: return "bus"
    if ratio > 1.8: return "truck"
    if h > w * 0.85: return "suv"
    return "sedan"


def estimate_body_ratio(box):
    if len(box) < 4: return 0.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    if w == 0: return 0.0
    return h / w


def plate_similarity(a, b):
    if not a or not b: return 0.0
    a, b = a.upper().strip(), b.upper().strip()
    if a == b: return 1.0
    if len(a) == len(b):
        diffs = sum(ca != cb for ca, cb in zip(a, b))
        if diffs == 1: return 0.85
        if diffs == 2: return 0.5
    if min(len(a), len(b)) >= 4 and a[-4:] == b[-4:]: return 0.6
    return 0.0


def color_histogram_similarity(h1, h2):
    if h1 is None or h2 is None: return 0.0
    if h1.sum() == 0 or h2.sum() == 0: return 0.0
    dist = cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
    return max(0.0, 1.0 - dist)


def face_embedding_similarity(e1, e2):
    if e1 is None or e2 is None: return 0.0
    n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
    if n1 == 0 or n2 == 0: return 0.0
    return float(np.dot(e1, e2) / (n1 * n2))


# -----------------------------------------------------------------
# Lightweight tracker (mirrors CrossCameraTracker logic)
# ---------------------------------------------------------------------------

class SimTracker:
    def __init__(self, topology=None, plate_thresh=0.8, match_thresh=0.55,
                 person_match_thresh=0.50, gallery_max_age=300):
        self._next_id = 1
        self._plate_index = {}
        self._global_tracks = {}
        self._local_to_global = {}
        self._topology = {}
        self.plate_thresh = plate_thresh
        self.match_thresh = match_thresh
        self.person_match_thresh = person_match_thresh
        self.gallery_max_age = gallery_max_age
        if topology:
            for src, dst, mn, mx in topology:
                self._topology[(src, dst)] = (mn, mx)
                self._topology[(dst, src)] = (mn, mx)

    def track_vehicle(self, camera, track_id, frame_time, box,
                      plate=None, plate_score=0.0, crop_bgr=None):
        if track_id in self._local_to_global:
            gid = self._local_to_global[track_id]
            self._update_vehicle(gid, camera, track_id, frame_time, box, plate, plate_score, crop_bgr)
            return gid

        color, color_hist = ("unknown", None)
        if crop_bgr is not None:
            color, color_hist = extract_vehicle_color(crop_bgr)
        vtype = estimate_vehicle_type(box)

        # plate match
        gid = self._plate_match(plate)
        if gid is None:
            gid = self._appearance_match_vehicle(camera, frame_time, color, color_hist, vtype)
        if gid is None:
            gid = self._create(camera, track_id, frame_time, box, "car",
                               plate=plate, plate_score=plate_score,
                               color=color, color_hist=color_hist, vtype=vtype)
        else:
            self._update_vehicle(gid, camera, track_id, frame_time, box, plate, plate_score, crop_bgr)

        self._local_to_global[track_id] = gid
        return gid

    def track_person(self, camera, track_id, frame_time, box,
                     crop_bgr=None, face_name=None, face_embedding=None):
        if track_id in self._local_to_global:
            gid = self._local_to_global[track_id]
            self._update_person(gid, camera, track_id, frame_time, box, crop_bgr, face_name, face_embedding)
            return gid

        uc, lc, uh, lh = ("unknown", "unknown", None, None)
        if crop_bgr is not None:
            uc, lc, uh, lh = extract_person_colors(crop_bgr)
        br = estimate_body_ratio(box)

        # face name match
        gid = self._face_name_match(face_name)
        if gid is None:
            gid = self._appearance_match_person(camera, frame_time, uc, lc, uh, lh, br, face_embedding)
        if gid is None:
            gid = self._create(camera, track_id, frame_time, box, "person",
                               upper_color=uc, lower_color=lc,
                               upper_hist=uh, lower_hist=lh,
                               body_ratio=br, face_name=face_name,
                               face_embedding=face_embedding)
        else:
            self._update_person(gid, camera, track_id, frame_time, box, crop_bgr, face_name, face_embedding)

        self._local_to_global[track_id] = gid
        return gid

    def end_track(self, track_id):
        if track_id in self._local_to_global:
            gid = self._local_to_global.pop(track_id)
            if gid in self._global_tracks:
                self._global_tracks[gid]["updated_at"] = time.time()

    def get_tracks(self):
        return dict(self._global_tracks)

    # --- internal ---

    def _plate_match(self, plate):
        if not plate: return None
        key = plate.upper().strip()
        if key in self._plate_index:
            return self._plate_index[key]
        best_gid, best_s = None, 0.0
        for p, gid in self._plate_index.items():
            s = plate_similarity(plate, p)
            if s > best_s:
                best_s, best_gid = s, gid
        if best_s >= self.plate_thresh:
            return best_gid
        return None

    def _face_name_match(self, face_name):
        if not face_name: return None
        for gid, gt in self._global_tracks.items():
            if gt.get("face_name") == face_name:
                return gid
        return None

    def _topo_candidates(self, camera, frame_time, label):
        results = []
        for gid, gt in self._global_tracks.items():
            if gt.get("label") != label: continue
            sightings = gt.get("sightings", [])
            if not sightings: continue
            last = sightings[-1]
            if last["camera"] == camera: continue
            elapsed = frame_time - last["last_seen"]
            if elapsed < 0: continue
            link = self._topology.get((last["camera"], camera))
            if self._topology:
                if link is None: continue
                if elapsed < link[0] or elapsed > link[1]: continue
            else:
                if elapsed > self.gallery_max_age: continue
            results.append((gid, gt))
        return results

    def _appearance_match_vehicle(self, camera, frame_time, color, color_hist, vtype):
        candidates = self._topo_candidates(camera, frame_time, "car")
        if not candidates: return None
        best_gid, best_score = None, 0.0
        for gid, gt in candidates:
            cs = 1.0 if (color != "unknown" and gt.get("color") != "unknown" and color == gt["color"]) else 0.0
            hs = color_histogram_similarity(color_hist, gt.get("color_hist"))
            # If both color names are known but different, override histogram score
            if color != "unknown" and gt.get("color") != "unknown" and color != gt["color"]:
                cs = 0.0
                hs = 0.0
            else:
                cs = max(cs, hs)
            ts = 1.0 if (vtype != "unknown" and gt.get("vtype") != "unknown" and vtype == gt["vtype"]) else 0.0
            combined = 0.4 * cs + 0.3 * ts + 0.3 * hs
            if combined > best_score:
                best_score, best_gid = combined, gid
        if best_score >= self.match_thresh:
            return best_gid
        return None

    def _appearance_match_person(self, camera, frame_time, uc, lc, uh, lh, br, face_emb):
        candidates = self._topo_candidates(camera, frame_time, "person")
        if not candidates: return None
        best_gid, best_score = None, 0.0
        for gid, gt in candidates:
            # face embedding
            fs = face_embedding_similarity(face_emb, gt.get("face_embedding"))
            # upper color
            ucs = max(
                1.0 if (uc != "unknown" and gt.get("upper_color") != "unknown" and uc == gt["upper_color"]) else 0.0,
                color_histogram_similarity(uh, gt.get("upper_hist"))
            )
            # lower color
            lcs = max(
                1.0 if (lc != "unknown" and gt.get("lower_color") != "unknown" and lc == gt["lower_color"]) else 0.0,
                color_histogram_similarity(lh, gt.get("lower_hist"))
            )
            # body ratio
            brs = 0.0
            if br > 0 and gt.get("body_ratio", 0) > 0:
                diff = abs(br - gt["body_ratio"]) / max(br, gt["body_ratio"])
                brs = max(0.0, 1.0 - diff * 2)
            combined = 0.35 * ucs + 0.25 * lcs + 0.15 * brs + 0.25 * fs
            if combined > best_score:
                best_score, best_gid = combined, gid
        if best_score >= self.person_match_thresh:
            return best_gid
        return None

    def _create(self, camera, track_id, frame_time, box, label, **kwargs):
        gid = f"xc-{self._next_id}"
        self._next_id += 1
        gt = {
            "global_id": gid, "label": label,
            "sightings": [{"camera": camera, "track_id": track_id,
                           "first_seen": frame_time, "last_seen": frame_time, "box": box}],
            "created_at": frame_time, "updated_at": frame_time,
        }
        gt.update(kwargs)
        self._global_tracks[gid] = gt
        plate = kwargs.get("plate")
        if plate:
            self._plate_index[plate.upper().strip()] = gid
        return gid

    def _update_vehicle(self, gid, camera, track_id, frame_time, box, plate, plate_score, crop_bgr):
        gt = self._global_tracks.get(gid)
        if not gt: return
        gt["updated_at"] = frame_time
        if plate and plate_score > gt.get("plate_score", 0):
            old = gt.get("plate")
            if old and old in self._plate_index: del self._plate_index[old]
            gt["plate"] = plate.upper().strip()
            gt["plate_score"] = plate_score
            self._plate_index[gt["plate"]] = gid
        if crop_bgr is not None:
            c, h = extract_vehicle_color(crop_bgr)
            if c != "unknown":
                gt["color"] = c
                gt["color_hist"] = h
        vt = estimate_vehicle_type(box)
        if vt != "unknown": gt["vtype"] = vt
        self._append_sighting(gt, camera, track_id, frame_time, box)

    def _update_person(self, gid, camera, track_id, frame_time, box, crop_bgr, face_name, face_emb):
        gt = self._global_tracks.get(gid)
        if not gt: return
        gt["updated_at"] = frame_time
        if face_name: gt["face_name"] = face_name
        if face_emb is not None: gt["face_embedding"] = face_emb
        if crop_bgr is not None:
            uc, lc, uh, lh = extract_person_colors(crop_bgr)
            if uc != "unknown": gt["upper_color"] = uc; gt["upper_hist"] = uh
            if lc != "unknown": gt["lower_color"] = lc; gt["lower_hist"] = lh
        gt["body_ratio"] = estimate_body_ratio(box)
        self._append_sighting(gt, camera, track_id, frame_time, box)

    def _append_sighting(self, gt, camera, track_id, frame_time, box):
        sightings = gt["sightings"]
        if sightings and sightings[-1]["camera"] == camera:
            sightings[-1]["last_seen"] = frame_time
            sightings[-1]["box"] = box
        else:
            sightings.append({"camera": camera, "track_id": track_id,
                              "first_seen": frame_time, "last_seen": frame_time, "box": box})


# ---------------------------------------------------------------------------
# Helper: create fake image crops
# ---------------------------------------------------------------------------

def make_car_crop(color_bgr):
    crop = np.full((120, 200, 3), color_bgr, dtype=np.uint8)
    # add windshield area (dark top 1/3)
    crop[:40, :, :] = [40, 40, 40]
    return crop

def make_person_crop(upper_bgr, lower_bgr):
    crop = np.zeros((300, 100, 3), dtype=np.uint8)
    crop[:150, :, :] = upper_bgr
    crop[150:, :, :] = lower_bgr
    return crop


# ===========================================================================
# SIMULATION SCENARIOS
# ===========================================================================

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_result(test_name, passed, detail=""):
    icon = "✅" if passed else "❌"
    print(f"  {icon} {test_name}" + (f" — {detail}" if detail else ""))

passed_count = 0
failed_count = 0

def check(name, condition, detail=""):
    global passed_count, failed_count
    if condition:
        passed_count += 1
    else:
        failed_count += 1
    print_result(name, condition, detail)


# ---------------------------------------------------------------------------
# Scenario 1: Vehicle plate exact match across cameras
# ---------------------------------------------------------------------------
print_header("Scenario 1: Vehicle — Plate Exact Match")

tracker = SimTracker(topology=[("cam_gate", "cam_parking", 5, 30)])

white_car = make_car_crop([240, 240, 240])

gid1 = tracker.track_vehicle("cam_gate", "v1", 1000.0, [100, 100, 400, 300],
                              plate="京A12345", plate_score=0.95, crop_bgr=white_car)
tracker.end_track("v1")

gid2 = tracker.track_vehicle("cam_parking", "v2", 1012.0, [200, 150, 500, 350],
                              plate="京A12345", plate_score=0.90, crop_bgr=white_car)

check("Same plate -> same global ID", gid1 == gid2, f"{gid1} == {gid2}")
tracks = tracker.get_tracks()
check("Only 1 global track", len(tracks) == 1)
check("2 sightings (2 cameras)", len(tracks[gid1]["sightings"]) == 2)
print(f"  Track: plate={tracks[gid1].get('plate')}, color={tracks[gid1].get('color')}")


# ---------------------------------------------------------------------------
# Scenario 2: Vehicle plate fuzzy match (1 char OCR error)
# ---------------------------------------------------------------------------
print_header("Scenario 2: Vehicle — Plate Fuzzy Match (OCR Error)")

tracker2 = SimTracker()

gid1 = tracker2.track_vehicle("cam1", "v1", 1000.0, [100, 100, 400, 300],
                               plate="沪B67890", plate_score=0.95)
tracker2.end_track("v1")

gid2 = tracker2.track_vehicle("cam2", "v2", 1010.0, [200, 150, 500, 350],
                               plate="沪B67891", plate_score=0.80)  # 1 char diff

check("1-char OCR error -> same global ID", gid1 == gid2, f"{gid1} == {gid2}")


# ---------------------------------------------------------------------------
# Scenario 3: Vehicle no plate — color + type match
# ---------------------------------------------------------------------------
print_header("Scenario 3: Vehicle — No Plate, Color + Type Match")

tracker3 = SimTracker()

red_sedan = make_car_crop([0, 0, 220])  # red BGR

gid1 = tracker3.track_vehicle("cam1", "v1", 1000.0, [100, 100, 400, 300],
                               crop_bgr=red_sedan)
tracker3.end_track("v1")

gid2 = tracker3.track_vehicle("cam2", "v2", 1010.0, [100, 100, 400, 300],
                               crop_bgr=red_sedan)

check("Same color+type, no plate -> same global ID", gid1 == gid2, f"{gid1} == {gid2}")


# ---------------------------------------------------------------------------
# Scenario 4: Vehicle no plate — different color = no match
# ---------------------------------------------------------------------------
print_header("Scenario 4: Vehicle — Different Color = No Match")

tracker4 = SimTracker()

gid1 = tracker4.track_vehicle("cam1", "v1", 1000.0, [100, 100, 400, 300],
                               crop_bgr=make_car_crop([240, 240, 240]))  # white
tracker4.end_track("v1")

gid2 = tracker4.track_vehicle("cam2", "v2", 1010.0, [100, 100, 400, 300],
                               crop_bgr=make_car_crop([0, 0, 220]))  # red

check("Different color -> different global ID", gid1 != gid2, f"{gid1} != {gid2}")
check("2 global tracks", len(tracker4.get_tracks()) == 2)


# ---------------------------------------------------------------------------
# Scenario 5: Vehicle topology blocks match (too fast)
# ---------------------------------------------------------------------------
print_header("Scenario 5: Vehicle — Topology Blocks (Too Fast)")

tracker5 = SimTracker(topology=[("cam1", "cam2", 10, 30)])

gid1 = tracker5.track_vehicle("cam1", "v1", 1000.0, [100, 100, 400, 300],
                               crop_bgr=make_car_crop([240, 240, 240]))
tracker5.end_track("v1")

# only 3 seconds later — too fast (min is 10s)
gid2 = tracker5.track_vehicle("cam2", "v2", 1003.0, [100, 100, 400, 300],
                               crop_bgr=make_car_crop([240, 240, 240]))

check("Too fast for topology -> different global ID", gid1 != gid2, f"{gid1} != {gid2}")


# ---------------------------------------------------------------------------
# Scenario 6: Person — face name match across cameras
# ---------------------------------------------------------------------------
print_header("Scenario 6: Person — Face Name Match")

tracker6 = SimTracker()

person_crop = make_person_crop([0, 0, 200], [200, 100, 0])  # red shirt, blue pants

gid1 = tracker6.track_person("cam_door", "p1", 1000.0, [100, 50, 200, 450],
                              crop_bgr=person_crop, face_name="张三")
tracker6.end_track("p1")

gid2 = tracker6.track_person("cam_hall", "p2", 1008.0, [150, 60, 250, 460],
                              crop_bgr=person_crop, face_name="张三")

check("Same face name -> same global ID", gid1 == gid2, f"{gid1} == {gid2}")
tracks = tracker6.get_tracks()
check("Person track has face_name", tracks[gid1].get("face_name") == "张三")
check("2 sightings", len(tracks[gid1]["sightings"]) == 2)


# ---------------------------------------------------------------------------
# Scenario 7: Person — clothing color match (no face)
# ---------------------------------------------------------------------------
print_header("Scenario 7: Person — Clothing Color Match (No Face)")

tracker7 = SimTracker()

red_shirt_blue_pants = make_person_crop([0, 0, 200], [200, 0, 0])

gid1 = tracker7.track_person("cam1", "p1", 1000.0, [100, 50, 200, 450],
                              crop_bgr=red_shirt_blue_pants)
tracker7.end_track("p1")

gid2 = tracker7.track_person("cam2", "p2", 1010.0, [100, 50, 200, 450],
                              crop_bgr=red_shirt_blue_pants)

check("Same clothing -> same global ID", gid1 == gid2, f"{gid1} == {gid2}")


# ---------------------------------------------------------------------------
# Scenario 8: Person — different clothing = no match
# ---------------------------------------------------------------------------
print_header("Scenario 8: Person — Different Clothing = No Match")

tracker8 = SimTracker()

gid1 = tracker8.track_person("cam1", "p1", 1000.0, [100, 50, 200, 450],
                              crop_bgr=make_person_crop([0, 0, 200], [200, 0, 0]))  # red+blue
tracker8.end_track("p1")

gid2 = tracker8.track_person("cam2", "p2", 1010.0, [100, 50, 200, 450],
                              crop_bgr=make_person_crop([0, 200, 0], [0, 200, 200]))  # green+yellow

check("Different clothing -> different global ID", gid1 != gid2, f"{gid1} != {gid2}")


# ---------------------------------------------------------------------------
# Scenario 9: Person — face embedding match
# ---------------------------------------------------------------------------
print_header("Scenario 9: Person — Face Embedding Match")

tracker9 = SimTracker()

face_vec = np.random.randn(128).astype(np.float32)
face_vec_noisy = face_vec + np.random.randn(128).astype(np.float32) * 0.05  # slight noise

gid1 = tracker9.track_person("cam1", "p1", 1000.0, [100, 50, 200, 450],
                              crop_bgr=make_person_crop([0, 0, 200], [200, 0, 0]),
                              face_embedding=face_vec)
tracker9.end_track("p1")

gid2 = tracker9.track_person("cam2", "p2", 1010.0, [100, 50, 200, 450],
                              crop_bgr=make_person_crop([0, 0, 200], [200, 0, 0]),
                              face_embedding=face_vec_noisy)

sim = face_embedding_similarity(face_vec, face_vec_noisy)
check("Similar face embedding -> same global ID", gid1 == gid2,
      f"{gid1} == {gid2}, cosine_sim={sim:.3f}")


# ---------------------------------------------------------------------------
# Scenario 10: Mixed — vehicles and persons don't cross-match
# ---------------------------------------------------------------------------
print_header("Scenario 10: Mixed — Vehicle and Person Don't Cross-Match")

tracker10 = SimTracker()

gid_car = tracker10.track_vehicle("cam1", "v1", 1000.0, [100, 100, 400, 300],
                                   crop_bgr=make_car_crop([240, 240, 240]))
tracker10.end_track("v1")

gid_person = tracker10.track_person("cam2", "p1", 1010.0, [100, 50, 200, 450],
                                     crop_bgr=make_person_crop([240, 240, 240], [240, 240, 240]))

check("Car and person -> different global IDs", gid_car != gid_person,
      f"car={gid_car}, person={gid_person}")
check("2 global tracks (1 car + 1 person)", len(tracker10.get_tracks()) == 2)


# ---------------------------------------------------------------------------
# Scenario 11: Multi-camera chain (3 cameras)
# ---------------------------------------------------------------------------
print_header("Scenario 11: Multi-Camera Chain (3 Cameras)")

tracker11 = SimTracker(topology=[
    ("cam_a", "cam_b", 5, 20),
    ("cam_b", "cam_c", 5, 20),
])

blue_suv = make_car_crop([200, 0, 0])  # blue

gid1 = tracker11.track_vehicle("cam_a", "v1", 1000.0, [100, 100, 300, 350],
                                plate="粤C55555", plate_score=0.95, crop_bgr=blue_suv)
tracker11.end_track("v1")

gid2 = tracker11.track_vehicle("cam_b", "v2", 1012.0, [100, 100, 300, 350],
                                plate="粤C55555", plate_score=0.90, crop_bgr=blue_suv)
tracker11.end_track("v2")

gid3 = tracker11.track_vehicle("cam_c", "v3", 1025.0, [100, 100, 300, 350],
                                plate="粤C55555", plate_score=0.88, crop_bgr=blue_suv)

check("3 cameras, same plate -> same global ID", gid1 == gid2 == gid3,
      f"{gid1} == {gid2} == {gid3}")
tracks = tracker11.get_tracks()
check("3 sightings", len(tracks[gid1]["sightings"]) == 3)
cams = [s["camera"] for s in tracks[gid1]["sightings"]]
check("Cameras: a -> b -> c", cams == ["cam_a", "cam_b", "cam_c"], str(cams))


# ===========================================================================
# SUMMARY
# ===========================================================================
print(f"\n{'='*60}")
print(f"  SIMULATION RESULTS")
print(f"{'='*60}")
print(f"  Passed: {passed_count}")
print(f"  Failed: {failed_count}")
print(f"  Total:  {passed_count + failed_count}")
print(f"{'='*60}")

if failed_count > 0:
    print("\n  ⚠️  Some tests failed!")
    sys.exit(1)
else:
    print("\n  🎉 All simulation tests passed!")
    sys.exit(0)

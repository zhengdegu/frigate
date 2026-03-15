#!/usr/bin/env python3
"""
Cross-Camera Tracking Demo
===========================
Standalone web demo that simulates multi-camera vehicle & person tracking.
No Frigate, FFmpeg, or GPU needed — just Python + OpenCV + FastAPI.

Run:  python3 demo/app.py
Open: http://localhost:8899
"""

import json
import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn

# =========================================================================
# Cross-Camera Tracker (standalone, same logic as frigate/track/cross_camera.py)
# =========================================================================

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


class CrossCameraTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._next_id = 1
        self._plate_index = {}
        self._global_tracks = {}
        self._local_to_global = {}
        self._events = []  # recent cross-camera events

    def track_vehicle(self, camera, track_id, frame_time, box,
                      plate=None, color_bgr=None):
        with self._lock:
            if track_id in self._local_to_global:
                gid = self._local_to_global[track_id]
                self._update(gid, camera, track_id, frame_time, box)
                return gid

            color = "unknown"
            color_hist = None
            if color_bgr is not None:
                crop = np.full((90, 120, 3), color_bgr, dtype=np.uint8)
                color, color_hist = extract_vehicle_color(crop)

            vtype = estimate_vehicle_type(box)

            # plate match
            gid = None
            if plate:
                key = plate.upper().strip()
                if key in self._plate_index:
                    gid = self._plate_index[key]
                else:
                    for p, g in self._plate_index.items():
                        if plate_similarity(plate, p) >= 0.8:
                            gid = g
                            break

            # appearance match
            if gid is None:
                best_gid, best_score = None, 0.0
                for g, gt in self._global_tracks.items():
                    if gt["label"] != "car": continue
                    sightings = gt["sightings"]
                    if not sightings: continue
                    last = sightings[-1]
                    if last["camera"] == camera: continue
                    elapsed = frame_time - last["last_seen"]
                    if elapsed < 0 or elapsed > 60: continue

                    cs = 1.0 if (color != "unknown" and gt.get("color") != "unknown" and color == gt["color"]) else 0.0
                    hs = color_histogram_similarity(color_hist, gt.get("color_hist"))
                    if color != "unknown" and gt.get("color") != "unknown" and color != gt["color"]:
                        cs = 0.0
                        hs = 0.0
                    else:
                        cs = max(cs, hs)
                    ts = 1.0 if (vtype != "unknown" and gt.get("vtype") != "unknown" and vtype == gt["vtype"]) else 0.0
                    combined = 0.4 * cs + 0.3 * ts + 0.3 * hs
                    if combined > best_score:
                        best_score, best_gid = combined, g
                if best_score >= 0.55:
                    gid = best_gid

            if gid is None:
                gid = self._create(camera, track_id, frame_time, box, "car",
                                   plate=plate, color=color, color_hist=color_hist, vtype=vtype)
            else:
                self._update(gid, camera, track_id, frame_time, box)
                gt = self._global_tracks[gid]
                if len(gt["sightings"]) > 1:
                    cams = [s["camera"] for s in gt["sightings"]]
                    self._events.append({
                        "time": time.time(),
                        "global_id": gid,
                        "plate": gt.get("plate"),
                        "color": gt.get("color"),
                        "type": "cross_camera_match",
                        "cameras": cams,
                    })
                    if len(self._events) > 50:
                        self._events = self._events[-50:]

            self._local_to_global[track_id] = gid
            return gid

    def end_track(self, track_id):
        with self._lock:
            self._local_to_global.pop(track_id, None)

    def get_tracks(self):
        with self._lock:
            return dict(self._global_tracks)

    def get_events(self):
        with self._lock:
            return list(self._events)

    def get_stats(self):
        with self._lock:
            tracks = self._global_tracks
            return {
                "total": len(tracks),
                "with_plate": sum(1 for t in tracks.values() if t.get("plate")),
                "cross_camera": sum(1 for t in tracks.values() if len(t.get("sightings", [])) > 1),
                "cameras": list(set(
                    s["camera"] for t in tracks.values() for s in t.get("sightings", [])
                )),
            }

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

    def _update(self, gid, camera, track_id, frame_time, box):
        gt = self._global_tracks.get(gid)
        if not gt: return
        gt["updated_at"] = frame_time
        sightings = gt["sightings"]
        if sightings and sightings[-1]["camera"] == camera:
            sightings[-1]["last_seen"] = frame_time
            sightings[-1]["box"] = box
        else:
            sightings.append({"camera": camera, "track_id": track_id,
                              "first_seen": frame_time, "last_seen": frame_time, "box": box})


# =========================================================================
# Simulated Vehicles
# =========================================================================

PLATES = ["京A12345", "沪B67890", "粤C55555", "川D99999", "浙E11111"]
COLORS_BGR = {
    "white": [240, 240, 240],
    "red": [0, 0, 220],
    "blue": [220, 0, 0],
    "black": [30, 30, 30],
    "silver": [180, 180, 180],
    "green": [0, 180, 0],
    "yellow": [0, 220, 220],
}
COLOR_NAMES = list(COLORS_BGR.keys())

CAM_W, CAM_H = 640, 360

@dataclass
class SimVehicle:
    vid: str
    plate: str
    color_name: str
    color_bgr: list
    w: int = 80
    h: int = 50
    # per-camera state
    cam_state: dict = field(default_factory=dict)

@dataclass
class CamVehicleState:
    x: float = 0.0
    y: float = 0.0
    vx: float = 2.0
    vy: float = 0.0
    visible: bool = False
    enter_time: float = 0.0
    exit_time: float = 0.0


class Simulation:
    def __init__(self, tracker: CrossCameraTracker):
        self.tracker = tracker
        self.cameras = ["cam_gate", "cam_road", "cam_parking"]
        self.vehicles = []
        self.frame_count = 0
        self.start_time = time.time()
        self._init_vehicles()

    def _init_vehicles(self):
        for i in range(5):
            plate = PLATES[i]
            cn = COLOR_NAMES[i % len(COLOR_NAMES)]
            v = SimVehicle(
                vid=f"v{i}",
                plate=plate,
                color_name=cn,
                color_bgr=COLORS_BGR[cn],
                w=random.randint(70, 100),
                h=random.randint(40, 60),
            )
            # init cam states with staggered entry
            for ci, cam in enumerate(self.cameras):
                st = CamVehicleState()
                st.x = -v.w - random.randint(50, 200)
                st.y = 120 + i * 45 + random.randint(-10, 10)
                st.vx = random.uniform(1.5, 3.5)
                st.vy = random.uniform(-0.3, 0.3)
                st.visible = False
                # stagger entry: each camera sees the vehicle at different times
                st.enter_time = self.start_time + i * random.uniform(3, 8) + ci * random.uniform(5, 15)
                v.cam_state[cam] = st
            self.vehicles.append(v)

    def step(self):
        now = time.time()
        self.frame_count += 1

        for v in self.vehicles:
            for cam in self.cameras:
                st = v.cam_state[cam]

                if now < st.enter_time:
                    continue

                if not st.visible:
                    st.visible = True
                    st.x = -v.w

                # move
                st.x += st.vx
                st.y += st.vy
                st.y = max(50, min(CAM_H - v.h - 10, st.y))

                # check bounds
                if st.x > CAM_W + 20:
                    # exited this camera
                    self.tracker.end_track(f"{v.vid}_{cam}")
                    st.visible = False
                    # schedule re-entry after a delay
                    st.enter_time = now + random.uniform(10, 25)
                    st.x = -v.w - random.randint(50, 150)
                    continue

                if st.x + v.w > 0 and st.x < CAM_W:
                    box = [int(st.x), int(st.y), int(st.x + v.w), int(st.y + v.h)]
                    # sometimes plate is not visible (30% chance)
                    plate = v.plate if random.random() > 0.3 else None
                    self.tracker.track_vehicle(
                        camera=cam,
                        track_id=f"{v.vid}_{cam}",
                        frame_time=now,
                        box=box,
                        plate=plate,
                        color_bgr=v.color_bgr,
                    )

    def render_camera(self, cam_name: str) -> np.ndarray:
        frame = np.full((CAM_H, CAM_W, 3), [40, 40, 40], dtype=np.uint8)

        # draw road
        cv2.rectangle(frame, (0, 80), (CAM_W, CAM_H - 30), (60, 60, 60), -1)
        # lane lines
        for x in range(0, CAM_W, 40):
            cv2.line(frame, (x, CAM_H // 2), (x + 20, CAM_H // 2), (100, 100, 100), 2)

        # camera label
        cv2.putText(frame, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)

        for v in self.vehicles:
            st = v.cam_state.get(cam_name)
            if not st or not st.visible: continue
            if st.x + v.w < 0 or st.x > CAM_W: continue

            x1, y1 = int(st.x), int(st.y)
            x2, y2 = x1 + v.w, y1 + v.h

            # draw vehicle body
            cv2.rectangle(frame, (x1, y1), (x2, y2), v.color_bgr, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

            # windshield
            cv2.rectangle(frame, (x1 + 5, y1 + 3), (x2 - 5, y1 + v.h // 3), (80, 80, 80), -1)

            # wheels
            cv2.circle(frame, (x1 + 12, y2), 6, (50, 50, 50), -1)
            cv2.circle(frame, (x2 - 12, y2), 6, (50, 50, 50), -1)

            # plate text
            cv2.putText(frame, v.plate[-4:], (x1 + 5, y2 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

            # global ID label
            tid = f"{v.vid}_{cam_name}"
            gid = self.tracker._local_to_global.get(tid, "?")
            cv2.putText(frame, str(gid), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # timestamp
        cv2.putText(frame, f"Frame: {self.frame_count}", (CAM_W - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return frame


# =========================================================================
# FastAPI App
# ======================================================================

tracker = CrossCameraTracker()
sim = Simulation(tracker)

app = FastAPI(title="Cross-Camera Tracking Demo")

# Background simulation thread
def simulation_loop():
    while True:
        sim.step()
        time.sleep(0.1)  # 10 FPS

sim_thread = threading.Thread(target=simulation_loop, daemon=True)
sim_thread.start()


def gen_mjpeg(cam_name: str):
    while True:
        frame = sim.render_camera(cam_name)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.1)


@app.get("/video/{cam_name}")
def video_feed(cam_name: str):
    return StreamingResponse(gen_mjpeg(cam_name), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/tracks")
def api_tracks():
    return JSONResponse(tracker.get_tracks())


@app.get("/api/events")
def api_events():
    return JSONResponse(tracker.get_events())


@app.get("/api/stats")
def api_stats():
    return JSONResponse(tracker.get_stats())


@app.get("/", response_class=HTMLResponse)
def index():
    cams_html = ""
    for cam in sim.cameras:
        cams_html += f'''
        <div class="cam">
            <img src="/video/{cam}" />
        </div>'''

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Cross-Camera Tracking Demo</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #1a1a2e; color: #eee; font-family: 'Segoe UI', sans-serif; }}
h1 {{ text-align: center; padding: 16px; color: #0ff; font-size: 22px; }}
.cameras {{ display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; padding: 0 12px; }}
.cam {{ border: 2px solid #333; border-radius: 8px; overflow: hidden; }}
.cam img {{ width: 640px; height: 360px; display: block; }}
.panel {{ display: flex; gap: 16px; padding: 16px; justify-content: center; flex-wrap: wrap; }}
.card {{ background: #16213e; border: 1px solid #0f3460; border-radius: 8px; padding: 14px; min-width: 300px; max-width: 500px; flex: 1; }}
.card h3 {{ color: #0ff; margin-bottom: 10px; font-size: 15px; }}
#stats {{ font-size: 14px; line-height: 1.8; }}
#events {{ font-size: 13px; max-height: 250px; overflow-y: auto; }}
.event {{ padding: 4px 0; border-bottom: 1px solid #0f3460; }}
.event .match {{ color: #0f0; font-weight: bold; }}
.event .plate {{ color: #ff0; }}
.event .cams {{ color: #0ff; }}
#tracks {{ font-size: 13px; max-height: 250px; overflow-y: auto; }}
.track {{ padding: 4px 0; border-bottom: 1px solid #0f3460; }}
.gid {{ color: #0f0; font-weight: bold; }}
.tplate {{ color: #ff0; }}
.tcolor {{ color: #f80; }}
.tcams {{ color: #0ff; }}
</style>
</head>
<body>
<h1>🎯 Cross-Camera Vehicle Tracking Demo</h1>
<div class="cameras">{cams_html}</div>
<div class="panel">
    <div class="card">
        <h3>📊 Statistics</h3>
        <div id="stats">Loading...</div>
    </div>
    <div class="card">
        <h3>🔗 Global Tracks</h3>
        <div id="tracks">Loading...</div>
    </div>
    <div class="card">
        <h3>⚡ Cross-Camera Events</h3>
        <div id="events">Waiting for matches...</div>
    </div>
</div>
<script>
async function refresh() {{
    try {{
        const [statsRes, tracksRes, eventsRes] = await Promise.all([
            fetch('/api/stats').then(r => r.json()),
            fetch('/api/tracks').then(r => r.json()),
            fetch('/api/events').then(r => r.json()),
        ]);

        document.getElementById('stats').innerHTML =
            `Total tracks: <b>${{statsRes.total}}</b><br>` +
            `With plate: <b>${{statsRes.with_plate}}</b><br>` +
            `Cross-camera matched: <b style="color:#0f0">${{statsRes.cross_camera}}</b><br>` +
            `Active cameras: <b>${{statsRes.cameras.join(', ')}}</b>`;

        let tracksHtml = '';
        for (const [gid, t] of Object.entries(tracksRes)) {{
            const cams = t.sightings.map(s => s.camera).join(' → ');
            const multi = t.sightings.length > 1 ? ' ⭐' : '';
            tracksHtml += `<div class="track">` +
                `<span class="gid">${{gid}}</span>${{multi}} ` +
                `<span class="tplate">${{t.plate || 'no plate'}}</span> ` +
                `<span class="tcolor">${{t.color || '?'}}</span> ` +
                `<span class="tcams">${{cams}}</span></div>`;
        }}
        document.getElementById('tracks').innerHTML = tracksHtml || 'No tracks yet';

        let evHtml = '';
        for (const e of eventsRes.reverse().slice(0, 20)) {{
            evHtml += `<div class="event">` +
                `<span class="match">MATCH</span> ` +
                `<span class="gid">${{e.global_id}}</span> ` +
                `<span class="plate">${{e.plate || '?'}}</span> ` +
                `<span class="cams">${{e.cameras.join(' → ')}}</span></div>`;
        }}
        document.getElementById('events').innerHTML = evHtml || 'Waiting...';
    }} catch(e) {{}}
}}
setInterval(refresh, 1000);
refresh();
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("=" * 50)
    print("  Cross-Camera Tracking Demo")
    print("  Open: http://localhost:8899")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8899, log_level="warning")

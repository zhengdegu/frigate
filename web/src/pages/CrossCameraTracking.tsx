import { useEffect, useState, useCallback } from "react";
import Heading from "@/components/ui/heading";
import { Button } from "@/components/ui/button";
import axios from "axios";
import TrackMapView from "@/components/TrackMapView";
import CameraSnapshotPanel from "@/components/CameraSnapshotPanel";

interface Sighting {
  camera: string;
  local_track_id: string;
  first_seen: number;
  last_seen: number;
  box: number[];
}

interface GlobalTrack {
  global_id: string;
  label: string;
  plate: string | null;
  color: string;
  vehicle_type: string;
  upper_color: string;
  lower_color: string;
  body_ratio: number;
  face_name: string | null;
  sightings: Sighting[];
  created_at: number;
  updated_at: number;
}

interface Stats {
  total_global_tracks: number;
  tracks_with_plate: number;
  tracks_without_plate: number;
  cameras_involved: string[];
}

interface CameraPathStep {
  camera: string;
  enter_time: number;
  exit_time: number;
}

interface AlertPath {
  alert_id: string;
  alert_camera: string;
  alert_start: number;
  alert_severity: string;
  alert_thumb: string;
  global_id: string;
  label: string;
  upper_color: string | null;
  lower_color: string | null;
  body_ratio: number | null;
  face_name: string | null;
  plate: string | null;
  color: string | null;
  vehicle_type: string | null;
  camera_path: CameraPathStep[];
  cameras_visited: number;
  total_sightings: number;
  first_seen: number;
  last_seen: number;
}

const COLOR_DOTS: Record<string, string> = {
  red: "🔴",
  orange: "🟠",
  yellow: "🟡",
  green: "🟢",
  blue: "🔵",
  purple: "🟣",
  black: "⚫",
  white: "⚪",
  silver: "⚪",
  gray: "⚫",
  cyan: "🔵",
  pink: "🩷",
};

function formatTime(ts: number): string {
  if (!ts) return "-";
  return new Date(ts * 1000).toLocaleTimeString();
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const min = Math.floor(seconds / 60);
  const sec = Math.round(seconds % 60);
  return `${min}m${sec}s`;
}

export default function CrossCameraTracking() {
  const [tracks, setTracks] = useState<Record<string, GlobalTrack>>({});
  const [stats, setStats] = useState<Stats | null>(null);
  const [alertPaths, setAlertPaths] = useState<AlertPath[]>([]);
  const [mapData, setMapData] = useState<{ cameras: Record<string, any>; tracks: any[] }>({ cameras: {}, tracks: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"map" | "alerts" | "tracks">("map");
  const [selectedMapTrack, setSelectedMapTrack] = useState<string | null>(null);
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [tracksRes, statsRes, alertsRes, mapRes] = await Promise.all([
        axios.get("/api/cross_camera/tracks"),
        axios.get("/api/cross_camera/stats"),
        axios.get("/api/cross_camera/alerts"),
        axios.get("/api/cross_camera/map_data"),
      ]);
      if (tracksRes.data.success) setTracks(tracksRes.data.tracks);
      if (statsRes.data.success) setStats(statsRes.data);
      if (alertsRes.data.success) setAlertPaths(alertsRes.data.alerts);
      if (mapRes.data.success) setMapData({ cameras: mapRes.data.cameras, tracks: mapRes.data.tracks });
      setError(null);
    } catch {
      setError("Failed to load cross-camera data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const handleCleanup = async () => {
    try {
      const res = await axios.delete("/api/cross_camera/expired");
      if (res.data.success) fetchData();
    } catch {
      // ignore
    }
  };

  if (loading) {
    return (
      <div className="flex h-full w-full items-center justify-center">
        <p>Loading cross-camera tracking data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full w-full items-center justify-center">
        <p className="text-danger">{error}</p>
      </div>
    );
  }

  const trackList = Object.values(tracks).sort(
    (a, b) => b.updated_at - a.updated_at,
  );

  return (
    <div className="flex size-full flex-col gap-4 p-4">
      <div className="flex items-center justify-between">
        <Heading as="h2">Cross-Camera Tracking</Heading>
        <div className="flex gap-2">
          <Button
            variant={tab === "map" ? "default" : "outline"}
            size="sm"
            onClick={() => setTab("map")}
          >
            🗺️ Map
          </Button>
          <Button
            variant={tab === "alerts" ? "default" : "outline"}
            size="sm"
            onClick={() => setTab("alerts")}
          >
            🚨 Alert Paths
          </Button>
          <Button
            variant={tab === "tracks" ? "default" : "outline"}
            size="sm"
            onClick={() => setTab("tracks")}
          >
            📡 All Tracks
          </Button>
          <Button variant="outline" size="sm" onClick={handleCleanup}>
            Clean Expired
          </Button>
        </div>
      </div>

      {stats && (
        <div className="flex gap-4 text-sm text-muted-foreground">
          <span>Total: {stats.total_global_tracks}</span>
          <span>With plate: {stats.tracks_with_plate}</span>
          <span>No plate: {stats.tracks_without_plate}</span>
          <span>Cameras: {stats.cameras_involved.join(", ") || "none"}</span>
        </div>
      )}

      {tab === "map" ? (
        <div className="flex flex-col gap-3">
          <TrackMapView
            apiKey={window.__FRIGATE_GOOGLE_MAPS_KEY || ""}
            mapData={mapData}
            selectedTrackId={selectedMapTrack}
            alertGlobalIds={new Set(alertPaths.map((a) => a.global_id))}
            onCameraClick={(cam) => setSelectedCamera((prev) => prev === cam ? null : cam)}
          />
          {selectedCamera && (
            <CameraSnapshotPanel
              cameraName={selectedCamera}
              onClose={() => setSelectedCamera(null)}
            />
          )}
        </div>
      ) : tab === "alerts" ? (
        <AlertPathsView alertPaths={alertPaths} onShowOnMap={(gid) => { setSelectedMapTrack(gid); setTab("map"); }} />
      ) : (
        <TracksView trackList={trackList} onShowOnMap={(gid) => { setSelectedMapTrack(gid); setTab("map"); }} />
      )}
    </div>
  );
}

function AlertPathsView({ alertPaths, onShowOnMap }: { alertPaths: AlertPath[]; onShowOnMap: (gid: string) => void }) {
  if (alertPaths.length === 0) {
    return (
      <div className="flex h-40 items-center justify-center text-muted-foreground">
        No alert paths — waiting for alerts with cross-camera matches
      </div>
    );
  }

  return (
    <div className="grid gap-4">
      {alertPaths.map((ap) => {
        const duration = ap.last_seen - ap.first_seen;
        return (
          <div
            key={ap.global_id}
            className="rounded-lg border bg-card p-4 shadow-sm"
          >
            {/* Header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3 flex-wrap">
                <span className="rounded bg-red-500/10 px-2 py-0.5 text-sm font-semibold text-red-600">
                  🚨 ALERT
                </span>
                <span className="font-mono text-sm font-bold">
                  {ap.global_id}
                </span>
                {ap.label === "person" ? (
                  <>
                    <span className="rounded bg-blue-500/10 px-2 py-0.5 text-sm font-semibold text-blue-600">
                      🧑 person
                    </span>
                    {ap.face_name && (
                      <span className="rounded bg-green-500/10 px-2 py-0.5 text-sm font-semibold text-green-600">
                        👤 {ap.face_name}
                      </span>
                    )}
                    {ap.upper_color && ap.upper_color !== "unknown" && (
                      <span className="text-sm">
                        👕 {COLOR_DOTS[ap.upper_color] || "❓"} {ap.upper_color}
                      </span>
                    )}
                    {ap.lower_color && ap.lower_color !== "unknown" && (
                      <span className="text-sm">
                        👖 {COLOR_DOTS[ap.lower_color] || "❓"} {ap.lower_color}
                      </span>
                    )}
                  </>
                ) : (
                  <>
                    {ap.plate && (
                      <span className="rounded bg-primary/10 px-2 py-0.5 text-sm font-semibold text-primary">
                        🚗 {ap.plate}
                      </span>
                    )}
                    {ap.color && (
                      <span className="text-sm">
                        {COLOR_DOTS[ap.color] || "❓"} {ap.color}
                      </span>
                    )}
                  </>
                )}
              </div>
              <div className="flex items-center gap-3 text-xs text-muted-foreground">
                <span>📷 {ap.cameras_visited} cameras</span>
                <span>⏱ {formatDuration(duration)}</span>
              </div>
            </div>

            {/* Camera Path Timeline */}
            <div className="mt-3">
              <div className="text-xs font-semibold text-muted-foreground mb-2">
                Movement Path:
              </div>
              <div className="flex items-center gap-1 flex-wrap">
                {ap.camera_path.map((step, i) => (
                  <div key={`${step.camera}-${i}`} className="flex items-center gap-1">
                    {i > 0 && (
                      <span className="text-lg text-muted-foreground">→</span>
                    )}
                    <div
                      className={`rounded-lg px-3 py-2 text-xs ${
                        step.camera === ap.alert_camera
                          ? "bg-red-500/10 border border-red-500/30"
                          : "bg-secondary"
                      }`}
                    >
                      <div className="font-bold">{step.camera}</div>
                      <div className="text-muted-foreground">
                        {formatTime(step.enter_time)}
                        {step.exit_time !== step.enter_time && (
                          <> → {formatTime(step.exit_time)}</>
                        )}
                      </div>
                      {step.exit_time > step.enter_time && (
                        <div className="text-muted-foreground">
                          ({formatDuration(step.exit_time - step.enter_time)})
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Footer */}
            <div className="mt-2 flex items-center justify-between">
              <div className="flex gap-4 text-xs text-muted-foreground">
                <span>First seen: {formatTime(ap.first_seen)}</span>
                <span>Last seen: {formatTime(ap.last_seen)}</span>
                <span>Total sightings: {ap.total_sightings}</span>
              </div>
              {ap.cameras_visited >= 2 && (
                <button
                  className="rounded bg-blue-500/10 px-2 py-1 text-xs font-semibold text-blue-600 hover:bg-blue-500/20"
                  onClick={() => onShowOnMap(ap.global_id)}
                >
                  🗺️ Show on Map
                </button>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function TracksView({ trackList, onShowOnMap }: { trackList: GlobalTrack[]; onShowOnMap: (gid: string) => void }) {
  if (trackList.length === 0) {
    return (
      <div className="flex h-40 items-center justify-center text-muted-foreground">
        No cross-camera tracks active
      </div>
    );
  }

  return (
    <div className="grid gap-3">
      {trackList.map((track) => (
        <div
          key={track.global_id}
          className="rounded-lg border bg-card p-4 shadow-sm"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3 flex-wrap">
              <span className="font-mono text-sm font-bold">
                {track.global_id}
              </span>
              {track.label === "person" ? (
                <>
                  <span className="rounded bg-blue-500/10 px-2 py-0.5 text-sm font-semibold text-blue-600">
                    🧑 person
                  </span>
                  {track.face_name && (
                    <span className="rounded bg-green-500/10 px-2 py-0.5 text-sm font-semibold text-green-600">
                      👤 {track.face_name}
                    </span>
                  )}
                  {track.upper_color && track.upper_color !== "unknown" && (
                    <span className="text-sm">
                      👕 {COLOR_DOTS[track.upper_color] || "❓"} {track.upper_color}
                    </span>
                  )}
                  {track.lower_color && track.lower_color !== "unknown" && (
                    <span className="text-sm">
                      👖 {COLOR_DOTS[track.lower_color] || "❓"} {track.lower_color}
                    </span>
                  )}
                  {track.body_ratio > 0 && (
                    <span className="text-xs text-muted-foreground">
                      📏 ratio: {track.body_ratio}
                    </span>
                  )}
                </>
              ) : (
                <>
                  {track.plate && (
                    <span className="rounded bg-primary/10 px-2 py-0.5 text-sm font-semibold text-primary">
                      🚗 {track.plate}
                    </span>
                  )}
                  <span className="text-sm">
                    {COLOR_DOTS[track.color] || "❓"} {track.color}
                  </span>
                  <span className="text-sm text-muted-foreground">
                    {track.vehicle_type}
                  </span>
                </>
              )}
            </div>
            <span className="text-xs text-muted-foreground">
              {track.sightings.length} camera
              {track.sightings.length > 1 ? "s" : ""}
            </span>
          </div>

          <div className="mt-2 flex items-center justify-between">
            <div className="flex flex-wrap gap-2">
              {track.sightings.map((s, i) => (
                <div
                  key={`${s.camera}-${i}`}
                  className="flex items-center gap-1 rounded bg-secondary px-2 py-1 text-xs"
                >
                  <span className="font-semibold">{s.camera}</span>
                  <span className="text-muted-foreground">
                    {formatTime(s.first_seen)} → {formatTime(s.last_seen)}
                  </span>
                </div>
              ))}
            </div>
            <button
              className="rounded bg-blue-500/10 px-2 py-1 text-xs font-semibold text-blue-600 hover:bg-blue-500/20 flex-shrink-0"
              onClick={() => onShowOnMap(track.global_id)}
            >
              🗺️ Show on Map
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

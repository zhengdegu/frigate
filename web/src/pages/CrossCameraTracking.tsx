import { useEffect, useState, useCallback } from "react";
import Heading from "@/components/ui/heading";
import { Button } from "@/components/ui/button";
import axios from "axios";

interface Sighting {
  camera: string;
  local_track_id: string;
  first_seen: number;
  last_seen: number;
  box: number[];
}

interface GlobalTrack {
  global_id: string;
  plate: string | null;
  color: string;
  vehicle_type: string;
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
};

function formatTime(ts: number): string {
  if (!ts) return "-";
  return new Date(ts * 1000).toLocaleTimeString();
}

export default function CrossCameraTracking() {
  const [tracks, setTracks] = useState<Record<string, GlobalTrack>>({});
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [tracksRes, statsRes] = await Promise.all([
        axios.get("/api/cross_camera/tracks"),
        axios.get("/api/cross_camera/stats"),
      ]);
      if (tracksRes.data.success) {
        setTracks(tracksRes.data.tracks);
      }
      if (statsRes.data.success) {
        setStats(statsRes.data);
      }
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
      if (res.data.success) {
        fetchData();
      }
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
        <Button variant="outline" size="sm" onClick={handleCleanup}>
          Clean Expired
        </Button>
      </div>

      {stats && (
        <div className="flex gap-4 text-sm text-muted-foreground">
          <span>Total: {stats.total_global_tracks}</span>
          <span>With plate: {stats.tracks_with_plate}</span>
          <span>No plate: {stats.tracks_without_plate}</span>
          <span>Cameras: {stats.cameras_involved.join(", ") || "none"}</span>
        </div>
      )}

      {trackList.length === 0 ? (
        <div className="flex h-40 items-center justify-center text-muted-foreground">
          No cross-camera tracks active
        </div>
      ) : (
        <div className="grid gap-3">
          {trackList.map((track) => (
            <div
              key={track.global_id}
              className="rounded-lg border bg-card p-4 shadow-sm"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="font-mono text-sm font-bold">
                    {track.global_id}
                  </span>
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
                </div>
                <span className="text-xs text-muted-foreground">
                  {track.sightings.length} camera
                  {track.sightings.length > 1 ? "s" : ""}
                </span>
              </div>

              <div className="mt-2 flex flex-wrap gap-2">
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
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

import { useEffect, useState } from "react";
import axios from "axios";

interface EventSnapshot {
  id: string;
  label: string;
  start_time: number;
  end_time: number | null;
  camera: string;
}

function formatTime(ts: number): string {
  if (!ts) return "-";
  return new Date(ts * 1000).toLocaleTimeString();
}

function formatDate(ts: number): string {
  if (!ts) return "-";
  return new Date(ts * 1000).toLocaleDateString();
}

export default function CameraSnapshotPanel({
  cameraName,
  onClose,
}: {
  cameraName: string;
  onClose: () => void;
}) {
  const [events, setEvents] = useState<EventSnapshot[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    axios
      .get<EventSnapshot[]>(`/api/events`, {
        params: { camera: cameraName, has_snapshot: 1 },
      })
      .then((res) => {
        if (!cancelled) {
          setEvents(Array.isArray(res.data) ? res.data : []);
        }
      })
      .catch(() => {
        if (!cancelled) setError("Failed to load snapshots");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [cameraName]);

  return (
    <div className="rounded-lg border bg-card p-4 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-lg">📸</span>
          <span className="font-semibold text-sm">
            Camera Snapshots — {cameraName}
          </span>
          <span className="text-xs text-muted-foreground">
            ({events.length} events)
          </span>
        </div>
        <button
          className="rounded bg-secondary px-2 py-1 text-xs hover:bg-secondary/80"
          onClick={onClose}
        >
          ✕ Close
        </button>
      </div>

      {loading && (
        <div className="flex h-24 items-center justify-center text-sm text-muted-foreground">
          Loading snapshots…
        </div>
      )}

      {error && (
        <div className="flex h-24 items-center justify-center text-sm text-danger">
          {error}
        </div>
      )}

      {!loading && !error && events.length === 0 && (
        <div className="flex h-24 items-center justify-center text-sm text-muted-foreground">
          No snapshot events for this camera
        </div>
      )}

      {!loading && !error && events.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 max-h-[360px] overflow-y-auto">
          {events.map((evt) => (
            <div
              key={evt.id}
              className="rounded-lg border bg-secondary/50 overflow-hidden"
            >
              <img
                src={`/api/events/${evt.id}/snapshot.jpg`}
                alt={`${evt.label} at ${cameraName}`}
                className="w-full h-24 object-cover"
                loading="lazy"
              />
              <div className="p-1.5 text-xs space-y-0.5">
                <div className="flex items-center justify-between">
                  <span className="font-semibold">
                    {evt.label === "person" ? "🧑" : "🚗"} {evt.label}
                  </span>
                </div>
                <div className="text-muted-foreground">
                  {formatDate(evt.start_time)} {formatTime(evt.start_time)}
                </div>
                {evt.end_time && evt.end_time !== evt.start_time && (
                  <div className="text-muted-foreground">
                    → {formatTime(evt.end_time)}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

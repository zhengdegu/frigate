import { useEffect, useState, useRef } from "react";
import {
  APIProvider,
  Map,
  AdvancedMarker,
  useMap,
  useMapsLibrary,
} from "@vis.gl/react-google-maps";

interface CameraLocation {
  name: string;
  latitude: number;
  longitude: number;
}

interface Waypoint {
  camera: string;
  latitude: number;
  longitude: number;
  enter_time: number;
  exit_time: number;
}

interface TrackPath {
  global_id: string;
  label: string;
  upper_color: string | null;
  lower_color: string | null;
  face_name: string | null;
  plate: string | null;
  color: string | null;
  vehicle_type: string | null;
  waypoints: Waypoint[];
  cameras_visited: number;
  first_seen: number;
  last_seen: number;
}

interface MapData {
  cameras: Record<string, CameraLocation>;
  tracks: TrackPath[];
}

const TRACK_COLORS = [
  "#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6",
  "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
];

function formatTime(ts: number): string {
  if (!ts) return "-";
  return new Date(ts * 1000).toLocaleTimeString();
}

interface TrackMapViewProps {
  apiKey: string;
  mapData: MapData;
}

function RouteRenderer({
  track,
  colorIndex,
}: {
  track: TrackPath;
  colorIndex: number;
}) {
  const map = useMap();
  const routesLib = useMapsLibrary("routes");
  const renderedRef = useRef(false);
  const renderersRef = useRef<google.maps.DirectionsRenderer[]>([]);

  useEffect(() => {
    if (!map || !routesLib || renderedRef.current) return;
    if (track.waypoints.length < 2) return;

    renderedRef.current = true;
    const color = TRACK_COLORS[colorIndex % TRACK_COLORS.length];
    const directionsService = new routesLib.DirectionsService();

    // Build segments between consecutive different cameras
    const segments: { origin: Waypoint; destination: Waypoint }[] = [];
    for (let i = 0; i < track.waypoints.length - 1; i++) {
      const from = track.waypoints[i];
      const to = track.waypoints[i + 1];
      if (from.camera !== to.camera) {
        // Avoid duplicate segments
        const exists = segments.some(
          (s) =>
            s.origin.camera === from.camera &&
            s.destination.camera === to.camera,
        );
        if (!exists) {
          segments.push({ origin: from, destination: to });
        }
      }
    }

    segments.forEach((seg) => {
      const travelMode =
        track.label === "car"
          ? google.maps.TravelMode.DRIVING
          : google.maps.TravelMode.WALKING;

      directionsService.route(
        {
          origin: { lat: seg.origin.latitude, lng: seg.origin.longitude },
          destination: {
            lat: seg.destination.latitude,
            lng: seg.destination.longitude,
          },
          travelMode,
        },
        (result, status) => {
          if (status === google.maps.DirectionsStatus.OK && result) {
            const renderer = new routesLib.DirectionsRenderer({
              map,
              directions: result,
              suppressMarkers: true,
              polylineOptions: {
             strokeColor: color,
                strokeWeight: 4,
                strokeOpacity: 0.8,
              },
            });
            renderersRef.current.push(renderer);
          }
        },
      );
    });

    return () => {
      renderersRef.current.forEach((r) => r.setMap(null));
      renderersRef.current = [];
      renderedRef.current = false;
    };
  }, [map, routesLib, track, colorIndex]);

  return null;
}

export default function TrackMapView({ apiKey, mapData }: TrackMapViewProps) {
  const [selectedTrack, setSelectedTrack] = useState<string | null>(null);
  const cameras = Object.values(mapData.cameras);

  if (cameras.length === 0) {
    return (
      <div className="flex h-96 items-center justify-center text-muted-foreground">
        <div className="text-center">
          <p className="text-lg">📍 No camera coordinates configured</p>
          <p className="mt-2 text-sm">
            Add latitude/longitude to your camera config:
          </p>
          <pre className="mt-2 rounded bg-secondary p-3 text-left text-xs">
{`cameras:
  cam0:
    latitude: 39.9042
    longitude: 116.4074`}
          </pre>
        </div>
      </div>
    );
  }

  // Calculate map center
  const centerLat =
    cameras.reduce((sum, c) => sum + c.latitude, 0) / cameras.length;
  const centerLng =
    cameras.reduce((sum, c) => sum + c.longitude, 0) / cameras.length;

  const visibleTracks = selectedTrack
    ? mapData.tracks.filter((t) => t.global_id === selectedTrack)
    : mapData.tracks.filter((t) => t.cameras_visited >= 2);

  return (
    <div className="flex gap-4 h-[600px]">
      {/* Map */}
      <div className="flex-1 rounded-lg overflow-hidden border">
        <APIProvider apiKey={apiKey}>
          <Map
            defaultCenter={{ lat: centerLat, lng: centerLng }}
            defaultZoom={16}
            mapId="cross-camera-map"
            gestureHandling="greedy"
            style={{ width: "100%", height: "100%" }}
          >
            {/* Camera markers */}
            {cameras.map((cam) => (
              <AdvancedMarker
                key={cam.name}
                position={{ lat: cam.latitude, lng: cam.longitude }}
                title={cam.name}
              >
                <div className="flex flex-col items-center">
                  <div className="rounded-full bg-blue-600 px-2 py-1 text-xs font-bold text-white shadow-lg">
                    📷 {cam.name}
                  </div>
                </div>
              </AdvancedMarker>
            ))}

            {/* Route renderers */}
            {visibleTracks.map((track, i) => (
              <RouteRenderer key={track.global_id} track={track} colorIndex={i} />
            ))}
          </Map>
        </APIProvider>
      </div>

      {/* Track list sidebar */}
      <div className="w-80 overflow-y-auto rounded-lg border bg-card p-3">
        <div className="text-sm font-semibold mb-3">
          Tracked Paths ({mapData.tracks.length})
        </div>

        <button
          className={`w-full text-left rounded px-2 py-1 mb-1 text-xs ${
            selectedTrack === null
              ? "bg-primary/10 text-primary font-semibold"
              : "hover:bg-secondary"
          }`}
          onClick={() => setSelectedTrack(null)}
        >
          Show all multi-camera tracks
        </button>

        <div className="mt-2 flex flex-col gap-1">
          {mapData.tracks.map((track, i) => {
            const color = TRACK_COLORS[i % TRACK_COLORS.length];
            const isSelected = selectedTrack === track.global_id;
            const path = track.waypoints
              .filter(
                (w, j) =>
                  j === 0 || w.camera !== track.waypoints[j - 1].camera,
              )
              .map((w) => w.camera)
              .join(" → ");

            return (
              <button
                key={track.global_id}
       className={`w-full text-left rounded px-2 py-2 text-xs ${
                  isSelected
                    ? "bg-primary/10 border border-primary/30"
                    : "hover:bg-secondary"
                }`}
                onClick={() =>
                  setSelectedTrack(isSelected ? null : track.global_id)
                }
              >
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full flex-shrink-0"
                    style={{ backgroundColor: color }}
                  />
                  <span className="font-mono font-bold">
                    {track.global_id}
                  </span>
                  <span className="text-muted-foreground">
                    {track.label === "person" ? "🧑" : "🚗"}
                  </span>
                </div>
                <div className="mt-1 text-muted-foreground">{path}</div>
                <div className="mt-1 text-muted-foreground">
                  {formatTime(track.first_seen)} → {formatTime(track.last_seen)}
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

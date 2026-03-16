import { useEffect, useState, useRef, useCallback } from "react";
import {
  APIProvider,
  Map,
  useMap,
  useMapsLibrary,
} from "@vis.gl/react-google-maps";

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
  cameras: Record<string, { name: string; latitude: number; longitude: number }>;
  tracks: TrackPath[];
}

// A segment of road points between two cameras
interface RouteSegment {
  fromCamera: string;
  toCamera: string;
  departTime: number; // exit_time of source camera
  arriveTime: number; // enter_time of destination camera
  points: google.maps.LatLng[]; // road points from Directions API
}

const TRACK_COLORS = [
  "#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6",
  "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
];

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

interface TrackMapViewProps {
  apiKey: string;
  mapData: MapData;
  selectedTrackId?: string | null;
  alertGlobalIds?: Set<string>;
  onCameraClick?: (cameraName: string) => void;
}

// Get position along road at a given time
function getPositionAtTime(
  waypoints: Waypoint[],
  routeSegments: RouteSegment[],
  currentTime: number,
): { lat: number; lng: number; camera: string } | null {
  if (waypoints.length === 0) return null;
  if (currentTime < waypoints[0].enter_time) return null;

  const last = waypoints[waypoints.length - 1];
  if (currentTime > last.exit_time) {
    return { lat: last.latitude, lng: last.longitude, camera: last.camera };
  }

  for (let i = 0; i < waypoints.length; i++) {
    const wp = waypoints[i];
    // At this camera
    if (currentTime >= wp.enter_time && currentTime <= wp.exit_time) {
      return { lat: wp.latitude, lng: wp.longitude, camera: wp.camera };
    }
    // Between two cameras
    if (i < waypoints.length - 1) {
      const next = waypoints[i + 1];
      if (currentTime > wp.exit_time && currentTime < next.enter_time) {
        // Find matching route segment
        const seg = routeSegments.find(
          (s) => s.fromCamera === wp.camera && s.toCamera === next.camera,
        );
        if (seg && seg.points.length > 1) {
          // Interpolate along road points
          const progress = (currentTime - wp.exit_time) / (next.enter_time - wp.exit_time);
          const totalPoints = seg.points.length;
          const exactIndex = progress * (totalPoints - 1);
          const idx = Math.floor(exactIndex);
          const frac = exactIndex - idx;

          if (idx >= totalPoints - 1) {
            const p = seg.points[totalPoints - 1];
            return { lat: p.lat(), lng: p.lng(), camera: `${wp.camera} → ${next.camera}` };
          }

          const p1 = seg.points[idx];
          const p2 = seg.points[idx + 1];
          return {
            lat: p1.lat() + (p2.lat() - p1.lat()) * frac,
            lng: p1.lng() + (p2.lng() - p1.lng()) * frac,
            camera: `${wp.camera} → ${next.camera}`,
          };
        }
        // Fallback: straight line
        const progress = (currentTime - wp.exit_time) / (next.enter_time - wp.exit_time);
        return {
          lat: wp.latitude + (next.latitude - wp.latitude) * progress,
          lng: wp.longitude + (next.longitude - wp.longitude) * progress,
          camera: `${wp.camera} → ${next.camera}`,
        };
      }
    }
  }
  return null;
}

// Camera + moving markers using native Google Maps API
function NativeMarkers({
  cameras,
  activeCameras,
  movingPos,
  movingLabel,
  movingColor,
  onCameraClick,
}: {
  cameras: { name: string; latitude: number; longitude: number }[];
  activeCameras: Set<string>;
  movingPos: { lat: number; lng: number } | null;
  movingLabel: string;
  movingColor: string;
  onCameraClick?: (cameraName: string) => void;
}) {
  const map = useMap();
  const markersRef = useRef<google.maps.Marker[]>([]);
  const movingRef = useRef<google.maps.Marker | null>(null);

  useEffect(() => {
    if (!map) return;
    markersRef.current.forEach((m) => m.setMap(null));
    markersRef.current = [];

    cameras.forEach((cam) => {
      const isActive = activeCameras.has(cam.name);
      const marker = new google.maps.Marker({
        map,
        position: { lat: cam.latitude, lng: cam.longitude },
        title: cam.name,
        label: {
          text: cam.name,
          fontSize: "11px",
          fontWeight: "bold",
          color: "white",
        },
        icon: {
          path: google.maps.SymbolPath.CIRCLE,
          scale: 18,
          fillColor: isActive ? "#dc2626" : "#2563eb",
          fillOpacity: 1,
          strokeColor: "white",
          strokeWeight: 2,
        },
      });
      if (onCameraClick) {
        marker.addListener("click", () => onCameraClick(cam.name));
        marker.setCursor("pointer");
      }
      markersRef.current.push(marker);
    });

    return () => {
      markersRef.current.forEach((m) => {
        google.maps.event.clearInstanceListeners(m);
        m.setMap(null);
      });
      markersRef.current = [];
    };
  }, [map, cameras, activeCameras, onCameraClick]);

  useEffect(() => {
    if (!map) return;
    if (!movingPos) {
      if (movingRef.current) {
        movingRef.current.setMap(null);
        movingRef.current = null;
      }
      return;
    }
    if (!movingRef.current) {
      movingRef.current = new google.maps.Marker({
        map,
        position: movingPos,
        label: { text: movingLabel, fontSize: "14px", fontWeight: "bold", color: "white" },
        icon: {
          path: google.maps.SymbolPath.CIRCLE,
          scale: 14,
          fillColor: movingColor,
          fillOpacity: 1,
          strokeColor: "white",
          strokeWeight: 3,
        },
        zIndex: 999,
      });
    } else {
      movingRef.current.setPosition(movingPos);
    }
  }, [map, movingPos, movingLabel, movingColor]);

  useEffect(() => {
    return () => {
      if (movingRef.current) {
        movingRef.current.setMap(null);
        movingRef.current = null;
      }
    };
  }, []);

  return null;
}

// Route renderer that also exports road points for playback
function RouteRenderer({
  track,
  colorIndex,
  onSegmentsReady,
}: {
  track: TrackPath;
  colorIndex: number;
  onSegmentsReady?: (segments: RouteSegment[]) => void;
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

    // Build unique segments
    const segmentDefs: { from: Waypoint; to: Waypoint }[] = [];
    for (let i = 0; i < track.waypoints.length - 1; i++) {
      const from = track.waypoints[i];
      const to = track.waypoints[i + 1];
      if (from.camera !== to.camera) {
        const exists = segmentDefs.some(
          (s) => s.from.camera === from.camera && s.to.camera === to.camera,
        );
        if (!exists) segmentDefs.push({ from, to });
      }
    }

    const collectedSegments: RouteSegment[] = [];
    let pending = segmentDefs.length;

    segmentDefs.forEach((seg) => {
      const travelMode =
        track.label === "car"
          ? google.maps.TravelMode.DRIVING
          : google.maps.TravelMode.WALKING;

      directionsService.route(
        {
          origin: { lat: seg.from.latitude, lng: seg.from.longitude },
          destination: { lat: seg.to.latitude, lng: seg.to.longitude },
          travelMode,
        },
        (result, status) => {
          if (status === google.maps.DirectionsStatus.OK && result) {
            const renderer = new routesLib.DirectionsRenderer({
              map,
              directions: result,
              suppressMarkers: true,
              polylineOptions: { strokeColor: color, strokeWeight: 4, strokeOpacity: 0.8 },
            });
            renderersRef.current.push(renderer);

            // Extract road points for playback
            const route = result.routes[0];
            if (route) {
              const points: google.maps.LatLng[] = [];
              route.legs.forEach((leg) => {
                leg.steps.forEach((step) => {
                  points.push(...step.path);
                });
              });
              collectedSegments.push({
                fromCamera: seg.from.camera,
                toCamera: seg.to.camera,
                departTime: seg.from.exit_time,
                arriveTime: seg.to.enter_time,
                points,
              });
            }
          }

          pending--;
          if (pending <= 0 && onSegmentsReady) {
            onSegmentsReady(collectedSegments);
          }
        },
      );
    });

    if (segmentDefs.length === 0 && onSegmentsReady) {
      onSegmentsReady([]);
    }

    return () => {
      renderersRef.current.forEach((r) => r.setMap(null));
      renderersRef.current = [];
      renderedRef.current = false;
    };
  }, [map, routesLib, track, colorIndex, onSegmentsReady]);

  return null;
}

export default function TrackMapView({ apiKey, mapData, selectedTrackId, alertGlobalIds, onCameraClick }: TrackMapViewProps) {
  const [selectedTrack, setSelectedTrack] = useState<string | null>(selectedTrackId || null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(5);
  const [routeSegments, setRouteSegments] = useState<RouteSegment[]>([]);
  const animFrameRef = useRef<number>(0);
  const lastFrameRef = useRef<number>(0);
  const cameras = Object.values(mapData.cameras);

  useEffect(() => {
    if (selectedTrackId) setSelectedTrack(selectedTrackId);
  }, [selectedTrackId]);

  const activeTrack = selectedTrack
    ? mapData.tracks.find((t) => t.global_id === selectedTrack) || null
    : null;

  let trackStart = 0;
  let trackEnd = 0;
  if (activeTrack) {
    trackStart = activeTrack.waypoints[0]?.enter_time || 0;
    trackEnd = activeTrack.waypoints[activeTrack.waypoints.length - 1]?.exit_time || 0;
  }
  const totalDuration = trackEnd - trackStart;

  useEffect(() => {
    if (activeTrack) {
      const start = activeTrack.waypoints[0]?.enter_time || 0;
      setPlaybackTime(start);
      setIsPlaying(false);
      setRouteSegments([]);

      // Fetch road points for this track
      if (activeTrack.waypoints.length >= 2 && typeof google !== "undefined") {
        const ds = new google.maps.DirectionsService();
        const segs: { from: Waypoint; to: Waypoint }[] = [];
        for (let i = 0; i < activeTrack.waypoints.length - 1; i++) {
          const from = activeTrack.waypoints[i];
          const to = activeTrack.waypoints[i + 1];
          if (from.camera !== to.camera) {
            const exists = segs.some(
              (s) => s.from.camera === from.camera && s.to.camera === to.camera,
            );
            if (!exists) segs.push({ from, to });
          }
        }

        const collected: RouteSegment[] = [];
        let pending = segs.length;
        if (pending === 0) return;

        segs.forEach((seg) => {
          const travelMode =
            activeTrack.label === "car"
              ? google.maps.TravelMode.DRIVING
              : google.maps.TravelMode.WALKING;

          ds.route(
            {
              origin: { lat: seg.from.latitude, lng: seg.from.longitude },
              destination: { lat: seg.to.latitude, lng: seg.to.longitude },
              travelMode,
            },
            (result, status) => {
              if (status === google.maps.DirectionsStatus.OK && result) {
                const points: google.maps.LatLng[] = [];
                result.routes[0]?.legs.forEach((leg) => {
                  leg.steps.forEach((step) => {
                    points.push(...step.path);
                  });
                });
                collected.push({
                  fromCamera: seg.from.camera,
                  toCamera: seg.to.camera,
                  departTime: seg.from.exit_time,
                  arriveTime: seg.to.enter_time,
                  points,
                });
              }
              pending--;
              if (pending <= 0) {
                setRouteSegments([...collected]);
              }
            },
          );
        });
      }
    }
  }, [selectedTrack]);

  const animate = useCallback(
    (timestamp: number) => {
      if (!lastFrameRef.current) lastFrameRef.current = timestamp;
      const delta = (timestamp - lastFrameRef.current) / 1000;
      lastFrameRef.current = timestamp;

      setPlaybackTime((prev) => {
        const next = prev + delta * playbackSpeed;
        if (next >= trackEnd) {
          setIsPlaying(false);
          return trackEnd;
        }
        return next;
      });

      animFrameRef.current = requestAnimationFrame(animate);
    },
    [playbackSpeed, trackEnd],
  );

  useEffect(() => {
    if (isPlaying) {
      lastFrameRef.current = 0;
      animFrameRef.current = requestAnimationFrame(animate);
    } else {
      cancelAnimationFrame(animFrameRef.current);
    }
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [isPlaying, animate]);

  if (cameras.length === 0) {
    return (
      <div className="flex h-96 items-center justify-center text-muted-foreground">
        <p className="text-lg">📍 No camera coordinates configured</p>
      </div>
    );
  }

  const centerLat = cameras.reduce((sum, c) => sum + c.latitude, 0) / cameras.length;
  const centerLng = cameras.reduce((sum, c) => sum + c.longitude, 0) / cameras.length;

  const movingPos = activeTrack
    ? getPositionAtTime(activeTrack.waypoints, routeSegments, playbackTime)
    : null;
  const activeCameras = new Set<string>();
  if (movingPos) activeCameras.add(movingPos.camera);

  const visibleTracks = activeTrack
    ? [activeTrack]
    : mapData.tracks.filter((t) => t.cameras_visited >= 2);

  const trackColor = activeTrack
    ? TRACK_COLORS[mapData.tracks.findIndex((t) => t.global_id === activeTrack.global_id) % TRACK_COLORS.length]
    : "#3b82f6";

  return (
    <div className="flex flex-col gap-3 h-[700px]">
      {activeTrack && totalDuration > 0 && (
        <div className="rounded-lg border bg-card p-3 flex flex-col gap-2">
          <div className="flex items-center gap-3">
            <button
              className="rounded bg-primary px-3 py-1 text-sm font-semibold text-primary-foreground"
              onClick={() => {
                if (playbackTime >= trackEnd) setPlaybackTime(trackStart);
                setIsPlaying(!isPlaying);
              }}
            >
              {isPlaying ? "⏸ Pause" : "▶ Play"}
            </button>
            <button
              className="rounded bg-secondary px-2 py-1 text-xs"
              onClick={() => { setPlaybackTime(trackStart); setIsPlaying(false); }}
            >
              ⏮ Reset
            </button>
            <div className="flex items-center gap-1 text-xs">
              <span>Speed:</span>
              {[1, 2, 5, 10, 20].map((s) => (
                <button
                  key={s}
                  className={`rounded px-2 py-0.5 ${playbackSpeed === s ? "bg-primary text-primary-foreground" : "bg-secondary"}`}
                  onClick={() => setPlaybackSpeed(s)}
                >
                  {s}x
                </button>
              ))}
            </div>
            <div className="ml-auto flex items-center gap-2">
              <span className="text-xs">
                {activeTrack.label === "person" ? "🧑" : "🚗"} {activeTrack.global_id}
              </span>
              <span className="text-sm font-mono">{formatTime(playbackTime)}</span>
              <span className="text-xs text-muted-foreground">
                ({formatDuration(playbackTime - trackStart)} / {formatDuration(totalDuration)})
              </span>
            </div>
          </div>
          <input
            type="range"
            min={trackStart}
            max={trackEnd}
            step={0.1}
            value={playbackTime}
            onChange={(e) => setPlaybackTime(parseFloat(e.target.value))}
            className="w-full"
          />
          <div className="flex gap-2 text-xs flex-wrap">
            {activeTrack.waypoints
              .filter((w, j) => j === 0 || w.camera !== activeTrack.waypoints[j - 1].camera)
              .map((w, j) => {
                const isActive = playbackTime >= w.enter_time && playbackTime <= w.exit_time;
                return (
                  <span
                    key={j}
                    className={`rounded px-2 py-0.5 border ${isActive ? "bg-red-500/20 border-red-500 text-red-600 font-semibold" : "bg-secondary border-transparent opacity-50"}`}
                  >
                    📷 {w.camera} ({formatTime(w.enter_time)})
                  </span>
                );
              })}
          </div>
        </div>
      )}

      <div className="flex gap-4 flex-1 min-h-0">
        <div className="flex-1 rounded-lg overflow-hidden border">
          <APIProvider apiKey={apiKey}>
            <Map
              defaultCenter={{ lat: centerLat, lng: centerLng }}
              defaultZoom={16}
              gestureHandling="greedy"
              style={{ width: "100%", height: "100%" }}
            >
              <NativeMarkers
                cameras={cameras}
                activeCameras={activeCameras}
                movingPos={movingPos ? { lat: movingPos.lat, lng: movingPos.lng } : null}
                movingLabel={activeTrack ? (activeTrack.label === "person" ? "🧑" : "🚗") : ""}
                movingColor={trackColor}
                onCameraClick={onCameraClick}
              />
              {visibleTracks.map((track, i) => (
                <RouteRenderer
                  key={track.global_id}
                  track={track}
                  colorIndex={i}
                  onSegmentsReady={track.global_id === selectedTrack ? setRouteSegments : undefined}
                />
              ))}
            </Map>
          </APIProvider>
        </div>

        <div className="w-80 overflow-y-auto rounded-lg border bg-card p-3">
          <div className="text-sm font-semibold mb-3">
            Tracked Paths ({mapData.tracks.length})
          </div>
          <div className="text-xs text-muted-foreground mb-2">
            Click a track to replay its path on the map
          </div>

          {selectedTrack && (
            <button
              className="w-full text-left rounded px-2 py-1 mb-2 text-xs bg-secondary hover:bg-secondary/80"
              onClick={() => { setSelectedTrack(null); setIsPlaying(false); }}
            >
              ← Show all tracks
            </button>
          )}

          <div className="flex flex-col gap-1">
            {mapData.tracks.map((track, i) => {
              const color = TRACK_COLORS[i % TRACK_COLORS.length];
              const isSelected = selectedTrack === track.global_id;
              const path = track.waypoints
                .filter((w, j) => j === 0 || w.camera !== track.waypoints[j - 1].camera)
                .map((w) => w.camera)
                .join(" → ");

              return (
                <button
                  key={track.global_id}
                  className={`w-full text-left rounded px-2 py-2 text-xs ${isSelected ? "bg-primary/10 border border-primary/30" : "hover:bg-secondary"}`}
                  onClick={() => setSelectedTrack(isSelected ? null : track.global_id)}
                >
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
                    <span className="font-mono font-bold">{track.global_id}</span>
                    <span className="text-muted-foreground">{track.label === "person" ? "🧑" : "🚗"}</span>
                    {alertGlobalIds?.has(track.global_id) && (
                      <span className="rounded bg-red-500/10 px-1 py-0.5 text-[10px] font-semibold text-red-600">🚨 ALERT</span>
                    )}
                  </div>
                  <div className="mt-1 text-muted-foreground">{path || track.waypoints[0]?.camera}</div>
                  <div className="mt-1 text-muted-foreground">
                    {formatTime(track.first_seen)} → {formatTime(track.last_seen)}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

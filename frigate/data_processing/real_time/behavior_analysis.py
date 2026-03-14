"""Real time processor for human behavior analysis using pose estimation.

Detects abnormal behaviors (fall, fight, etc.) by analyzing body keypoints
from a MoveNet Lightning TFLite model over a sliding temporal window.
"""

import datetime
import logging
import os
from collections import deque
from typing import Any, NamedTuple

import cv2
import numpy as np

from frigate.comms.embeddings_updater import EmbeddingsRequestEnum
from frigate.comms.event_metadata_updater import (
    EventMetadataPublisher,
    EventMetadataTypeEnum,
)
from frigate.comms.inter_process import InterProcessRequestor
from frigate.config import FrigateConfig
from frigate.config.classification import BehaviorDetectionModelConfig, BehaviorType
from frigate.const import MODEL_CACHE_DIR
from frigate.log import suppress_stderr_during
from frigate.util.builtin import EventsPerSecond, InferenceSpeed

from ..types import DataProcessorMetrics
from .api import RealTimeProcessorApi

try:
    from tflite_runtime.interpreter import Interpreter
except ModuleNotFoundError:
    from ai_edge_litert.interpreter import Interpreter

logger = logging.getLogger(__name__)

# MoveNet Lightning keypoint indices
KP_NOSE = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

# Arm keypoints for fight motion analysis
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10

# Pose model input size
POSE_INPUT_SIZE = 192

# Fall detection thresholds
FALL_ANGLE_UPRIGHT_MAX = 30.0  # degrees — torso considered upright
FALL_ANGLE_FALLEN_MIN = 60.0  # degrees — torso considered fallen
FALL_HIP_DROP_RATE_THRESHOLD = 0.02  # normalized y-coord drop per frame
FALL_TRANSITION_FRAMES = 10  # look-back window for angle transition

# Fight detection thresholds
FIGHT_IOU_THRESHOLD = 0.3
FIGHT_ARM_MOTION_STD_THRESHOLD = 0.015  # normalized displacement std


class FrameRecord(NamedTuple):
    """Single frame observation for a tracked person."""

    keypoints: np.ndarray  # shape (17, 3) — y, x, confidence
    timestamp: float
    bbox: tuple[int, int, int, int]  # y1, x1, y2, x2


class BehaviorAnalysisProcessor(RealTimeProcessorApi):
    """Detects abnormal human behaviors via pose estimation and rule-based analysis.

    Uses MoveNet Lightning to extract body keypoints, then applies temporal
    rule engines (fall detection, fight detection) over a sliding window
    of per-person observations.
    """

    def __init__(
        self,
        config: FrigateConfig,
        model_config: BehaviorDetectionModelConfig,
        sub_label_publisher: EventMetadataPublisher,
        requestor: InterProcessRequestor,
        metrics: DataProcessorMetrics,
    ) -> None:
        super().__init__(config, metrics)
        self.model_config = model_config
        self.sub_label_publisher = sub_label_publisher
        self.requestor = requestor

        # TFLite interpreter state
        self.interpreter: Interpreter | None = None
        self.tensor_input_details: dict[str, Any] | None = None
        self.tensor_output_details: dict[str, Any] | None = None

        # Sliding windows keyed by "{camera}:{object_id}"
        self.windows: dict[str, deque[FrameRecord]] = {}

        # Cooldown tracking: key → last trigger timestamp per behavior type
        self.cooldowns: dict[str, dict[str, float]] = {}

        # Metrics
        self.detections_per_second = EventsPerSecond()
        if (
            self.metrics
            and self.model_config.name in self.metrics.classification_speeds
        ):
            self.inference_speed = InferenceSpeed(
                self.metrics.classification_speeds[self.model_config.name]
            )
        else:
            self.inference_speed = None

        self.detections_per_second.start()
        self.__build_detector()

    def __build_detector(self) -> None:
        """Load the MoveNet Lightning TFLite model."""
        model_dir = os.path.join(MODEL_CACHE_DIR, self.model_config.pose_model)
        model_path = os.path.join(model_dir, "model.tflite")

        if not os.path.exists(model_path):
            logger.warning(
                "Behavior analysis model not found at %s — processor disabled",
                model_path,
            )
            self.interpreter = None
            self.tensor_input_details = None
            self.tensor_output_details = None
            return

        try:
            with suppress_stderr_during("tflite_interpreter_init"):
                self.interpreter = Interpreter(
                    model_path=model_path,
                    num_threads=2,
                )
                self.interpreter.allocate_tensors()
            self.tensor_input_details = self.interpreter.get_input_details()
            self.tensor_output_details = self.interpreter.get_output_details()
            logger.info(
                "Loaded pose estimation model '%s' for behavior analysis",
                self.model_config.pose_model,
            )
        except Exception:
            logger.exception(
                "Failed to load pose estimation model '%s'",
                self.model_config.pose_model,
            )
            self.interpreter = None
            self.tensor_input_details = None
            self.tensor_output_details = None

    # ------------------------------------------------------------------
    # Pose estimation
    # ------------------------------------------------------------------

    def _run_pose_estimation(self, crop: np.ndarray) -> np.ndarray | None:
        """Run MoveNet Lightning on an RGB crop.

        Args:
            crop: RGB image of a person region.

        Returns:
            Array of shape (17, 3) with (y, x, confidence) per keypoint,
            or None if the model is unavailable.
        """
        if self.interpreter is None:
            return None

        try:
            resized = cv2.resize(crop, (POSE_INPUT_SIZE, POSE_INPUT_SIZE))
            input_data = np.expand_dims(resized.astype(np.uint8), axis=0)

            self.interpreter.set_tensor(
                self.tensor_input_details[0]["index"], input_data
            )
            self.interpreter.invoke()

            # MoveNet output: [1, 1, 17, 3]
            output = self.interpreter.get_tensor(
                self.tensor_output_details[0]["index"]
            )
            keypoints = output[0, 0]  # shape (17, 3)
            return keypoints.astype(np.float32)
        except Exception:
            logger.debug("Pose estimation inference failed", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _torso_angle(keypoints: np.ndarray) -> float | None:
        """Compute the angle between the torso vector and vertical.

        The torso vector goes from the shoulder midpoint to the hip midpoint.
        Returns angle in degrees (0 = upright, 90 = horizontal), or None
        if keypoint confidence is too low.
        """
        ls = keypoints[KP_LEFT_SHOULDER]
        rs = keypoints[KP_RIGHT_SHOULDER]
        lh = keypoints[KP_LEFT_HIP]
        rh = keypoints[KP_RIGHT_HIP]

        # Require minimum confidence on all four keypoints
        if min(ls[2], rs[2], lh[2], rh[2]) < 0.2:
            return None

        shoulder_mid = np.array([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2])
        hip_mid = np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2])

        # Torso vector (in image coords: y increases downward)
        torso_vec = hip_mid - shoulder_mid
        # Vertical reference (pointing down in image coords)
        vertical = np.array([1.0, 0.0])

        norm = np.linalg.norm(torso_vec)
        if norm < 1e-6:
            return None

        cos_angle = np.dot(torso_vec, vertical) / norm
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    @staticmethod
    def _hip_midpoint_y(keypoints: np.ndarray) -> float | None:
        """Return the y-coordinate of the hip midpoint, or None if low confidence."""
        lh = keypoints[KP_LEFT_HIP]
        rh = keypoints[KP_RIGHT_HIP]
        if min(lh[2], rh[2]) < 0.2:
            return None
        return float((lh[0] + rh[0]) / 2)

    @staticmethod
    def _bbox_iou(
        a: tuple[int, int, int, int], b: tuple[int, int, int, int]
    ) -> float:
        """Compute IoU between two bboxes (y1, x1, y2, x2)."""
        y1 = max(a[0], b[0])
        x1 = max(a[1], b[1])
        y2 = min(a[2], b[2])
        x2 = min(a[3], b[3])

        inter = max(0, y2 - y1) * max(0, x2 - x1)
        if inter == 0:
            return 0.0

        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    # ------------------------------------------------------------------
    # Rule engines
    # ------------------------------------------------------------------

    def _detect_fall(self, window: deque[FrameRecord]) -> tuple[bool, float]:
        """Detect a fall event from the sliding window.

        Looks for a rapid transition of torso angle from upright (<30°) to
        fallen (>60°) combined with a downward hip drop rate exceeding the
        threshold.

        Returns:
            (detected, confidence) tuple.
        """
        if len(window) < FALL_TRANSITION_FRAMES:
            return False, 0.0

        recent = list(window)[-FALL_TRANSITION_FRAMES:]

        angles: list[float] = []
        hip_ys: list[float] = []

        for record in recent:
            angle = self._torso_angle(record.keypoints)
            hip_y = self._hip_midpoint_y(record.keypoints)
            if angle is not None:
                angles.append(angle)
            if hip_y is not None:
                hip_ys.append(hip_y)

        if len(angles) < 3 or len(hip_ys) < 3:
            return False, 0.0

        # Check angle transition: early frames upright, later frames fallen
        early_angles = angles[: len(angles) // 2]
        late_angles = angles[len(angles) // 2 :]

        had_upright = any(a < FALL_ANGLE_UPRIGHT_MAX for a in early_angles)
        now_fallen = any(a > FALL_ANGLE_FALLEN_MIN for a in late_angles)

        if not (had_upright and now_fallen):
            return False, 0.0

        # Check hip drop rate (positive = downward in image coords)
        hip_deltas = [
            hip_ys[i + 1] - hip_ys[i] for i in range(len(hip_ys) - 1)
        ]
        max_drop_rate = max(hip_deltas) if hip_deltas else 0.0

        if max_drop_rate < FALL_HIP_DROP_RATE_THRESHOLD:
            return False, 0.0

        # Confidence based on angle change magnitude and drop rate
        angle_change = max(late_angles) - min(early_angles)
        angle_conf = min(angle_change / 90.0, 1.0)
        drop_conf = min(max_drop_rate / (FALL_HIP_DROP_RATE_THRESHOLD * 3), 1.0)
        confidence = round((angle_conf + drop_conf) / 2, 3)

        return True, confidence

    def _detect_fight(
        self,
        camera: str,
        current_obj_id: str,
        window: deque[FrameRecord],
    ) -> tuple[bool, float]:
        """Detect a fight event by analyzing multi-person interactions.

        Requires at least two persons in the same camera frame with high
        bbox overlap (IoU > threshold) and significant arm motion from both.

        Returns:
            (detected, confidence) tuple.
        """
        if len(window) < 5:
            return False, 0.0

        # Find other persons on the same camera with sufficient history
        other_windows: list[tuple[str, deque[FrameRecord]]] = []
        prefix = f"{camera}:"
        for key, w in list(self.windows.items()):
            if key.startswith(prefix) and not key.endswith(f":{current_obj_id}"):
                if len(w) >= 5:
                    other_windows.append((key, w))

        if not other_windows:
            return False, 0.0

        current_latest = window[-1]
        max_confidence = 0.0

        for _other_key, other_window in other_windows:
            other_latest = other_window[-1]

            # Check bbox overlap
            iou = self._bbox_iou(current_latest.bbox, other_latest.bbox)
            if iou < FIGHT_IOU_THRESHOLD:
                continue

            # Compute arm motion std for both persons
            arm_std_current = self._arm_motion_std(window)
            arm_std_other = self._arm_motion_std(other_window)

            if (
                arm_std_current > FIGHT_ARM_MOTION_STD_THRESHOLD
                and arm_std_other > FIGHT_ARM_MOTION_STD_THRESHOLD
            ):
                iou_conf = min(iou / 0.6, 1.0)
                motion_conf = min(
                    (arm_std_current + arm_std_other)
                    / (FIGHT_ARM_MOTION_STD_THRESHOLD * 6),
                    1.0,
                )
                confidence = round((iou_conf + motion_conf) / 2, 3)
                max_confidence = max(max_confidence, confidence)

        if max_confidence > 0:
            return True, max_confidence

        return False, 0.0

    @staticmethod
    def _arm_motion_std(window: deque[FrameRecord]) -> float:
        """Compute the mean standard deviation of arm keypoint displacements.

        Measures frame-to-frame displacement of elbows and wrists, then
        returns the average std across those keypoints.
        """
        arm_indices = [KP_LEFT_ELBOW, KP_RIGHT_ELBOW, KP_LEFT_WRIST, KP_RIGHT_WRIST]
        records = list(window)[-10:]  # use last 10 frames

        if len(records) < 3:
            return 0.0

        displacements: list[list[float]] = [[] for _ in arm_indices]

        for i in range(1, len(records)):
            prev_kp = records[i - 1].keypoints
            curr_kp = records[i].keypoints
            for j, kp_idx in enumerate(arm_indices):
                if prev_kp[kp_idx][2] > 0.2 and curr_kp[kp_idx][2] > 0.2:
                    dy = curr_kp[kp_idx][0] - prev_kp[kp_idx][0]
                    dx = curr_kp[kp_idx][1] - prev_kp[kp_idx][1]
                    displacements[j].append(float(np.sqrt(dy**2 + dx**2)))

        stds = []
        for d in displacements:
            if len(d) >= 2:
                stds.append(float(np.std(d)))

        return float(np.mean(stds)) if stds else 0.0

    # ------------------------------------------------------------------
    # Cooldown management
    # ------------------------------------------------------------------

    def _is_in_cooldown(
        self, key: str, behavior_type: str, now: float
    ) -> bool:
        """Check whether a behavior event is still within cooldown."""
        if key not in self.cooldowns:
            return False
        last = self.cooldowns[key].get(behavior_type)
        if last is None:
            return False
        return (now - last) < self.model_config.cooldown_seconds

    def _set_cooldown(self, key: str, behavior_type: str, now: float) -> None:
        """Record a cooldown timestamp for a behavior event."""
        if key not in self.cooldowns:
            self.cooldowns[key] = {}
        self.cooldowns[key][behavior_type] = now

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process_frame(self, obj_data: dict[str, Any], frame: np.ndarray) -> None:
        """Process a single tracked object within a frame.

        Only handles person objects that are not false positives. Runs pose
        estimation, updates the sliding window, and evaluates all enabled
        behavior detection rules.

        Args:
            obj_data: Tracked object metadata (label, id, camera, box, etc.).
            frame: Full YUV frame from the camera.
        """
        if self.interpreter is None:
            return

        if obj_data.get("label") != "person":
            return

        if obj_data.get("false_positive"):
            return

        camera: str = obj_data["camera"]
        object_id: str = obj_data["id"]
        event_id: str = obj_data.get("event_id", object_id)
        box = obj_data["box"]  # [y1, x1, y2, x2]

        now = datetime.datetime.now().timestamp()

        # Crop person region from frame
        try:
            y1, x1, y2, x2 = box
            # Convert YUV to RGB for the crop region
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_NV21)
            crop = rgb_frame[y1:y2, x1:x2]
            if crop.size == 0:
                return
        except Exception:
            logger.debug(
                "Failed to crop person region for %s on %s",
                object_id,
                camera,
                exc_info=True,
            )
            return

        # Run pose estimation
        keypoints = self._run_pose_estimation(crop)
        if keypoints is None:
            return

        self.detections_per_second.update()

        # Update sliding window
        window_key = f"{camera}:{object_id}"
        if window_key not in self.windows:
            self.windows[window_key] = deque(
                maxlen=self.model_config.window_size
            )

        self.windows[window_key].append(
            FrameRecord(
                keypoints=keypoints,
                timestamp=now,
                bbox=(y1, x1, y2, x2),
            )
        )

        window = self.windows[window_key]

        # Run enabled behavior detectors
        detectors: dict[str, callable] = {
            BehaviorType.fall.value: lambda: self._detect_fall(window),
            BehaviorType.fight.value: lambda: self._detect_fight(
                camera, object_id, window
            ),
        }

        for behavior_type in self.model_config.behavior_types:
            bt = behavior_type.value
            detector = detectors.get(bt)
            if detector is None:
                continue

            detected, confidence = detector()

            if not detected:
                continue

            if confidence < self.model_config.min_confidence:
                logger.debug(
                    "%s: %s confidence %.3f below threshold %.2f for %s on %s",
                    self.model_config.name,
                    bt,
                    confidence,
                    self.model_config.min_confidence,
                    object_id,
                    camera,
                )
                continue

            if self._is_in_cooldown(window_key, bt, now):
                logger.debug(
                    "%s: %s for %s on %s still in cooldown",
                    self.model_config.name,
                    bt,
                    object_id,
                    camera,
                )
                continue

            # Publish behavior event
            logger.info(
                "%s: Detected %s (confidence=%.3f) for object %s on %s",
                self.model_config.name,
                bt,
                confidence,
                object_id,
                camera,
            )

            self.sub_label_publisher.publish(
                (camera, event_id, bt, confidence),
                EventMetadataTypeEnum.behavior_event.value,
            )

            self._set_cooldown(window_key, bt, now)

    def handle_request(
        self, topic: str, request_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Handle metadata requests such as model reload.

        Args:
            topic: Request topic string.
            request_data: Request payload.

        Returns:
            Response dict if handled, None otherwise.
        """
        if topic == EmbeddingsRequestEnum.reload_classification_model.value:
            if request_data.get("model_name") == self.model_config.name:
                self.__build_detector()
                logger.info(
                    "Successfully reloaded pose model for behavior analysis '%s'",
                    self.model_config.name,
                )
                return {
                    "success": True,
                    "message": f"Reloaded behavior analysis model {self.model_config.name}.",
                }
            return None
        return None

    def expire_object(self, object_id: str, camera: str) -> None:
        """Clean up state for an object that is no longer tracked.

        Args:
            object_id: The expired object's ID.
            camera: Camera name the object was on.
        """
        window_key = f"{camera}:{object_id}"
        self.windows.pop(window_key, None)
        self.cooldowns.pop(window_key, None)

# detectors/fall_detector.py
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

from config import config


class FallDetector:

    def __init__(self):
        self.available = False
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.45,  # Raised: reject poor detections
            min_tracking_confidence=0.45    # Raised: reject poor tracking
        )

        self.frame_count = 0
        self.current_mode = "SAFETY"

        self.fall_counter = 0
        self.upright_counter = 0
        self.fall_active = False

        self.SAFETY = config.detection.FALL_SAFETY_THRESHOLDS
        self.SPORT = config.detection.FALL_SPORT_THRESHOLDS

        # History buffers
        confirm = self.SAFETY['confirm']
        self.aspect_history = deque(maxlen=confirm * 3)
        self.shoulder_hip_diff_history = deque(maxlen=confirm * 3)
        self.hip_y_history = deque(maxlen=confirm * 3)
        self.body_orientation_history = deque(maxlen=confirm * 3)

        self.available = True

        if config.LOG_DETECTIONS:
            print("[FallDetector] OK — Anti-False-Positive v7")
            print(f"  SAFETY: confirm={self.SAFETY['confirm']}f "
                  f"checks={self.SAFETY['min_checks']}/4 "
                  f"vis={self.SAFETY['visibility']}")
            print(f"  SPORT : confirm={self.SPORT['confirm']}f "
                  f"checks={self.SPORT['min_checks']}/4 "
                  f"vis={self.SPORT['visibility']}")

    def set_mode(self, mode):
        mode = mode.upper()
        if mode != self.current_mode:
            old = self.current_mode
            self.current_mode = mode
            self._reset_counters()
            if config.LOG_DETECTIONS:
                print(f"[FallDetector] Mode: {old} → {mode}")

    def _get_thresholds(self):
        return self.SPORT if self.current_mode == "SPORT" else self.SAFETY

    def detect(self, frame):
        empty = {
            "detected": False, "confidence": 0.0,
            "landmarks": None, "alert": False, "debug": {}
        }

        if not self.available:
            return empty

        self.frame_count += 1
        thr = self._get_thresholds()

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb)
        except Exception:
            return empty

        if not result.pose_landmarks:
            self._handle_no_detection(thr)
            return {**empty, "detected": self.fall_active}

        lm = result.pose_landmarks.landmark
        PL = self.mp_pose.PoseLandmark

        # Get key landmarks
        ls = lm[PL.LEFT_SHOULDER.value]
        rs = lm[PL.RIGHT_SHOULDER.value]
        lh = lm[PL.LEFT_HIP.value]
        rh = lm[PL.RIGHT_HIP.value]
        nose = lm[PL.NOSE.value]
        lk = lm[PL.LEFT_KNEE.value]
        rk = lm[PL.RIGHT_KNEE.value]
        la = lm[PL.LEFT_ANKLE.value]
        ra = lm[PL.RIGHT_ANKLE.value]

        # Core body visibility check
        core_vis = float(np.mean([
            ls.visibility, rs.visibility,
            lh.visibility, rh.visibility
        ]))

        if core_vis < thr["visibility"]:
            self._handle_no_detection(thr)
            return {**empty, "detected": self.fall_active}

        # ── NEW: Full body visibility check ───────────────────
        # Require legs to be somewhat visible too
        # This rejects partial detections (like a tripod or racket)
        leg_vis = float(np.mean([
            lk.visibility, rk.visibility,
            la.visibility, ra.visibility
        ]))

        # If legs are completely invisible AND we're not in sport mode,
        # be very cautious — could be a non-human object
        if leg_vis < 0.15 and self.current_mode == "SAFETY":
            self._handle_no_detection(thr)
            if config.LOG_DETECTIONS and self.frame_count % 30 == 0:
                print(f"[Fall] Rejected: legs not visible (leg_vis={leg_vis:.2f})")
            return {**empty, "detected": self.fall_active}

        # Collect visible body points
        all_points = [
            (lm[PL.NOSE.value], lm[PL.NOSE.value].visibility),
            (lm[PL.LEFT_SHOULDER.value], ls.visibility),
            (lm[PL.RIGHT_SHOULDER.value], rs.visibility),
            (lm[PL.LEFT_HIP.value], lh.visibility),
            (lm[PL.RIGHT_HIP.value], rh.visibility),
            (lm[PL.LEFT_KNEE.value], lk.visibility),
            (lm[PL.RIGHT_KNEE.value], rk.visibility),
            (lm[PL.LEFT_ANKLE.value], la.visibility),
            (lm[PL.RIGHT_ANKLE.value], ra.visibility),
        ]

        visible_coords = [(p.x, p.y) for p, v in all_points if v > 0.2]

        # ── Need at least 6 visible points (not just 4) ───────
        if len(visible_coords) < 6:
            self._handle_no_detection(thr)
            return {**empty, "detected": self.fall_active}

        # Bounding box
        xs = [p[0] for p in visible_coords]
        ys = [p[1] for p in visible_coords]
        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys)

        # Aspect ratio
        aspect = bbox_w / bbox_h if bbox_h > 0.01 else 0

        # Shoulder/hip metrics
        shoulder_y = (ls.y + rs.y) / 2
        hip_y = (lh.y + rh.y) / 2
        shoulder_hip_diff = abs(hip_y - shoulder_y)

        shoulder_x = (ls.x + rs.x) / 2
        hip_x = (lh.x + rh.x) / 2
        dx = hip_x - shoulder_x
        dy = hip_y - shoulder_y

        if abs(dy) > 0.001:
            orientation_deg = np.degrees(np.arctan2(abs(dx), abs(dy)))
        else:
            orientation_deg = 90

        # Update histories
        self.aspect_history.append(aspect)
        self.shoulder_hip_diff_history.append(shoulder_hip_diff)
        self.hip_y_history.append(hip_y)
        self.body_orientation_history.append(orientation_deg)

        # Smoothed values
        avg_aspect = float(np.mean(self.aspect_history))
        avg_sh_diff = float(np.mean(self.shoulder_hip_diff_history))
        avg_hip_y = float(np.mean(self.hip_y_history))
        avg_orientation = float(np.mean(self.body_orientation_history))

        # ── Fall detection checks ─────────────────────────────
        check_aspect = avg_aspect >= thr["aspect_min"]
        check_sh_diff = avg_sh_diff <= thr["shoulder_hip_diff_max"]
        check_hip = avg_hip_y >= thr["hip_y_min"]
        check_orientation = avg_orientation >= thr["orientation_min"]

        checks_passed = sum([
            check_aspect, check_sh_diff,
            check_hip, check_orientation
        ])

        # ── Anti-false-positive guards ────────────────────────

        # Guard 1: Need history before deciding
        history_ready = len(self.aspect_history) >= max(3, thr["confirm"])

        # Guard 2: Aspect must be CONSISTENTLY wide (not flickering)
        aspect_stable = False
        if len(self.aspect_history) >= 3:
            recent = list(self.aspect_history)[-3:]
            aspect_stable = all(a >= thr["aspect_min"] * 0.9 for a in recent)

        # Guard 3: Body must NOT be in top 25% of frame
        hip_not_at_top = avg_hip_y > 0.25

        # Guard 4: Enough landmark points
        enough_landmarks = len(visible_coords) >= 6

        # Guard 5: Spine vector must make sense
        # (hip should be below shoulder in a real person)
        hip_below_shoulder = hip_y > shoulder_y - 0.05

        # ── Final fall decision ───────────────────────────────
        is_falling = (
            checks_passed >= thr["min_checks"]
            and history_ready
            and aspect_stable
            and hip_not_at_top
            and enough_landmarks
            and hip_below_shoulder
        )

        # ── Update counters ───────────────────────────────────
        if is_falling:
            self.fall_counter = min(
                self.fall_counter + 1,
                thr["confirm"] * 2
            )
            self.upright_counter = 0
        else:
            self.upright_counter += 1
            if self.upright_counter >= thr["reset"]:
                self.fall_counter = max(0, self.fall_counter - 1)
                if self.fall_counter == 0:
                    self.fall_active = False

        # ── Trigger alert ─────────────────────────────────────
        new_alert = False
        if self.fall_counter >= thr["confirm"]:
            if not self.fall_active:
                self.fall_active = True
                new_alert = True
                if config.LOG_DETECTIONS:
                    print(
                        f"\n[FallDetector] 🚨 FALL DETECTED! Mode:{self.current_mode}"
                    )
                    print(
                        f"  asp:{avg_aspect:.2f}({'✓' if check_aspect else '✗'}) "
                        f"sh:{avg_sh_diff:.3f}({'✓' if check_sh_diff else '✗'}) "
                        f"hy:{avg_hip_y:.2f}({'✓' if check_hip else '✗'}) "
                        f"ori:{avg_orientation:.0f}°"
                        f"({'✓' if check_orientation else '✗'}) "
                        f"chk:{checks_passed}/4 "
                        f"leg_vis:{leg_vis:.2f}"
                    )

        # ── Debug log ─────────────────────────────────────────
        if config.LOG_DETECTIONS and self.frame_count % 30 == 0:
            guards = (
                f"hist:{'✓' if history_ready else '✗'} "
                f"stbl:{'✓' if aspect_stable else '✗'} "
                f"top:{'✓' if hip_not_at_top else '✗'} "
                f"lmk:{'✓' if enough_landmarks else '✗'}"
            )
            print(
                f"[Fall/{self.current_mode}] "
                f"asp:{avg_aspect:.2f}({'✓' if check_aspect else '✗'}) "
                f"sh:{avg_sh_diff:.3f}({'✓' if check_sh_diff else '✗'}) "
                f"hy:{avg_hip_y:.2f}({'✓' if check_hip else '✗'}) "
                f"ori:{avg_orientation:.0f}°({'✓' if check_orientation else '✗'}) "
                f"chk:{checks_passed}/4 cnt:{self.fall_counter}/{thr['confirm']} "
                f"vis:{core_vis:.2f} leg:{leg_vis:.2f} | {guards}"
            )

        confidence = min(self.fall_counter / thr["confirm"], 1.0)

        return {
            "detected": self.fall_active,
            "confidence": confidence,
            "landmarks": result.pose_landmarks,
            "alert": new_alert,
            "debug": {
                "aspect": round(avg_aspect, 2),
                "shoulder_hip_diff": round(avg_sh_diff, 3),
                "hip_y": round(avg_hip_y, 2),
                "orientation": round(avg_orientation, 0),
                "checks": checks_passed,
                "visibility": round(core_vis, 2),
                "leg_visibility": round(leg_vis, 2),
                "guards": {
                    "history_ready": history_ready,
                    "aspect_stable": aspect_stable,
                    "hip_not_at_top": hip_not_at_top,
                    "enough_landmarks": enough_landmarks
                }
            }
        }

    def _handle_no_detection(self, thr):
        """No pose detected — gradually decay fall counter."""
        self.upright_counter += 1
        if self.upright_counter >= thr["reset"]:
            self.fall_counter = max(0, self.fall_counter - 1)
            if self.fall_counter == 0:
                self.fall_active = False

    def _reset_counters(self):
        self.fall_counter = 0
        self.upright_counter = 0
        self.fall_active = False
        self.aspect_history.clear()
        self.shoulder_hip_diff_history.clear()
        self.hip_y_history.clear()
        self.body_orientation_history.clear()

    def draw_skeleton(self, frame, landmarks):
        if self.available and landmarks:
            drawing_spec = self.mp_draw.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=3
            )
            connection_spec = self.mp_draw.DrawingSpec(
                color=(255, 255, 0), thickness=2
            )
            self.mp_draw.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=connection_spec
            )
        return frame

    def set_thresholds(self, mode="SAFETY", **kwargs):
        thr = self.SPORT if mode == "SPORT" else self.SAFETY
        for key, val in kwargs.items():
            if key in thr and val is not None:
                thr[key] = val
        if config.LOG_DETECTIONS:
            print(f"[FallDetector] {mode} thresholds updated: {kwargs}")

    def reset(self):
        self._reset_counters()
        self.frame_count = 0
        if config.LOG_DETECTIONS:
            print("[FallDetector] Reset")
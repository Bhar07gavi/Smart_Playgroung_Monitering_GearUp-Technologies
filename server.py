# server.py
# ============================================================
# Smart Playground Monitor — v5.0 (FINAL)
# ============================================================

import cv2
import numpy as np
import threading
import time
import json
import os
import socket
import urllib.request
import asyncio
from collections import deque
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import uvicorn
from dotenv import load_dotenv

load_dotenv()

from cloud.supabase_client import SupabaseClient
from detectors.motion_detector import MotionDetector
from detectors.fall_detector import FallDetector
from detectors.sport_detector import SportDetector
from detectors.uniform_detector import UniformDetector

from tracking.player_tracker import PlayerTracker
from tracking.speed_estimator import SpeedEstimator

from referee.referee_manager import RefereeManager

from video.event_buffer import EventBuffer
from video.clip_writer import ClipWriter

from config import config


# ============================================================
# ESP32 READER
# ============================================================

class ESP32Reader:
    def __init__(self):
        self._frame   = None
        self._lock    = threading.Lock()
        self._running = False

        self.connected   = False
        self.fps         = 0.0
        self.frame_count = 0

        self._fps_frames = 0
        self._fps_time   = time.time()

        self.ip     = config.esp32.IP_ADDRESS
        self.rssi   = 0
        self.heap   = 0
        self.uptime = 0
        self._retry = 0

    def start(self):
        self._running = True
        threading.Thread(target=self._stream_loop,
                         daemon=True, name="ESP32Stream").start()
        threading.Thread(target=self._info_loop,
                         daemon=True, name="ESP32Info").start()
        print(f"[ESP32] Stream URL: {config.esp32.STREAM_URL}")

    def stop(self):
        self._running = False

    def get_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def _stream_loop(self):
        SOI = b"\xff\xd8"
        EOI = b"\xff\xd9"

        while self._running:
            try:
                self._retry += 1
                if self._retry > 1:
                    wait = min(self._retry * 2, config.esp32.MAX_RETRIES)
                    print(f"[ESP32] Retry {self._retry} in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[ESP32] Connecting to {config.esp32.STREAM_URL}")

                req = urllib.request.Request(
                    config.esp32.STREAM_URL,
                    headers={
                        "Connection":    "keep-alive",
                        "Cache-Control": "no-cache",
                        "User-Agent":    "SmartPlayground"
                    }
                )
                stream = urllib.request.urlopen(
                    req, timeout=config.esp32.READ_TIMEOUT)

                self.connected = True
                self._retry    = 0
                print("[ESP32] Connected")

                buffer = bytes()
                while self._running:
                    chunk = stream.read(config.esp32.CHUNK_SIZE)
                    if not chunk:
                        break
                    buffer += chunk

                    if len(buffer) > config.esp32.CHUNK_SIZE * 50:
                        buffer = buffer[-config.esp32.CHUNK_SIZE * 25:]

                    while True:
                        s = buffer.find(SOI)
                        e = buffer.find(EOI)
                        if s == -1 or e == -1:
                            break
                        if e < s:
                            buffer = buffer[s + 2:]
                            continue

                        jpg    = buffer[s:e + 2]
                        buffer = buffer[e + 2:]

                        arr   = np.frombuffer(jpg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self._lock:
                                self._frame = frame
                            self.frame_count += 1
                            self._fps_frames += 1
                            elapsed = time.time() - self._fps_time
                            if elapsed >= 2.0:
                                self.fps = round(
                                    self._fps_frames / elapsed, 1)
                                self._fps_frames = 0
                                self._fps_time   = time.time()

            except Exception as e:
                self.connected = False
                print(f"[ESP32] Error: {e}")

    def _info_loop(self):
        while True:
            if not self._running:
                break
            try:
                resp = urllib.request.urlopen(
                    config.esp32.STATUS_URL, timeout=5)
                data        = json.loads(resp.read().decode())
                self.rssi   = data.get("rssi",       0)
                self.heap   = data.get("heap_free",  0)
                self.uptime = data.get("uptime_sec", 0)
                self.ip     = data.get("ip", config.esp32.IP_ADDRESS)
            except Exception:
                pass
            time.sleep(config.esp32.TARGET_FPS / 2)

    def get_info(self):
        return {
            "ip":          self.ip,
            "rssi":        self.rssi,
            "heap_free":   self.heap,
            "uptime_sec":  self.uptime,
            "fps":         self.fps,
            "frame_count": self.frame_count,
            "connected":   self.connected,
            "stream_url":  config.esp32.STREAM_URL
        }


# ============================================================
# STREAM RELAY
# ============================================================

class StreamRelay:
    def __init__(self):
        self._jpeg = None
        self._lock = threading.Lock()

        def _make_img(msg1, msg2="", col=(0, 180, 255)):
            img = np.zeros(
                (config.esp32.FRAME_HEIGHT,
                 config.esp32.FRAME_WIDTH, 3),
                dtype=np.uint8)
            cv2.putText(img, msg1,
                        (int(config.esp32.FRAME_WIDTH  * 0.08),
                         int(config.esp32.FRAME_HEIGHT * 0.45)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.80, col, 2)
            if msg2:
                cv2.putText(img, msg2,
                            (int(config.esp32.FRAME_WIDTH  * 0.15),
                             int(config.esp32.FRAME_HEIGHT * 0.58)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (70, 70, 70), 1)
            _, buf = cv2.imencode(".jpg", img)
            return buf.tobytes()

        self._placeholder = _make_img(
            "Waiting for ESP32-CAM...", "", (0, 180, 255))
        self._paused      = _make_img(
            "Detection Paused",
            "Press START to resume", (70, 70, 70))

    def update(self, frame):
        if (frame.shape[0] != config.esp32.FRAME_HEIGHT or
                frame.shape[1] != config.esp32.FRAME_WIDTH):
            frame = cv2.resize(
                frame,
                (config.esp32.FRAME_WIDTH, config.esp32.FRAME_HEIGHT))
        ok, buf = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, config.api.STREAM_QUALITY])
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()

    def get_jpeg(self):
        with self._lock:
            return self._jpeg if self._jpeg else self._placeholder

    def get_paused(self):
        return self._paused


# ============================================================
# DETECTION ACTIVE FLAG
# ============================================================

_detection_active = True
_det_lock         = threading.Lock()


def is_active() -> bool:
    with _det_lock:
        return _detection_active


# ============================================================
# INITIALIZE COMPONENTS
# ============================================================

print("\n" + "=" * 60)
print(" SMART PLAYGROUND MONITOR v5.0")
print("=" * 60)

esp32        = ESP32Reader()
relay        = StreamRelay()

motion_det   = MotionDetector()
fall_det     = FallDetector()
sport_det    = SportDetector()
uniform_det  = UniformDetector()

player_track = PlayerTracker()
speed_est    = SpeedEstimator(
    pixels_per_meter=config.detection.PIXELS_PER_METER)

referee_man  = RefereeManager()
event_buf    = EventBuffer()
clip_writer  = ClipWriter()
cloud        = SupabaseClient()

print("=" * 60)


# ============================================================
# SHARED STATE
# ============================================================

state = {
    "mode":           "SAFETY",
    "sport":          None,
    "sport_confidence": 0.0,

    "sport_candidate":      None,
    "sport_candidate_conf": 0.0,

    "fall_detected": False,
    "fall_conf":     0.0,

    "unauthorized":   False,
    "uniform_status": "NO_PERSON",

    "referee_msg":  "",
    "referee_mode": "OFF",

    "motion_score": 0.0,

    "player_count":   0,
    "avg_speed_kmh":  0.0,
    "fastest_speed":  0.0,
    "fastest_player": None,

    "recording":    False,
    "rec_progress": 0.0,

    "esp32_connected":  False,
    "esp32_fps":        0.0,
    "esp32_ip":         "",
    "esp32_rssi":       0,
    "esp32_uptime_sec": 0,
    "esp32_heap_free":  0,
    "frame_count":      0,

    "local_clips":  [],
    "cloud_events": [],

    "sport_switch_candidate": None,
    "sport_switch_progress":  0.0,
    "frames_in_sport":        0,

    "detection_active": True,
}

state_lock = threading.Lock()


# ============================================================
# AI IMAGE ENHANCEMENT
# ============================================================

def enhance_frame(frame):
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        return cv2.filter2D(enhanced, -1, kernel)
    except Exception:
        return frame


# ============================================================
# SMART AI LOOP — v8 FINAL
# ============================================================

def ai_loop():
    print("[AI] Smart loop v8 started")

    # ── All thresholds — hardcoded to prevent AttributeError ──
    SPORT_ENTER_FRAMES        = 25    # Frames needed to confirm sport entry
    SPORT_EXIT_FRAMES         = 10    # Frames of no-sport → SAFETY
    SPORT_ENTER_CONF          = 0.80  # Min confidence to enter SPORT
    SPORT_CANDIDATE_CONF      = 0.50  # Min conf to show candidate on screen
    SPORT_LOCK_CONF           = 0.75  # New sport needs this to compete
    SPORT_LOCK_MIN_FRAMES     = 10    # Min frames in sport before switch
    SPORT_SWITCH_FRAMES       = 20    # Frames new sport must appear (normal)
    SPORT_SWITCH_FAST_CONF    = 0.90  # Fast-track confidence
    SPORT_SWITCH_FAST_FRAMES  = 10    # Fast-track frame count
    SPORT_CONF_REFEREE        = 0.80  # Min conf for referee system
    WARMUP_FRAMES             = 30    # Startup warmup frames

    # ── Motion guards — prevent crowd/people triggering SPORT ─
    # When motion is very high (e.g. crowd walking), we need
    # even higher confidence to confirm it's really a sport
    MOTION_SPORT_GUARD_THRESH = 50.0  # If motion > this, use higher conf
    MOTION_SPORT_GUARD_CONF   = 0.88  # Need 88%+ conf when motion > 50

    # ── Minimum consecutive high-confidence frames ─────────────
    # Even if conf is high, we need N frames without dropping below
    # SPORT_ENTER_CONF to prevent single-frame false triggers
    SPORT_STABLE_FRAMES       = 5     # Must be stable for 5 frames at 80%+
    # ─────────────────────────────────────────────────────────

    frame_count  = 0
    current_mode = "SAFETY"

    motion_history = deque(maxlen=20)
    last_cloud     = 0
    last_esp32     = 0

    locked_sport      = None
    locked_sport_conf = 0.0

    sport_confirm_frames    = 0
    sport_stable_count      = 0   # Consecutive frames at high conf
    non_sport_frames        = 0
    frames_in_current_sport = 0
    sport_switch_candidate  = None
    sport_switch_frames     = 0

    print(f"[AI] Warming up ({WARMUP_FRAMES} frames)...")

    while True:

        # ── PAUSE CHECK ───────────────────────────────────────
        if not is_active():
            with state_lock:
                state["detection_active"] = False
                state["mode"]             = "PAUSED"
                state["referee_msg"]      = "Detection paused by user."
            relay._jpeg = relay.get_paused()
            time.sleep(0.2)
            continue

        with state_lock:
            state["detection_active"] = True
        # ─────────────────────────────────────────────────────

        frame = esp32.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        frame_count += 1

        # ── Warmup ────────────────────────────────────────────
        if frame_count <= WARMUP_FRAMES:
            relay.update(frame)
            if frame_count == WARMUP_FRAMES:
                print("[AI] Warm-up complete — starting detection")
            continue

        ai_frame = enhance_frame(frame)

        # ─────────────────────────────────────────────────────
        # DETECTORS
        # ─────────────────────────────────────────────────────

        motion = motion_det.calculate(frame)
        motion_history.append(motion)
        avg_motion = float(np.mean(motion_history))

        fall = fall_det.detect(frame)

        # Uniform — skip on fall or high motion
       
        UNIFORM_MOTION_SKIP = 60.0  # Only skip if truly chaotic motion

        if fall.get("detected") or fall.get("confidence", 0) > 0.3:
            uniform = {
                "status":     "FALL_OVERRIDE",
                "confidence": 0.0,
                "stable":     False,
                "alert":      False
            }
        elif avg_motion > UNIFORM_MOTION_SKIP:
            uniform = {
                "status":     "MOTION_TOO_HIGH",
                "confidence": 0.0,
                "stable":     False,
                "alert":      False
            }
        else:
            # Motion is acceptable — run detection
            uniform = uniform_det.predict(ai_frame)
        # Sport — skip on dark frames
        gray            = cv2.cvtColor(ai_frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < 30:
            sport = {"class": "unknown", "confidence": 0.0}
        else:
            sport = sport_det.predict(ai_frame)

        sport_class = sport.get("class", "unknown")
        sport_conf  = float(sport.get("confidence", 0.0))
        fall_active = (fall.get("detected", False)
                       or fall.get("alert", False))

        # ── Motion-aware confidence threshold ─────────────────
        # Crowd movement creates high motion + false sport detections
        # Require higher confidence when motion is very high
        effective_enter_conf = (
            MOTION_SPORT_GUARD_CONF
            if avg_motion > MOTION_SPORT_GUARD_THRESH
            else SPORT_ENTER_CONF
        )

        valid_sport = (
            sport_class != "unknown"
            and sport_conf >= effective_enter_conf
        )

        sport_visible = (
            sport_class != "unknown"
            and sport_conf >= SPORT_CANDIDATE_CONF
        )

        # Track time in current sport
        if current_mode == "SPORT":
            frames_in_current_sport += 1
        else:
            frames_in_current_sport = 0

        # ─────────────────────────────────────────────────────
        # MODE SWITCHING — v8 WITH MOTION GUARD
        # ─────────────────────────────────────────────────────

        # P1: Fall → SAFETY instantly
        if config.detection.FALL_FORCE_SAFETY and fall_active:
            if current_mode != "SAFETY":
                if config.LOG_DETECTIONS:
                    print("[AI] FALL → forced SAFETY")
                current_mode = "SAFETY"
                event_buf.clear()
            sport_confirm_frames    = 0
            sport_stable_count      = 0
            non_sport_frames        = 0
            frames_in_current_sport = 0
            sport_switch_candidate  = None
            sport_switch_frames     = 0
            locked_sport            = None
            locked_sport_conf       = 0.0
            if hasattr(fall_det, "set_mode"):
                fall_det.set_mode("SAFETY")

        # P2: Valid sport (passes motion guard)
        elif valid_sport:

            # Already in SPORT → lock logic
            if current_mode == "SPORT" and locked_sport is not None:

                if sport_class == locked_sport:
                    # Same sport — reinforce
                    sport_switch_candidate = None
                    sport_switch_frames    = 0
                    locked_sport_conf      = sport_conf
                    non_sport_frames       = 0
                    sport_stable_count     += 1

                else:
                    # Different sport — needs lock conf + min time
                    can_switch = (
                        sport_conf >= SPORT_LOCK_CONF
                        and frames_in_current_sport >= SPORT_LOCK_MIN_FRAMES
                    )
                    if can_switch:
                        if sport_class == sport_switch_candidate:
                            sport_switch_frames += 1
                        else:
                            sport_switch_candidate = sport_class
                            sport_switch_frames    = 1
                            if config.LOG_DETECTIONS:
                                print(
                                    f"[AI] Switch candidate: "
                                    f"{sport_class} ({sport_conf:.0%})"
                                )

                        needed = (SPORT_SWITCH_FAST_FRAMES
                                  if sport_conf >= SPORT_SWITCH_FAST_CONF
                                  else SPORT_SWITCH_FRAMES)

                        if sport_switch_frames >= needed:
                            tag = ("(FAST)"
                                   if sport_conf >= SPORT_SWITCH_FAST_CONF
                                   else "")
                            if config.LOG_DETECTIONS:
                                print(
                                    f"[AI] Sport SWITCH {tag}: "
                                    f"{locked_sport} → {sport_class} "
                                    f"({sport_conf:.0%})"
                                )
                            locked_sport            = sport_class
                            locked_sport_conf       = sport_conf
                            frames_in_current_sport = 0
                            sport_switch_candidate  = None
                            sport_switch_frames     = 0
                            sport_stable_count      = 0
                            player_track.set_sport(locked_sport)
                            speed_est.reset()
                            if hasattr(uniform_det, "reset"):
                                uniform_det.reset()

                        else:
                            if (config.LOG_DETECTIONS
                                    and sport_switch_frames % 5 == 0):
                                print(
                                    f"[AI] Switch: {sport_class} "
                                    f"({sport_conf:.0%}) "
                                    f"{sport_switch_frames}/{needed}f"
                                    f" — keeping {locked_sport}"
                                )
                    else:
                        sport_switch_candidate = None
                        sport_switch_frames    = 0
                        if config.LOG_DETECTIONS and frame_count % 60 == 0:
                            reason = (
                                f"conf {sport_conf:.0%} < {SPORT_LOCK_CONF:.0%}"
                                if sport_conf < SPORT_LOCK_CONF
                                else f"{frames_in_current_sport}f < "
                                     f"{SPORT_LOCK_MIN_FRAMES}f"
                            )
                            print(
                                f"[AI] Ignoring {sport_class} "
                                f"({reason})"
                            )

            # Not in SPORT → entry path
            else:
                non_sport_frames = 0

                # Track stability — consecutive frames at effective_enter_conf
                if sport_conf >= effective_enter_conf:
                    sport_stable_count   += 1
                    sport_confirm_frames += 1
                else:
                    # Below threshold even in valid_sport branch
                    # (can happen between the two conf checks)
                    sport_stable_count = 0

                if config.LOG_DETECTIONS and sport_confirm_frames % 5 == 0:
                    print(
                        f"[AI] Sport entry: {sport_class} "
                        f"({sport_conf:.0%}) "
                        f"{sport_confirm_frames}/{SPORT_ENTER_FRAMES}f "
                        f"stable={sport_stable_count} "
                        f"motion={avg_motion:.1f} "
                        f"eff_conf={effective_enter_conf:.0%}"
                    )

                # Require BOTH enough frames AND stable high-conf frames
                entry_ready = (
                    sport_confirm_frames >= SPORT_ENTER_FRAMES
                    and sport_stable_count >= SPORT_STABLE_FRAMES
                )

                if entry_ready and current_mode != "SPORT":
                    if config.LOG_DETECTIONS:
                        print(
                            f"[AI] SPORT confirmed: {sport_class} "
                            f"({sport_conf:.0%}) → SPORT mode "
                            f"(motion={avg_motion:.1f})"
                        )
                    current_mode      = "SPORT"
                    locked_sport      = sport_class
                    locked_sport_conf = sport_conf
                    event_buf.clear()
                    if hasattr(uniform_det, "reset"):
                        uniform_det.reset()
                    player_track.set_sport(sport_class)
                    speed_est.reset()
                    frames_in_current_sport = 0
                    sport_switch_candidate  = None
                    sport_switch_frames     = 0
                    if hasattr(fall_det, "set_mode"):
                        fall_det.set_mode("SPORT")

                elif current_mode == "SPORT":
                    locked_sport      = sport_class
                    locked_sport_conf = sport_conf

        # P3: High motion + sport visible
        # Motion guard enforced — needs very high conf in high-motion scenes
        elif (avg_motion >= config.detection.MOTION_SPORT_THRESHOLD
              and sport_visible):

            # How much conf do we need given the motion level?
            needed_conf = (
                MOTION_SPORT_GUARD_CONF
                if avg_motion > MOTION_SPORT_GUARD_THRESH
                else SPORT_LOCK_CONF
            )

            if current_mode != "SPORT" and sport_conf >= needed_conf:
                sport_confirm_frames += 1
                non_sport_frames      = 0

                if sport_conf >= effective_enter_conf:
                    sport_stable_count += 1
                else:
                    sport_stable_count = 0

                entry_ready = (
                    sport_confirm_frames >= SPORT_ENTER_FRAMES
                    and sport_stable_count >= SPORT_STABLE_FRAMES
                )

                if entry_ready:
                    if config.LOG_DETECTIONS:
                        print(
                            f"[AI] HIGH MOTION + {sport_class} "
                            f"({sport_conf:.0%}) → SPORT"
                        )
                    current_mode = "SPORT"
                    event_buf.clear()
                    if hasattr(uniform_det, "reset"):
                        uniform_det.reset()
                    player_track.set_sport(sport_class)
                    speed_est.reset()
                    locked_sport            = sport_class
                    locked_sport_conf       = max(sport_conf,
                                                  SPORT_ENTER_CONF)
                    frames_in_current_sport = 0
                    sport_switch_candidate  = None
                    sport_switch_frames     = 0
                    if hasattr(fall_det, "set_mode"):
                        fall_det.set_mode("SPORT")

            elif current_mode != "SPORT":
                # High motion but conf not sufficient → reset counter
                if config.LOG_DETECTIONS and frame_count % 30 == 0:
                    print(
                        f"[AI] Rejected: {sport_class} ({sport_conf:.0%}) "
                        f"motion={avg_motion:.1f} "
                        f"need {needed_conf:.0%} — likely crowd"
                    )
                sport_confirm_frames = 0
                sport_stable_count   = 0

            else:
                # Already in SPORT — motion hint cannot switch sport
                if sport_class == locked_sport:
                    locked_sport_conf = sport_conf

        # P4: In SPORT + sport still visible → STAY
        elif current_mode == "SPORT" and sport_visible:
            sport_confirm_frames   = max(0, sport_confirm_frames - 1)
            non_sport_frames       = 0
            sport_switch_candidate = None
            sport_switch_frames    = 0
            if sport_class == locked_sport:
                locked_sport_conf = sport_conf

        # P5: No sport visible → count toward SAFETY
        elif not sport_visible:
            non_sport_frames     += 1
            sport_confirm_frames  = 0
            sport_stable_count    = 0

            if (non_sport_frames >= SPORT_EXIT_FRAMES
                    and current_mode != "SAFETY"):
                if config.LOG_DETECTIONS:
                    print(
                        f"[AI] No sport {non_sport_frames}f → SAFETY"
                    )
                current_mode            = "SAFETY"
                locked_sport            = None
                locked_sport_conf       = 0.0
                frames_in_current_sport = 0
                sport_switch_candidate  = None
                sport_switch_frames     = 0
                event_buf.clear()
                player_track.reset()
                speed_est.reset()
                if hasattr(fall_det, "set_mode"):
                    fall_det.set_mode("SAFETY")

        # P6: Default — drift to SAFETY
        else:
            non_sport_frames     += 1
            sport_confirm_frames  = max(0, sport_confirm_frames - 1)
            sport_stable_count    = max(0, sport_stable_count - 1)

            if (non_sport_frames >= SPORT_EXIT_FRAMES
                    and current_mode != "SAFETY"):
                if config.LOG_DETECTIONS:
                    print("[AI] Drifting to SAFETY")
                current_mode            = "SAFETY"
                locked_sport            = None
                locked_sport_conf       = 0.0
                frames_in_current_sport = 0
                sport_switch_candidate  = None
                sport_switch_frames     = 0
                event_buf.clear()
                player_track.reset()
                speed_est.reset()
                if hasattr(fall_det, "set_mode"):
                    fall_det.set_mode("SAFETY")

        # ─────────────────────────────────────────────────────
        # DEFAULTS
        # ─────────────────────────────────────────────────────
        players        = []
        speeds         = {}
        avg_speed_kmh  = 0.0
        fastest_speed  = 0.0
        fastest_player = None
        ref_msg        = ""
        ref_mode       = "OFF"

        # ─────────────────────────────────────────────────────
        # SPORT MODE — tracking + referee
        # ─────────────────────────────────────────────────────
        if current_mode == "SPORT" and locked_sport:
            player_track.set_sport(locked_sport)
            players        = player_track.detect(frame)
            speeds         = speed_est.update(players)
            avg_speed_kmh  = speed_est.avg_speed(speeds)
            fastest_player, fastest_speed = speed_est.fastest(speeds)

            if locked_sport_conf >= SPORT_CONF_REFEREE:
                referee  = referee_man.update(
                    locked_sport, frame, players, speeds,
                    sport_confidence=locked_sport_conf)
                ref_msg  = referee.get("message", "")
                ref_mode = referee.get("mode", "ANALYTICS")
            else:
                ref_msg  = f"Detecting {locked_sport}..."
                ref_mode = "WAITING"

        # ─────────────────────────────────────────────────────
        # EVENT TRIGGERS
        # ─────────────────────────────────────────────────────
                # EVENT TRIGGERS
        if fall.get("alert") or fall.get("detected"):
            event_buf.trigger(
                "fall",
                {"confidence": fall.get("confidence", 0.0)})
        elif current_mode == "SAFETY":
            # Trigger on alert OR stable unauthorized detection
            unauth_stable = (
                uniform.get("status") == "UNAUTHORIZED"
                and uniform.get("stable", False)
                and uniform.get("confidence", 0) > 0.45
            )
            if uniform.get("alert") or unauth_stable:
                event_buf.trigger(
                    "unauthorized",
                    {"confidence": uniform.get("confidence", 0.0)})
        clip = event_buf.add_frame(frame)
        if clip:
            handle_clip(clip, current_mode, locked_sport)

        # ─────────────────────────────────────────────────────
        # ANNOTATE + RELAY
        # ─────────────────────────────────────────────────────
        annotated = annotate(
            frame.copy(), current_mode, avg_motion,
            fall, uniform, sport,
            locked_sport, locked_sport_conf,
            ref_msg, ref_mode, frame_count,
            players, speeds, avg_speed_kmh,
            fastest_speed, fastest_player,
            SPORT_CANDIDATE_CONF
        )
        relay.update(annotated)

        # ─────────────────────────────────────────────────────
        # CLOUD PUSH
        # ─────────────────────────────────────────────────────
        now = time.time()
        if now - last_cloud >= config.api.STATUS_WS_INTERVAL:
            push_cloud(
                current_mode, locked_sport, locked_sport_conf,
                fall, uniform, ref_msg, ref_mode, avg_motion,
                frame_count, len(players),
                avg_speed_kmh, fastest_speed, fastest_player)
            last_cloud = now

        if now - last_esp32 >= config.esp32.ESP32_INTERVAL:
            threading.Thread(
                target=cloud.push_esp32_status,
                args=(esp32.get_info(),),
                daemon=True).start()
            last_esp32 = now

        # ─────────────────────────────────────────────────────
        # STATE UPDATE
        # ─────────────────────────────────────────────────────
        esp_info = esp32.get_info()
        with state_lock:
            state.update({
                "mode":  current_mode,
                "sport": locked_sport if current_mode == "SPORT" else None,
                "sport_confidence": (
                    round(locked_sport_conf, 3)
                    if current_mode == "SPORT" else 0.0),

                "sport_candidate": (
                    sport_class
                    if sport_class != "unknown" and not valid_sport
                    else None),
                "sport_candidate_conf": round(sport_conf, 3),

                "fall_detected": fall.get("detected", False),
                "fall_conf":     fall.get("confidence", 0.0),

                "unauthorized": (
                    current_mode == "SAFETY"
                    and uniform.get("status") == "UNAUTHORIZED"),
                "uniform_status": uniform.get("status", ""),

                "referee_msg":  ref_msg,
                "referee_mode": ref_mode,

                "motion_score": round(avg_motion, 2),

                "player_count":   len(players),
                "avg_speed_kmh":  avg_speed_kmh,
                "fastest_speed":  fastest_speed,
                "fastest_player": fastest_player,

                "recording":    event_buf.is_recording,
                "rec_progress": event_buf.progress,

                "esp32_connected":  esp32.connected,
                "esp32_fps":        esp32.fps,
                "esp32_ip":         esp_info["ip"],
                "esp32_rssi":       esp_info["rssi"],
                "esp32_uptime_sec": esp_info["uptime_sec"],
                "esp32_heap_free":  esp_info["heap_free"],
                "frame_count":      frame_count,

                "local_clips":  clip_writer.get_all_local_clips(),

                "sport_switch_candidate": sport_switch_candidate,
                "sport_switch_progress": (
                    round(sport_switch_frames / SPORT_SWITCH_FRAMES, 2)
                    if sport_switch_candidate else 0.0),
                "frames_in_sport":  frames_in_current_sport,
                "detection_active": True,
            })


# ============================================================
# HELPERS
# ============================================================

def handle_clip(clip, mode, sport):
    saved = clip_writer.write(clip)
    if not saved:
        return
    metadata = {
        "event_type": clip.get("event_type", "unknown"),
        "sport": sport,
        "mode":  mode
    }
    def upload():
        try:
            eid = cloud.process_event(saved, metadata)
            if eid:
                evts = cloud.get_recent_events(10)
                with state_lock:
                    state["cloud_events"] = evts
        except Exception as e:
            print(f"[Cloud] Upload error: {e}")
    threading.Thread(target=upload, daemon=True).start()


def push_cloud(mode, sport, sport_conf, fall, uniform,
               ref_msg, ref_mode, motion, frame_count,
               player_count, avg_speed, fastest_speed, fastest_player):
    data = {
        "mode":             mode,
        "sport_detected":   sport,
        "sport_confidence": round(sport_conf, 3),
        "fall_detected":    fall.get("detected", False),
        "fall_confidence":  round(fall.get("confidence", 0.0), 3),
        "unauthorized": (mode == "SAFETY"
                         and uniform.get("status") == "UNAUTHORIZED"),
        "uniform_status":  uniform.get("status", ""),
        "referee_mode":    ref_mode,
        "referee_message": ref_msg,
        "motion_score":    round(motion, 2),
        "recording_event": event_buf.is_recording,
        "esp32_connected": esp32.connected,
        "esp32_fps":       esp32.fps,
        "esp32_ip":        esp32.ip,
        "frame_count":     frame_count,
        "player_count":    player_count,
        "avg_speed_kmh":   avg_speed,
        "fastest_speed":   fastest_speed,
        "fastest_player":  fastest_player,
        "max_players":     player_track.max_players,
        "sport_desc":      player_track.description
    }
    threading.Thread(
        target=cloud.push_live_status,
        args=(data,), daemon=True).start()


def annotate(frame, mode, motion, fall, uniform, sport,
             locked_sport, locked_conf, ref_msg, ref_mode,
             frame_count, players, speeds,
             avg_speed, fastest_speed, fastest_player,
             sport_candidate_conf=0.50):
    h, w = frame.shape[:2]

    banner = (30, 150, 30) if mode == "SPORT" else (140, 70, 0)
    cv2.rectangle(frame, (0, 0), (w, 34), banner, -1)
    cv2.putText(
        frame,
        f"MODE:{mode}  Motion:{motion:.1f}  "
        f"FPS:{esp32.fps:.0f}  "
        f"{'CLOUD:ON' if cloud.connected else 'CLOUD:OFF'}",
        (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    y = 58

    if mode == "SPORT" and locked_sport:
        cv2.putText(
            frame,
            f"Sport: {locked_sport.upper()} {locked_conf:.0%}",
            (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
        y += 28
        cv2.putText(
            frame,
            f"Players:{len(players)}  Avg:{avg_speed:.1f}km/h",
            (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,180), 1)
        y += 24
        if ref_msg:
            cv2.putText(frame, ref_msg[:60],
                        (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (0,255,255), 1)
        try:
            player_track.draw(frame, players, speeds)
        except Exception:
            pass
    else:
        status = uniform.get("status", "")
        if status == "UNAUTHORIZED":
            cv2.rectangle(frame, (0, y-18), (w, y+14), (0,0,200), -1)
            cv2.putText(
                frame,
                f"UNAUTHORIZED! {uniform.get('confidence',0):.0%}",
                (6, y+4), cv2.FONT_HERSHEY_SIMPLEX,
                0.68, (255,255,255), 2)
        elif status == "AUTHORIZED":
            cv2.putText(
                frame,
                f"AUTHORIZED {uniform.get('confidence',0):.0%}",
                (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (0,220,0), 2)
        else:
            cv2.putText(frame, "Safety monitoring...",
                        (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (160,160,160), 1)

        sc = sport.get("class", "unknown")
        sf = sport.get("confidence", 0.0)
        if (sc != "unknown" and sf > sport_candidate_conf
                and np.mean(cv2.cvtColor(
                    frame, cv2.COLOR_BGR2GRAY)) > 50):
            cv2.putText(
                frame,
                f"Detecting: {sc} ({sf:.0%})",
                (6, y+30), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (130,130,130), 1)

    if fall.get("detected"):
        cv2.rectangle(frame, (0, h-42), (w, h), (0,0,180), -1)
        cv2.putText(
            frame,
            f"FALL DETECTED! {fall.get('confidence',0):.0%}",
            (10, h-13), cv2.FONT_HERSHEY_SIMPLEX,
            0.72, (255,255,255), 2)
        if hasattr(fall_det, "draw_skeleton"):
            fall_det.draw_skeleton(frame, fall.get("landmarks"))

    if event_buf.is_recording:
        cv2.circle(frame, (w-20, 18), 8, (0,0,255), -1)
        cv2.putText(frame, "REC", (w-60, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        bar = int((w-20) * event_buf.progress)
        cv2.rectangle(frame, (10, h-6), (10+bar, h-2),
                      (0,0,255), -1)
    return frame


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="Smart Playground Monitor v5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.mount("/clips", StaticFiles(directory="saved_clips"), name="clips")


# ── MJPEG stream ──────────────────────────────────────────────

def mjpeg_gen():
    while True:
        jpeg = relay.get_jpeg() if is_active() else relay.get_paused()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + jpeg + b"\r\n")
        time.sleep(1.0 / 20)


@app.get("/stream")
def stream():
    return StreamingResponse(
        mjpeg_gen(),
        media_type="multipart/x-mixed-replace; boundary=frame")


# ── Stream control ────────────────────────────────────────────

@app.post("/api/stream/stop")
def api_stream_stop():
    global _detection_active
    with _det_lock:
        _detection_active = False
    event_buf.clear()
    print("[API] Detection STOPPED by user")
    return {"status": "stopped", "active": False}


@app.post("/api/stream/start")
def api_stream_start():
    global _detection_active
    with _det_lock:
        _detection_active = True
    print("[API] Detection STARTED by user")
    return {"status": "started", "active": True}


@app.get("/api/stream/status")
def api_stream_status():
    return {"active": is_active()}


# ── Core routes ───────────────────────────────────────────────

@app.get("/")
def index():
    p = Path("dashboard.html")
    if p.exists():
        return HTMLResponse(content=p.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>dashboard.html not found</h1>")


@app.get("/api/status")
def api_status():
    with state_lock:
        return {**state,
                "cloud_connected":  cloud.connected,
                "detection_active": is_active()}


@app.get("/api/clips/local")
def api_local_clips():
    return {"clips": clip_writer.get_all_local_clips()}


@app.get("/api/clips/cloud")
def api_cloud_clips():
    return {"events": cloud.get_recent_events(10)}


@app.get("/api/esp32")
def api_esp32():
    return esp32.get_info()


@app.get("/api/health")
def api_health():
    return {
        "status":           "ok",
        "cloud":            cloud.connected,
        "esp32":            esp32.connected,
        "detection_active": is_active(),
        "frame_count":      esp32.frame_count,
        "timestamp":        time.time()
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            with state_lock:
                data = {**state,
                        "type":             "update",
                        "cloud_connected":  cloud.connected,
                        "detection_active": is_active()}
            await ws.send_text(json.dumps(data, default=str))
            await asyncio.sleep(config.api.STATUS_WS_INTERVAL)
    except (WebSocketDisconnect, Exception):
        pass


# ── Tracker APIs ──────────────────────────────────────────────

@app.get("/api/tracker/stats")
def tracker_stats():
    return player_track.get_stats()

@app.get("/api/tracker/set_sport")
def tracker_set_sport(sport: str = Query(...)):
    player_track.set_sport(sport)
    return {"status": "ok", "sport": sport,
            "max_players": player_track.max_players,
            "description": player_track.description}

@app.get("/api/tracker/set_max")
def tracker_set_max(max: int = Query(...)):
    player_track.set_max_players(max)
    return {"status": "ok", "max_players": player_track.max_players}

@app.get("/api/tracker/set_team")
def tracker_set_team(player: int = Query(...),
                     team:   str = Query(...)):
    ok = player_track.set_player_team(player, team)
    return {"status": "ok" if ok else "error",
            "player": player, "team": team.upper()}

@app.get("/api/tracker/swap_team")
def tracker_swap_team(player: int = Query(...)):
    team = player_track.swap_player_team(player)
    return {"status": "ok", "player": player, "team": team}

@app.get("/api/tracker/team_mode")
def tracker_team_mode(mode: str = Query(...)):
    player_track.set_team_mode(mode)
    return {"status": "ok", "mode": mode.upper()}

@app.get("/api/tracker/split_mode")
def tracker_split_mode(mode: str = Query(...)):
    player_track.set_split_mode(mode)
    return {"status": "ok", "split_mode": mode}

@app.get("/api/tracker/clear_teams")
def tracker_clear_teams():
    player_track.clear_manual_teams()
    return {"status": "ok"}

@app.get("/api/tracker/reset")
def tracker_reset():
    player_track.reset()
    return {"status": "ok"}

@app.get("/api/tracker/full_reset")
def tracker_full_reset():
    player_track.full_reset()
    return {"status": "ok"}


# ── Uniform APIs ──────────────────────────────────────────────

@app.get("/api/uniform/threshold")
def uniform_threshold(val: float = Query(...)):
    uniform_det.set_threshold(val)
    return {"status": "ok", "threshold": val}

@app.get("/api/uniform/status")
def uniform_status_api():
    return {
        "available":   uniform_det.available,
        "threshold":   config.detection.UNIFORM_CONFIDENCE_THRESHOLD,
        "vote_window": config.detection.UNIFORM_VOTE_WINDOW,
        "cooldown":    uniform_det.cooldown
    }


# ── Fall APIs ─────────────────────────────────────────────────

@app.get("/api/fall/status")
def fall_status():
    return {
        "available":    getattr(fall_det, "available",    True),
        "current_mode": getattr(fall_det, "current_mode", "SAFETY"),
        "fall_active":  getattr(fall_det, "fall_active",  False),
        "counter":      getattr(fall_det, "fall_counter", 0)
    }

@app.get("/api/fall/thresholds")
def fall_thresholds(
    aspect_min:            float = Query(None),
    shoulder_hip_diff_max: float = Query(None),
    hip_y_min:             float = Query(None),
    orientation_min:       int   = Query(None),
    min_checks:            int   = Query(None),
    confirm:               int   = Query(None),
    reset:                 int   = Query(None),
    visibility:            float = Query(None),
    mode:                  str   = Query("SAFETY")
):
    if hasattr(fall_det, "set_thresholds"):
        kw = {}
        if aspect_min            is not None: kw["aspect_min"]            = aspect_min
        if shoulder_hip_diff_max is not None: kw["shoulder_hip_diff_max"] = shoulder_hip_diff_max
        if hip_y_min             is not None: kw["hip_y_min"]             = hip_y_min
        if orientation_min       is not None: kw["orientation_min"]       = orientation_min
        if min_checks            is not None: kw["min_checks"]            = min_checks
        if confirm               is not None: kw["confirm"]               = confirm
        if reset                 is not None: kw["reset"]                 = reset
        if visibility            is not None: kw["visibility"]            = visibility
        fall_det.set_thresholds(mode=mode, **kw)
    return {"status": "ok"}


# ── Camera APIs ───────────────────────────────────────────────

@app.get("/api/camera/control")
def camera_control(
    quality:  int = Query(None),
    bright:   int = Query(None),
    contrast: int = Query(None),
    sharp:    int = Query(None),
    flip:     int = Query(None),
    mirror:   int = Query(None)
):
    params = {}
    if quality  is not None: params["val"]      = quality
    if bright   is not None: params["bright"]   = bright
    if contrast is not None: params["contrast"] = contrast
    if sharp    is not None: params["sharp"]    = sharp
    if flip     is not None: params["flip"]     = flip
    if mirror   is not None: params["mirror"]   = mirror

    result = {}
    for key, val in params.items():
        try:
            urllib.request.urlopen(
                f"http://{esp32.ip}/quality?{key}={val}",
                timeout=3)
            result[key] = "ok"
        except Exception as e:
            result[key] = f"failed: {e}"
    return {"status": "ok", "cam_ip": esp32.ip, "results": result}


# ============================================================
# ENTRY POINT
# ============================================================

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


if __name__ == "__main__":
    ip = get_local_ip()

    print("\n" + "=" * 60)
    print(" SMART PLAYGROUND MONITOR v5.0 — RUNNING")
    print("=" * 60)
    print(f" ESP32     : {config.esp32.STREAM_URL}")
    print(f" Dashboard : http://{ip}:{config.api.PORT}")
    print(f" Stream    : http://{ip}:{config.api.PORT}/stream")
    print(f" API       : http://{ip}:{config.api.PORT}/api/status")
    print(f" Cloud     : {'Connected' if cloud.connected else 'Offline'}")
    print("=" * 60)
    print(" Smart Mode v8 Logic:")
    print("  1. Warmup 30f          → SAFETY only")
    print("  2. Fall detected       → SAFETY instant")
    print("  3. Sport 80%+ x 25f    → SPORT mode")
    print("     Motion>50 needs 88% → crowd rejected")
    print("  4. Same sport seen     → STAY locked")
    print("  5. New sport 75%+      → Switch 20f/10f fast")
    print("  6. No sport 10f        → SAFETY")
    print("  7. Stop button         → Pause + no clips")
    print("=" * 60)

    esp32.start()

    threading.Thread(
        target=ai_loop,
        daemon=True,
        name="AILoop"
    ).start()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.api.PORT,
        log_level="warning",
        access_log=False
    )
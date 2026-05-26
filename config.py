# config.py
# ============================================================
# MASTER CONFIGURATION FILE
# Final version for Smart Playground Monitor v5.0
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────
# ESP32-CAM SETTINGS
# ─────────────────────────────────────────────────────────────
class ESP32Config:
    IP_ADDRESS = os.getenv("ESP32_IP", "192.168.1.45")

    TARGET_FPS      = 15
    RECONNECT_DELAY = 3
    MAX_RETRIES     = 10
    READ_TIMEOUT    = 10
    CHUNK_SIZE      = 4096

    FRAME_WIDTH  = 640
    FRAME_HEIGHT = 480
    ESP32_INTERVAL = 10

    @property
    def STREAM_URL(self):
        return f"http://{self.IP_ADDRESS}/stream"

    @property
    def SNAPSHOT_URL(self):
        return f"http://{self.IP_ADDRESS}/capture"

    @property
    def STATUS_URL(self):
        return f"http://{self.IP_ADDRESS}/status"


# ─────────────────────────────────────────────────────────────
# MODEL SETTINGS
# ─────────────────────────────────────────────────────────────
class ModelConfig:
    SPORTS_MODEL_PATH  = "models/sports_v2.tflite"
    UNIFORM_MODEL_PATH = "models/uniform_detection_final.tflite"
    YOLO_MODEL_PATH    = "yolov8n.pt"

    SPORT_CLASSES   = ["badminton", "basketball", "cricket", "football"]
    UNIFORM_CLASSES = ["authorized", "unauthorized"]


# ─────────────────────────────────────────────────────────────
# DETECTION SETTINGS
# ─────────────────────────────────────────────────────────────
class DetectionConfig:

    # ── Motion thresholds ─────────────────────────────────────
    MOTION_SPORT_THRESHOLD = 4.5
    MOTION_IDLE_THRESHOLD  = 2.0

    # ── Sport mode entry / exit ───────────────────────────────
    SPORT_ENTER_CONF     = 0.65   # Min confidence to start counting toward SPORT
    SPORT_EXIT_CONF      = 0.30   # Below this = sport considered "gone"
    SPORT_CANDIDATE_CONF = 0.35   # Min confidence to show candidate on screen
    SPORT_CONF_REFEREE   = 0.65   # Min confidence to activate referee system

    SPORT_ENTER_FRAMES = 15       # Consecutive frames needed to enter SPORT
    SPORT_EXIT_FRAMES  = 20       # Frames of no-sport before → SAFETY

    # ── Sport lock (prevents instant sport switching) ─────────
    SPORT_LOCK_CONF       = 0.75  # New sport needs >= this to compete
    SPORT_LOCK_MIN_FRAMES = 20    # Must be in current sport this many frames first
    SPORT_SWITCH_FRAMES   = 15    # Consecutive frames new sport must appear (normal)
    SPORT_SWITCH_FAST_CONF   = 0.90  # Fast-track confidence threshold
    SPORT_SWITCH_FAST_FRAMES = 8     # Fast-track frame count

    # ── Mode flags ────────────────────────────────────────────
    FALL_FORCE_SAFETY            = True
    UNIFORM_REQUIRES_LOW_MOTION  = True

    # ── Fall detector thresholds v7 ───────────────────────────
    FALL_SAFETY_THRESHOLDS = {
        "aspect_min":           0.75,
        "shoulder_hip_diff_max": 0.07,
        "hip_y_min":            0.50,
        "orientation_min":      65,
        "min_checks":           3,
        "confirm":              5,
        "reset":                20,
        "visibility":           0.45,
    }
    FALL_SPORT_THRESHOLDS = {
        "aspect_min":           0.90,
        "shoulder_hip_diff_max": 0.06,
        "hip_y_min":            0.55,
        "orientation_min":      72,
        "min_checks":           3,
        "confirm":              8,
        "reset":                12,
        "visibility":           0.50,
    }

    # ── Player tracking ───────────────────────────────────────
    MAX_PLAYERS_GLOBAL       = 22
    PLAYER_TRACK_MAX_DIST    = 150
    PLAYER_LOST_FRAMES       = 20
    PLAYER_YOLO_CONF_THRESHOLD = 0.40
    PLAYER_MIN_AREA          = 800
    PLAYER_MAX_AREA          = 120000
    PLAYER_MIN_WIDTH         = 20
    PLAYER_MIN_HEIGHT        = 40
    PLAYER_MAX_ASPECT        = 4.0
    PLAYER_IOU_THRESHOLD     = 0.3
    PLAYER_CONF_THRESHOLD    = 0.40

    # ── Sport player limits (hard constraints) ────────────────
    SPORT_PLAYER_LIMITS = {
        "badminton": {
            "singles": 2,
            "doubles": 4,
            "default": 4,
        },
        "basketball": {
            "default":    10,
            "half_court": 6,
        },
        "cricket": {
            "default": 13,
        },
        "football": {
            "default":     22,
            "small_sided": 10,
        },
    }

    # ── Speed estimation ──────────────────────────────────────
    PIXELS_PER_METER   = 50
    MAX_SPEED_MPS      = 12.0
    SPEED_SMOOTH_WINDOW = 7

    # ── Detection voting windows ──────────────────────────────
    SPORT_VOTE_WINDOW   = 10
    UNIFORM_VOTE_WINDOW = 12

    # ── Uniform detection ─────────────────────────────────────
    UNIFORM_CONFIDENCE_THRESHOLD = 0.50
    UNIFORM_ALERT_COOLDOWN       = 45


# ─────────────────────────────────────────────────────────────
# EVENT BUFFER SETTINGS
# ─────────────────────────────────────────────────────────────
class BufferConfig:
    PRE_EVENT_SECONDS  = 3
    POST_EVENT_SECONDS = 2
    FPS                = 15
    SAVE_DIR           = "saved_clips"
    MAX_CLIPS_STORED   = 50


# ─────────────────────────────────────────────────────────────
# CLOUD (SUPABASE) SETTINGS
# ─────────────────────────────────────────────────────────────
class CloudConfig:
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

    LIVE_STATUS_TABLE  = "live_status"
    EVENTS_TABLE       = "events"
    ESP32_STATUS_TABLE = "esp32_status"
    VIDEO_BUCKET       = "event-clips"

    PLAYGROUND_DOC_ID    = "playground_1"
    ESP32_DOC_ID         = "esp32_1"
    EVENT_RETENTION_DAYS = 30


# ─────────────────────────────────────────────────────────────
# API / WEB SERVER SETTINGS
# ─────────────────────────────────────────────────────────────
class APIConfig:
    HOST = "0.0.0.0"
    PORT = int(os.getenv("SERVER_PORT", 8000))

    STREAM_QUALITY      = 85
    STATUS_WS_INTERVAL  = 0.5


# ─────────────────────────────────────────────────────────────
# REFEREE / SPORTS ANALYTICS SETTINGS
# ─────────────────────────────────────────────────────────────
class RefereeConfig:
    EVENT_COOLDOWN_SEC = 3.0

    FOOTBALL_FAST_BREAK_KMH  = 12.0
    FOOTBALL_ACTIVE_KMH      = 7.0

    CRICKET_FAST_RUN_KMH     = 10.0
    CRICKET_ACTIVE_KMH       = 6.0

    BASKETBALL_FAST_DRIVE_KMH   = 10.0
    BASKETBALL_TRANSITION_KMH   = 6.0

    BADMINTON_QUICK_EXCHANGE_KMH = 8.0

    MAX_ANALYTIC_PLAYERS = 22


# ─────────────────────────────────────────────────────────────
# MASTER CONFIG OBJECT
# ─────────────────────────────────────────────────────────────
class Config:
    # Supabase shortcuts (top-level for legacy access)
    SUPABASE_URL         = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY         = os.getenv("SUPABASE_ANON_KEY", "")
    SUPABASE_ANON_KEY    = os.getenv("SUPABASE_ANON_KEY", "")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

    LIVE_STATUS_TABLE  = "live_status"
    EVENTS_TABLE       = "events"
    ESP32_STATUS_TABLE = "esp32_status"
    VIDEO_BUCKET       = "event-clips"

    PLAYGROUND_ID        = "playground_1"
    EVENT_RETENTION_DAYS = 30

    # Sub-config instances
    esp32     = ESP32Config()
    model     = ModelConfig()
    detection = DetectionConfig()
    buffer    = BufferConfig()
    cloud     = CloudConfig()
    api       = APIConfig()
    referee   = RefereeConfig()

    # Debug / display flags
    SHOW_LOCAL_WINDOW = False
    LOG_DETECTIONS    = True


# ─────────────────────────────────────────────────────────────
# STARTUP VALIDATION — catches missing attributes immediately
# ─────────────────────────────────────────────────────────────
def _validate_config():
    required = [
        # Sport switching
        "SPORT_ENTER_CONF",
        "SPORT_EXIT_CONF",
        "SPORT_CANDIDATE_CONF",
        "SPORT_CONF_REFEREE",
        "SPORT_ENTER_FRAMES",
        "SPORT_EXIT_FRAMES",
        "SPORT_LOCK_CONF",
        "SPORT_LOCK_MIN_FRAMES",
        "SPORT_SWITCH_FRAMES",
        "SPORT_SWITCH_FAST_CONF",
        "SPORT_SWITCH_FAST_FRAMES",
        # Fall detection
        "FALL_FORCE_SAFETY",
        "FALL_SAFETY_THRESHOLDS",
        "FALL_SPORT_THRESHOLDS",
        # Player tracking
        "MAX_PLAYERS_GLOBAL",
        "PLAYER_MIN_AREA",
        "PLAYER_MAX_AREA",
        "PLAYER_MIN_WIDTH",
        "PLAYER_MIN_HEIGHT",
        "PLAYER_MAX_ASPECT",
        "PLAYER_IOU_THRESHOLD",
        "PLAYER_TRACK_MAX_DIST",
        "PLAYER_LOST_FRAMES",
        # Speed
        "PIXELS_PER_METER",
        "MAX_SPEED_MPS",
        "SPEED_SMOOTH_WINDOW",
        # Uniform
        "UNIFORM_CONFIDENCE_THRESHOLD",
        "UNIFORM_ALERT_COOLDOWN",
        "UNIFORM_VOTE_WINDOW",
        # Motion
        "MOTION_SPORT_THRESHOLD",
        "MOTION_IDLE_THRESHOLD",
    ]

    det     = DetectionConfig()
    missing = [a for a in required if not hasattr(det, a)]

    if missing:
        raise AttributeError(
            f"\n[Config] MISSING DetectionConfig attributes:\n"
            + "\n".join(f"  - {a}" for a in missing)
            + "\nAdd them to DetectionConfig in config.py before running."
        )

    print(f"[Config] OK — all {len(required)} required attributes present")


_validate_config()

config = Config()
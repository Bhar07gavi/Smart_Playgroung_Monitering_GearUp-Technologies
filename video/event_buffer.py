# video/event_buffer.py
# ============================================================
import threading
import time
from collections import deque
from config import config


class EventBuffer:

    def __init__(self):
        fps      = config.buffer.FPS
        pre_sec  = config.buffer.PRE_EVENT_SECONDS
        post_sec = config.buffer.POST_EVENT_SECONDS

        self._pre_frames  = int(pre_sec  * fps)
        self._post_frames = int(post_sec * fps)
        self._fps         = fps

        self._pre_buf  = deque(maxlen=self._pre_frames)
        self._post_buf = []

        self._recording    = False
        self._event_type   = None
        self._event_meta   = {}
        self._post_count   = 0
        self._lock         = threading.Lock()

        self.is_recording  = False
        self.progress      = 0.0

        print(f"[EventBuffer] Ready. "
              f"Pre-event:{self._pre_frames}f, "
              f"Post-event:{self._post_frames}f")

    def add_frame(self, frame) -> dict | None:
        """
        Add a frame to the buffer.
        Returns a completed clip dict when post-event is done,
        else None.
        """
        with self._lock:
            if not self._recording:
                self._pre_buf.append(frame)
                self.is_recording = False
                self.progress     = 0.0
                return None

            # Recording post-event frames
            self._post_buf.append(frame)
            self._post_count += 1
            self.progress     = min(
                self._post_count / self._post_frames, 1.0)
            self.is_recording = True

            if self._post_count >= self._post_frames:
                # Clip complete
                clip = {
                    "frames":     list(self._pre_buf) + self._post_buf,
                    "event_type": self._event_type,
                    "meta":       self._event_meta,
                    "fps":        self._fps,
                    "sport":      self._event_meta.get("sport", ""),
                    "timestamp":  time.time(),
                }
                # Reset
                self._recording  = False
                self._post_buf   = []
                self._post_count = 0
                self._event_type = None
                self._event_meta = {}
                self.is_recording = False
                self.progress     = 0.0
                return clip

            return None

    def trigger(self, event_type: str, meta: dict = None):
        """Trigger a recording event."""
        with self._lock:
            if self._recording:
                return  # Already recording
            self._recording  = True
            self._event_type = event_type
            self._event_meta = meta or {}
            self._post_buf   = []
            self._post_count = 0
            self.is_recording = True
            self.progress     = 0.0
        print(f"[EventBuffer] 🔴 Recording triggered for: {event_type}")

    def clear(self):
        """Clear all buffers — called on mode switch."""
        with self._lock:
            self._pre_buf.clear()
            self._post_buf   = []
            self._post_count = 0
            self._recording  = False
            self._event_type = None
            self._event_meta = {}
            self.is_recording = False
            self.progress     = 0.0

    @property
    def is_triggered(self) -> bool:
        return self._recording
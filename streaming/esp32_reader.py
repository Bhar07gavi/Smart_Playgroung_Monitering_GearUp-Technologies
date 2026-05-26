# streaming/esp32_reader.py
# ============================================================
# Reads MJPEG stream from ESP32-CAM continuously
# Handles disconnections and auto-reconnects
# Provides latest frame to main pipeline via get_frame()
# ============================================================

import cv2
import numpy as np
import threading
import time
import urllib.request
from config import config


class ESP32StreamReader:
    """
    Background thread reads MJPEG stream from ESP32-CAM.
    Main thread calls get_frame() anytime — always gets latest.

    Thread safety: uses Lock around frame storage.
    """

    def __init__(self):
        self.stream_url    = config.esp32.STREAM_URL
        self.reconnect_sec = config.esp32.RECONNECT_DELAY
        self.max_retries   = config.esp32.MAX_RETRIES
        self.timeout       = config.esp32.READ_TIMEOUT
        self.chunk_size    = config.esp32.CHUNK_SIZE

        # Shared latest frame
        self._frame      = None
        self._frame_lock = threading.Lock()

        # State
        self._running    = False
        self._connected  = False
        self._thread     = None
        self._retry_count = 0

        # Stats
        self.frame_count = 0
        self.fps_actual  = 0.0
        self._fps_frames = 0
        self._fps_timer  = time.time()

    # ── Public interface ───────────────────────────────────────

    def start(self):
        """Start background stream reader thread."""
        self._running = True
        self._thread  = threading.Thread(
            target    = self._read_loop,
            daemon    = True,
            name      = "ESP32Reader"
        )
        self._thread.start()
        print(f"[ESP32Reader] Started | URL: {self.stream_url}")

    def stop(self):
        """Stop reader thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[ESP32Reader] Stopped")

    def get_frame(self):
        """
        Get latest frame. Non-blocking.
        Returns None if no frame received yet.
        """
        with self._frame_lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    @property
    def is_connected(self):
        return self._connected

    # ── Background thread ──────────────────────────────────────

    def _read_loop(self):
        """Outer loop — reconnects on failure."""
        while self._running:
            try:
                self._connect_and_read()
            except Exception as e:
                self._connected   = False
                self._retry_count += 1

                if self._retry_count >= self.max_retries:
                    print(f"[ESP32Reader] Max retries reached. Resetting counter.")
                    self._retry_count = 0

                wait = self.reconnect_sec * min(self._retry_count, 5)
                print(f"[ESP32Reader] Disconnected: {e}")
                print(f"  Retry {self._retry_count} in {wait}s...")
                time.sleep(wait)

    def _connect_and_read(self):
        """Connect to MJPEG stream and decode frames until error."""
        print(f"[ESP32Reader] Connecting to {self.stream_url} ...")

        stream = urllib.request.urlopen(
            self.stream_url, timeout=self.timeout
        )

        self._connected   = True
        self._retry_count = 0
        print("[ESP32Reader] Connected!")

        # MJPEG uses JPEG magic bytes as frame delimiters
        JPEG_SOI = b'\xff\xd8'   # Start of Image
        JPEG_EOI = b'\xff\xd9'   # End of Image

        buf = bytes()

        while self._running:
            chunk = stream.read(self.chunk_size)
            if not chunk:
                raise ConnectionError("Stream returned empty chunk")

            buf += chunk

            while True:
                start = buf.find(JPEG_SOI)
                end   = buf.find(JPEG_EOI)

                if start == -1 or end == -1:
                    break   # Need more data

                if end < start:
                    # Corrupt — skip to next SOI
                    buf = buf[start + 2:]
                    continue

                # Extract one complete JPEG
                jpg  = buf[start : end + 2]
                buf  = buf[end + 2:]

                frame = self._decode_jpeg(jpg)
                if frame is not None:
                    with self._frame_lock:
                        self._frame = frame
                    self.frame_count += 1
                    self._update_fps()

    def _decode_jpeg(self, jpg_bytes):
        """Decode JPEG bytes to BGR frame."""
        try:
            arr   = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    def _update_fps(self):
        """Update fps_actual every 2 seconds."""
        self._fps_frames += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 2.0:
            self.fps_actual  = round(self._fps_frames / elapsed, 1)
            self._fps_frames = 0
            self._fps_timer  = time.time()
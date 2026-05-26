# streaming/stream_relay.py
import cv2
import numpy as np
import threading
import time

from config import config # Import config

class StreamRelay:
    def __init__(self):
        self._jpeg = None
        self._lock = threading.Lock()

        # Placeholder image for when no frame is available
        placeholder = np.zeros(
            (config.esp32.FRAME_HEIGHT, config.esp32.FRAME_WIDTH, 3),
            dtype=np.uint8
        )
        cv2.putText(
            placeholder,
            "Waiting for ESP32-CAM...",
            (
                int(config.esp32.FRAME_WIDTH * 0.15),
                int(config.esp32.FRAME_HEIGHT * 0.45)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 180, 255),
            2
        )

        _, buf = cv2.imencode(".jpg", placeholder)
        self._placeholder = buf.tobytes()

    def update(self, frame):
        # Resize frame for web streaming if it's not the default size
        if frame.shape[0] != config.esp32.FRAME_HEIGHT or \
           frame.shape[1] != config.esp32.FRAME_WIDTH:
            frame = cv2.resize(frame, (config.esp32.FRAME_WIDTH, config.esp32.FRAME_HEIGHT))
            
        ok, buf = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, config.api.STREAM_QUALITY]
        )

        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()

    def get_jpeg(self):
        with self._lock:
            return self._jpeg if self._jpeg else self._placeholder
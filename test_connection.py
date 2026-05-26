# detectors/sport_detector.py

import os
import numpy as np
import cv2
from collections import deque

SPORTS_MODEL  = "models/sports_v2.tflite"
SPORT_CLASSES = ["badminton", "basketball", "cricket", "football"]


class SportDetector:
    """
    Classifies sport using your trained TFLite model.
    Uses voting window to prevent flickering.
    """

    def __init__(self):
        self.available = False
        self.votes     = deque(maxlen=10)

        if not os.path.exists(SPORTS_MODEL):
            print(f"[SportDetector] Model not found: {SPORTS_MODEL}")
            return

        try:
            import tensorflow as tf
            self.interp = tf.lite.Interpreter(model_path=SPORTS_MODEL)
            self.interp.allocate_tensors()
            self.inp    = self.interp.get_input_details()
            self.out    = self.interp.get_output_details()
            s           = self.inp[0]['shape']
            self.ih     = s[1]
            self.iw     = s[2]
            self.available = True
            print(f"[SportDetector] Loaded | {self.iw}x{self.ih}")
        except Exception as e:
            print(f"[SportDetector] Load error: {e}")

    def predict(self, frame):
        """
        Returns:
        {
            class      : "cricket" or "unknown",
            confidence : float,
            stable     : bool
        }
        """
        empty = {"class": "unknown", "confidence": 0.0, "stable": False}

        if not self.available:
            return empty

        img = cv2.resize(frame, (self.iw, self.ih))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img.astype(np.float32) / 255.0, 0)

        self.interp.set_tensor(self.inp[0]['index'], img)
        self.interp.invoke()
        raw = self.interp.get_tensor(self.out[0]['index'])[0]

        idx  = int(np.argmax(raw))
        conf = float(raw[idx])

        if conf < 0.60:
            return empty

        self.votes.append(idx)
        counts = {}
        for v in self.votes:
            counts[v] = counts.get(v, 0) + 1
        winner = max(counts, key=counts.get)
        stable = (
            len(self.votes) == 10
            and counts[winner] / 10 > 0.6
        )

        cls = SPORT_CLASSES[winner] \
              if winner < len(SPORT_CLASSES) else "unknown"

        return {
            "class"     : cls,
            "confidence": float(raw[winner]),
            "stable"    : stable
        }

    def reset(self):
        self.votes.clear()
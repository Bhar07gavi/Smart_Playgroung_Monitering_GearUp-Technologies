# detectors/uniform_detector.py
import os
import cv2
import numpy as np
import tensorflow as tf
from collections import deque

from config import config # Import config within the module

class UniformDetector:
    ALERT_COOLDOWN = config.detection.UNIFORM_ALERT_COOLDOWN

    def __init__(self):
        self.available   = False
        self.interp      = None
        self.votes       = deque(
            maxlen=config.detection.UNIFORM_VOTE_WINDOW)
        self.cooldown    = 0
        self.frame_count = 0
        self.classes     = config.model.UNIFORM_CLASSES

        if not os.path.exists(config.model.UNIFORM_MODEL_PATH):
            if config.LOG_DETECTIONS:
                print(f"[UniformDetector] Model not found: "
                      f"{config.model.UNIFORM_MODEL_PATH}")
            return

        if config.LOG_DETECTIONS:
            print(f"[UniformDetector] Loading "
                  f"{config.model.UNIFORM_MODEL_PATH}...")
        try:
            from ai_edge_litert.interpreter import Interpreter
            self.interp = Interpreter(
                model_path=config.model.UNIFORM_MODEL_PATH)
        except ImportError:
            import tensorflow as tf
            self.interp = tf.lite.Interpreter(
                model_path=config.model.UNIFORM_MODEL_PATH)

        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()
        self.out = self.interp.get_output_details()
        self.ih  = int(self.inp[0]['shape'][1])
        self.iw  = int(self.inp[0]['shape'][2])
        self.available = True

        if config.LOG_DETECTIONS:
            print(f"[UniformDetector] OK | {self.iw}x{self.ih}")
            print(f"  Threshold    : "
                  f"{config.detection.UNIFORM_CONFIDENCE_THRESHOLD:.0%}")
            print(f"  Vote window  : "
                  f"{config.detection.UNIFORM_VOTE_WINDOW}")

    def _preprocess(self, frame):
        import cv2
        img   = cv2.resize(frame, (self.iw, self.ih))
        img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img   = img.astype(np.float32) / 255.0
        img   = np.expand_dims(img, axis=0)
        dtype = self.inp[0]["dtype"]
        if dtype == np.float32:
            return img
        scale, zero = self.inp[0]["quantization"]
        return np.clip(img / scale + zero, 0, 255).astype(dtype)

    def _dequantize(self, output):
        if self.out[0]["dtype"] == np.float32:
            return output
        scale, zero = self.out[0]["quantization"]
        return (output.astype(np.float32) - zero) * scale

    def predict(self, frame) -> dict:
        self.frame_count += 1

        if not self.available:
            return {"status": "NO_MODEL", "confidence": 0.0,
                    "stable": False, "alert": False, "scores": {}}

        if self.cooldown > 0:
            self.cooldown -= 1

        try:
            img = self._preprocess(frame)
            self.interp.set_tensor(self.inp[0]['index'], img)
            self.interp.invoke()
            raw  = self.interp.get_tensor(
                self.out[0]['index']).flatten()
            unauth_score = float(self._dequantize(raw)[0])
            auth_score   = 1.0 - unauth_score

        except Exception as e:
            return self._error_result(f"Inference: {e}")

        # ── FIX: Use lower threshold for voting ───────────────
        # Was 0.50 → now 0.42 to catch more unauthorized cases
        # The alert still needs +0.05 above threshold
        VOTE_THRESHOLD = 0.42

        is_unauth_vote = unauth_score >= VOTE_THRESHOLD
        self.votes.append(1 if is_unauth_vote else 0)

        unauth_votes = sum(self.votes)
        auth_votes   = len(self.votes) - unauth_votes
        is_unauth    = unauth_votes > auth_votes

        # Stability: 80% of vote window filled + 65%+ majority
        min_votes = int(config.detection.UNIFORM_VOTE_WINDOW * 0.8)
        stable    = False
        if len(self.votes) >= min_votes:
            ratio  = unauth_votes / len(self.votes)
            stable = ratio > 0.65 or ratio < 0.35

        if is_unauth:
            status     = "UNAUTHORIZED"
            confidence = unauth_score
        else:
            status     = "AUTHORIZED"
            confidence = auth_score

        # Alert: unauthorized + stable + cooldown done + strong score
        alert = (
            status == "UNAUTHORIZED"
            and stable
            and self.cooldown == 0
            and unauth_score >= VOTE_THRESHOLD + 0.05
        )

        if alert:
            self.cooldown = self.ALERT_COOLDOWN
            if config.LOG_DETECTIONS:
                print(f"[Uniform] ⚠️ UNAUTHORIZED! "
                      f"{unauth_score:.1%}")

        if config.LOG_DETECTIONS and self.frame_count % 30 == 0:
            print(f"[Uniform] auth:{auth_score:.3f} "
                  f"unauth:{unauth_score:.3f} "
                  f"votes:{unauth_votes}/{len(self.votes)} "
                  f"→ {status}")

        return {
            "status":     status,
            "confidence": confidence,
            "stable":     stable,
            "alert":      alert,
            "scores": {
                "authorized":   round(auth_score,   4),
                "unauthorized": round(unauth_score, 4)
            }
        }

    def _error_result(self, msg):
        if config.LOG_DETECTIONS:
            print(f"[UniformDetector] Error: {msg}")
        return {"status": "ERROR", "confidence": 0.0,
                "stable": False, "alert": False, "scores": {}}

    def set_threshold(self, value):
        config.detection.UNIFORM_CONFIDENCE_THRESHOLD = max(
            0.01, min(0.95, float(value)))
        self.votes.clear()
        if config.LOG_DETECTIONS:
            print(f"[Uniform] Threshold → "
                  f"{config.detection.UNIFORM_CONFIDENCE_THRESHOLD:.2f}")

    def set_swap(self, swap):
        if config.LOG_DETECTIONS:
            print("[Uniform] Swap ignored: sigmoid model.")

    def reset(self):
        self.votes.clear()
        self.cooldown    = 0
        self.frame_count = 0
        if config.LOG_DETECTIONS:
            print("[UniformDetector] Reset")
    
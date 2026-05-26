# detectors/uniform_detector.py
# detectors/sport_detector.py
import os
import cv2
import numpy as np
import tensorflow as tf
from collections import deque

from config import config # Import config within the module

class SportDetector:
    def __init__(self):
        self.available = False
        self.interp = None
        # FIX: Use config.detection.SPORT_VOTE_WINDOW
        self.votes = deque(maxlen=config.detection.SPORT_VOTE_WINDOW) 
        self.frame_count = 0
        
        self.classes = config.model.SPORT_CLASSES

        if not os.path.exists(config.model.SPORTS_MODEL_PATH):
            if config.LOG_DETECTIONS: print(f"[SportDetector] Model not found: {config.model.SPORTS_MODEL_PATH}")
            return

        if config.LOG_DETECTIONS: print(f"[SportDetector] Loading {config.model.SPORTS_MODEL_PATH}...")
        try:
            # Try to load with ai_edge_litert first (for performance)
            from ai_edge_litert.interpreter import Interpreter
            self.interp = Interpreter(model_path=config.model.SPORTS_MODEL_PATH)
        except ImportError:
            # Fallback to TensorFlow Lite Interpreter
            self.interp = tf.lite.Interpreter(model_path=config.model.SPORTS_MODEL_PATH)
        
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()
        self.out = self.interp.get_output_details()

        self.ih = int(self.inp[0]['shape'][1])
        self.iw = int(self.inp[0]['shape'][2])
        self.available = True

        if config.LOG_DETECTIONS:
            print(f"[SportDetector] OK | {self.iw}x{self.ih}")
            print(f"  Classes      : {self.classes}")
            # Correctly get number of outputs if self.nc is not defined before this.
            # Assuming model output corresponds to number of classes.
            print(f"  Num outputs  : {len(self.classes)}") 

    def _preprocess(self, frame):
        img = cv2.resize(frame, (self.ih, self.iw))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def _dequantize_output(self, output):
        if self.out[0]["dtype"] == np.float32:
            return output.astype(np.float32)
        scale, zero = self.out[0]["quantization"]
        if scale > 0:
            return (output.astype(np.float32) - zero) * scale
        return output.astype(np.float32)

    def predict(self, frame):
        self.frame_count += 1
        if not self.available:
            return {"class": "NO_MODEL", "confidence": 0.0, "scores": {}}

        # Preprocess
        try:
            img = self._preprocess(frame)
        except Exception as e:
            return self._error_result(f"Preprocess: {e}")

        # Inference
        try:
            self.interp.set_tensor(self.inp[0]['index'], img)
            self.interp.invoke()
            raw_scores = self.interp.get_tensor(self.out[0]['index'])[0]
            scores = self._dequantize_output(raw_scores)
        except Exception as e:
            return self._error_result(f"Inference: {e}")

        # Voting for stability
        self.votes.append(scores)
        smoothed_scores = np.mean(list(self.votes), axis=0)

        idx = int(np.argmax(smoothed_scores))
        conf = float(smoothed_scores[idx])

        # Build scores dict
        result_scores = {}
        for i, cls_name in enumerate(self.classes):
            result_scores[cls_name] = round(float(smoothed_scores[i]), 4)

        if config.LOG_DETECTIONS and self.frame_count % 30 == 0:
            print(f"[Sport] {self.classes[idx]}:{conf:.1%} raw:{np.max(raw_scores):.1%}")

        return {
            "class": self.classes[idx],
            "confidence": conf,
            "scores": result_scores
        }

    def _error_result(self, msg):
        if config.LOG_DETECTIONS: print(f"[SportDetector] Error: {msg}")
        return {"class": "ERROR", "confidence": 0.0, "scores": {}}

    def reset(self):
        self.votes.clear()
        self.frame_count = 0
        if config.LOG_DETECTIONS: print("[SportDetector] Reset")
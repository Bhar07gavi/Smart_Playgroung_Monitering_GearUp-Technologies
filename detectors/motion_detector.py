# detectors/motion_detector.py
import cv2
import numpy as np
from collections import deque

from config import config # Import config within the module

class MotionDetector:
    def __init__(self):
        # Use config value for smooth_window
        self.smooth_window = int(config.buffer.FPS * 1.5) # Smooth motion over 1.5 seconds

        self.prev_gray = None
        self.motion_history = deque(maxlen=self.smooth_window)

    def calculate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0

        diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_score = np.mean(thresh)

        self.prev_gray = gray # Update for next frame
        return motion_score

    def reset(self):
        self.prev_gray = None
        self.motion_history.clear()
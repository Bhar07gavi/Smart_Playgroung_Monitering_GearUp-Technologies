# tracking/speed_estimator.py
# ============================================================
# Speed estimation using tracked centroids
# - Smoothed speed
# - Handles temporary disappearance
# - Use calibration (pixels_per_meter) for real speed
# ============================================================

import numpy as np
import time
from collections import deque


class SpeedEstimator:
    def __init__(self, pixels_per_meter=50.0):
        self.ppm = float(pixels_per_meter)

        self.SMOOTH_WINDOW = 7
        self.MAX_SPEED_MPS = 12.0
        self.MAX_GAP = 1.0

        # pid -> (cx, cy, timestamp)
        self.prev_pos = {}

        # pid -> deque of speeds
        self.speed_hist = {}

    def set_calibration(self, pixels_per_meter):
        self.ppm = max(1.0, float(pixels_per_meter))

    def update(self, players):
        """
        players: list of dicts from PlayerTracker.detect()
        returns:
            {
              pid: {
                 "speed_mps": ...,
                 "speed_kmh": ...,
                 "direction": (dx, dy)
              }
            }
        """
        now = time.time()
        speeds = {}

        active_ids = set()

        for p in players:
            pid = p["id"]
            cx, cy = p["center"]
            active_ids.add(pid)

            if pid in self.prev_pos:
                px, py, pt = self.prev_pos[pid]
                dt = now - pt

                if 0.03 < dt <= self.MAX_GAP:
                    pixel_dist = float(np.hypot(cx - px, cy - py))
                    metre_dist = pixel_dist / self.ppm

                    speed_mps = metre_dist / dt
                    speed_mps = min(speed_mps, self.MAX_SPEED_MPS)

                    if pid not in self.speed_hist:
                        self.speed_hist[pid] = deque(maxlen=self.SMOOTH_WINDOW)

                    self.speed_hist[pid].append(speed_mps)
                    smooth_mps = float(np.mean(self.speed_hist[pid]))

                    if pixel_dist > 0:
                        direction = (
                            round((cx - px) / pixel_dist, 2),
                            round((cy - py) / pixel_dist, 2)
                        )
                    else:
                        direction = (0.0, 0.0)

                    speeds[pid] = {
                        "speed_mps": round(smooth_mps, 2),
                        "speed_kmh": round(smooth_mps * 3.6, 1),
                        "direction": direction
                    }

            self.prev_pos[pid] = (cx, cy, now)

        # cleanup old IDs
        for pid in list(self.prev_pos.keys()):
            if pid not in active_ids:
                _, _, pt = self.prev_pos[pid]
                if now - pt > self.MAX_GAP:
                    del self.prev_pos[pid]
                    self.speed_hist.pop(pid, None)

        return speeds

    def avg_speed(self, speeds):
        if not speeds:
            return 0.0
        vals = [s["speed_kmh"] for s in speeds.values()]
        return round(float(np.mean(vals)), 1)

    def fastest(self, speeds):
        if not speeds:
            return None, 0.0
        pid = max(speeds, key=lambda k: speeds[k]["speed_mps"])
        return pid, speeds[pid]["speed_mps"]

    def reset(self):
        self.prev_pos = {}
        self.speed_hist = {}
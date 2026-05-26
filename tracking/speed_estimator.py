# tracking/speed_estimator.py
import time
import numpy as np
from collections import deque

from config import config # Import config

class SpeedEstimator:
    def __init__(self, pixels_per_meter=config.detection.PIXELS_PER_METER):
        self.ppm = float(pixels_per_meter)
        self.fps = config.buffer.FPS # Use configured FPS
        self.smooth_window = config.detection.SPEED_SMOOTH_WINDOW
        self.max_speed_mps = config.detection.MAX_SPEED_MPS
        self.max_gap_seconds = config.detection.PLAYER_LOST_FRAMES / self.fps # Based on lost frames

        self.prev_pos = {}
        self.speed_hist = {}

        print("[SpeedEstimator] Ready")
        print(f"  pixels_per_meter: {self.ppm}")
        print(f"  FPS fallback    : {self.fps}")

    def update(self, players):
        now = time.time()
        speeds = {}
        active_ids = set()

        for p in players:
            if "id" not in p or "center" not in p: continue
            pid = p["id"]; cx, cy = p["center"]
            active_ids.add(pid)

            if pid in self.prev_pos:
                px, py, pt = self.prev_pos[pid]
                dt = now - pt
                if 0.02 < dt < self.max_gap_seconds: # Filter very small/large time deltas
                    pixel_dist = float(np.hypot(cx - px, cy - py))
                    meter_dist = pixel_dist / self.ppm
                    speed_mps = min(meter_dist / dt, self.max_speed_mps)
                    
                    if pid not in self.speed_hist: self.speed_hist[pid] = deque(maxlen=self.smooth_window)
                    self.speed_hist[pid].append(speed_mps)
                    smooth_mps = float(np.mean(self.speed_hist[pid]))

                    dx, dy = (round((cx - px) / pixel_dist, 3), round((cy - py) / pixel_dist, 3)) if pixel_dist > 0 else (0.0, 0.0)

                    speeds[pid] = {
                        "speed_mps": round(smooth_mps, 2),
                        "speed_kmh": round(smooth_mps * 3.6, 1),
                        "direction": (dx, dy),
                        "moving": smooth_mps > 0.25 # Player is considered moving above 0.25 m/s
                    }
            self.prev_pos[pid] = (cx, cy, now)
        
        for pid in list(self.prev_pos.keys()):
            if pid not in active_ids and now - self.prev_pos[pid][2] > self.max_gap_seconds:
                del self.prev_pos[pid]; self.speed_hist.pop(pid, None)

        return speeds
    
    def avg_speed(self, speeds):
        if not speeds: return 0.0
        # Only average for players considered "moving"
        moving_speeds = [s.get("speed_kmh", 0.0) for s in speeds.values() if s.get("moving", False)]
        if not moving_speeds: return 0.0
        return round(float(np.mean(moving_speeds)), 1)

    def fastest(self, speeds):
        if not speeds: return None, 0.0
        pid = max(speeds, key=lambda k: speeds[k].get("speed_kmh", 0.0))
        return pid, speeds[pid].get("speed_kmh", 0.0)

    def reset(self):
        self.prev_pos.clear(); self.speed_hist.clear()
        print("[SpeedEstimator] Reset")
# tracking/player_tracker.py
# ============================================================
# Player Tracker — Sport-aware with proper player limits
# Players numbered from P1 fresh on each sport switch
# ============================================================

import cv2
import numpy as np
from collections import defaultdict
from config import config

# Sport configurations with HARD player limits
SPORT_CONFIG = {
    "badminton": {
        "max_players": 4,       # Max 4 (doubles). Singles = 2
        "yolo_conf": 0.50,
        "description": "Badminton (max 4 players)",
        "label_color": (0, 255, 255),
    },
    "basketball": {
        "max_players": 10,      # 5v5
        "yolo_conf": 0.45,
        "description": "Basketball (max 10 players)",
        "label_color": (0, 140, 255),
    },
    "cricket": {
        "max_players": 13,      # 11 + 2 batsmen on field
        "yolo_conf": 0.50,
        "description": "Cricket (max 13 players)",
        "label_color": (0, 255, 0),
    },
    "football": {
        "max_players": 22,      # 11v11
        "yolo_conf": 0.45,
        "description": "Football (max 22 players)",
        "label_color": (255, 100, 0),
    },
}


class PlayerTracker:

    def __init__(self):
        self.sport = None
        self.max_players = 4            # Default safe limit
        self.yolo_conf = 0.45
        self.description = "Initializing..."
        self.label_color = (200, 200, 200)

        # ── Tracking state ────────────────────────────────────
        self._next_id = 1               # Always starts from 1, resets on sport change
        self._tracked = {}              # id → {bbox, centroid, lost, team}
        self._id_map = {}               # centroid_key → player_id
        self.frame_count = 0

        # ── Team state ────────────────────────────────────────
        self._team_mode = "AUTO"
        self._split_mode = "VERTICAL"
        self._manual_teams = {}         # player_id → "A" or "B"

        # ── Stats ─────────────────────────────────────────────
        self._total_detected = 0

        self._load_yolo()

    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            print("[PlayerTracker] Loading YOLOv8 model...")
            self.yolo = YOLO(config.model.YOLO_MODEL_PATH)
            self.available = True
            print("[PlayerTracker] YOLO tracker ready")
        except Exception as e:
            print(f"[PlayerTracker] YOLO load failed: {e}")
            self.yolo = None
            self.available = False

    def set_sport(self, sport: str):
        """Set sport and apply correct player limit. Resets player IDs."""
        sport = sport.lower().strip()

        if sport == self.sport:
            return  # No change needed

        old_sport = self.sport
        self.sport = sport

        cfg = SPORT_CONFIG.get(sport, {
            "max_players": 10,
            "yolo_conf": 0.45,
            "description": f"{sport.title()} (default)",
            "label_color": (200, 200, 200),
        })

        self.max_players = cfg["max_players"]
        self.yolo_conf = cfg["yolo_conf"]
        self.description = cfg["description"]
        self.label_color = cfg["label_color"]

        # ── FULL RESET on sport change ────────────────────────
        # Player IDs restart from P1 for new sport
        self._full_reset_ids()

        print(
            f"[Tracker] Sport set: {sport}, "
            f"Max Players: {self.max_players}, "
            f"YOLO Conf: {self.yolo_conf}"
        )
        if old_sport and old_sport != sport:
            print(f"[Tracker] Player IDs reset: {old_sport} → {sport}")

    def _full_reset_ids(self):
        """Reset all tracking — player IDs restart from P1."""
        self._next_id = 1
        self._tracked = {}
        self._id_map = {}
        self._manual_teams = {}
        self.frame_count = 0

    def detect(self, frame) -> list:
        """Detect and track players. Returns list of player dicts."""
        if not self.available or self.yolo is None:
            return []

        self.frame_count += 1
        h, w = frame.shape[:2]

        try:
            results = self.yolo(
                frame,
                classes=[0],                    # Person only
                conf=self.yolo_conf,
                verbose=False
            )
        except Exception as e:
            print(f"[Tracker] YOLO error: {e}")
            return []

        # ── Extract valid detections ──────────────────────────
        raw_detections = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                bw = x2 - x1
                bh = y2 - y1
                area = bw * bh

                # ── Filter bad detections ─────────────────────
                if bw < config.detection.PLAYER_MIN_WIDTH:
                    continue
                if bh < config.detection.PLAYER_MIN_HEIGHT:
                    continue
                if area < config.detection.PLAYER_MIN_AREA:
                    continue
                if area > config.detection.PLAYER_MAX_AREA:
                    continue
                if bw / bh > config.detection.PLAYER_MAX_ASPECT:
                    continue

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                raw_detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "centroid": (cx, cy),
                    "conf": conf,
                    "w": bw,
                    "h": bh,
                })

        # ── Sort by confidence, take top N ───────────────────
        raw_detections.sort(key=lambda d: d["conf"], reverse=True)
        raw_detections = raw_detections[:self.max_players]  # Hard limit

        # ── Match to existing tracked players ─────────────────
        players = self._match_and_update(raw_detections, w, h)

        self._total_detected += len(players)
        return players

    def _match_and_update(self, detections, frame_w, frame_h) -> list:
        """Match detections to tracked IDs using nearest centroid."""

        # Mark all tracked as potentially lost
        for pid in self._tracked:
            self._tracked[pid]["lost"] += 1

        matched_ids = set()
        players = []

        for det in detections:
            cx, cy = det["centroid"]
            best_id = None
            best_dist = config.detection.PLAYER_TRACK_MAX_DIST

            # Find nearest existing tracked player
            for pid, info in self._tracked.items():
                if pid in matched_ids:
                    continue
                px, py = info["centroid"]
                dist = np.hypot(cx - px, cy - py)
                if dist < best_dist:
                    best_dist = dist
                    best_id = pid

            if best_id is not None:
                # Update existing player
                self._tracked[best_id].update({
                    "bbox": det["bbox"],
                    "centroid": (cx, cy),
                    "lost": 0,
                    "conf": det["conf"],
                })
                matched_ids.add(best_id)
                pid = best_id
            else:
                # New player — assign next sequential ID
                pid = self._next_id
                self._next_id += 1
                self._tracked[pid] = {
                    "bbox": det["bbox"],
                    "centroid": (cx, cy),
                    "lost": 0,
                    "conf": det["conf"],
                    "team": "A",    # Default team
                }
                matched_ids.add(pid)

            # Determine team
            team = self._get_team(pid, cx, frame_w, frame_h)
            self._tracked[pid]["team"] = team

            players.append({
                "id": pid,
                "label": f"P{pid}",         # Clean label: P1, P2, P3...
                "bbox": det["bbox"],
                "centroid": (cx, cy),
                "conf": det["conf"],
                "team": team,
            })

        # ── Remove lost players ───────────────────────────────
        lost_ids = [
            pid for pid, info in self._tracked.items()
            if info["lost"] > config.detection.PLAYER_LOST_FRAMES
        ]
        for pid in lost_ids:
            del self._tracked[pid]

        # ── Sort players by ID for consistent display ─────────
        players.sort(key=lambda p: p["id"])

        return players

    def _get_team(self, player_id, cx, frame_w, frame_h) -> str:
        """Determine team assignment."""

        # Manual override takes priority
        if player_id in self._manual_teams:
            return self._manual_teams[player_id]

        if self._team_mode == "MANUAL":
            return self._tracked.get(player_id, {}).get("team", "A")

        # Auto assignment based on split mode
        if self._split_mode == "VERTICAL":
            return "A" if cx < frame_w // 2 else "B"
        elif self._split_mode == "HORIZONTAL":
            cy = self._tracked.get(player_id, {}).get("centroid", (0, 0))[1]
            return "A" if cy < frame_h // 2 else "B"

        return "A"

    def draw(self, frame, players: list, speeds: dict):
        """Draw player boxes, labels, speeds on frame."""
        if not players:
            return frame

        h, w = frame.shape[:2]

        # Sport-specific max display
        display_players = players[:self.max_players]

        for p in display_players:
            pid = p["id"]
            x1, y1, x2, y2 = p["bbox"]
            team = p.get("team", "A")
            label = p.get("label", f"P{pid}")

            # Team colors
            color = (0, 180, 255) if team == "A" else (255, 100, 0)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Speed label
            speed = speeds.get(pid, 0.0)
            display_label = f"{label} {speed:.1f}km/h"

            # Background for text
            (tw, th), _ = cv2.getTextSize(
                display_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, 1
            )
            label_y = max(y1 - 5, th + 5)
            cv2.rectangle(
                frame,
                (x1, label_y - th - 4),
                (x1 + tw + 4, label_y + 2),
                color, -1
            )
            cv2.putText(
                frame, display_label,
                (x1 + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1
            )

            # Centroid dot
            cx, cy = p["centroid"]
            cv2.circle(frame, (cx, cy), 4, color, -1)

        # ── Sport info overlay ────────────────────────────────
        info = (
            f"{self.sport.upper() if self.sport else 'SPORT'} | "
            f"Players: {len(display_players)}/{self.max_players}"
        )
        cv2.putText(
            frame, info,
            (w - len(info) * 7, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40, (200, 200, 200), 1
        )

        return frame

    def reset(self):
        """Soft reset — keep sport config but reset tracking."""
        self._full_reset_ids()
        if config.LOG_DETECTIONS:
            print(f"[Tracker] Reset (sport={self.sport})")

    def full_reset(self):
        """Hard reset — clear everything including sport."""
        self.sport = None
        self.max_players = 4
        self.description = "No sport"
        self._full_reset_ids()
        if config.LOG_DETECTIONS:
            print("[Tracker] Full reset")

    def get_stats(self) -> dict:
        return {
            "sport": self.sport,
            "max_players": self.max_players,
            "description": self.description,
            "active_players": len(self._tracked),
            "next_id": self._next_id,
            "total_detected": self._total_detected,
            "frame_count": self.frame_count,
            "team_mode": self._team_mode,
            "split_mode": self._split_mode,
        }

    def set_max_players(self, n: int):
        """Manual override of max players."""
        self.max_players = max(1, min(n, config.detection.MAX_PLAYERS_GLOBAL))

    def set_player_team(self, player_id: int, team: str) -> bool:
        if player_id in self._tracked:
            team = team.upper()
            self._manual_teams[player_id] = team
            self._tracked[player_id]["team"] = team
            return True
        return False

    def swap_player_team(self, player_id: int) -> str:
        current = self._tracked.get(player_id, {}).get("team", "A")
        new_team = "B" if current == "A" else "A"
        self._manual_teams[player_id] = new_team
        if player_id in self._tracked:
            self._tracked[player_id]["team"] = new_team
        return new_team

    def set_team_mode(self, mode: str):
        self._team_mode = mode.upper()
        self._manual_teams.clear()

    def set_split_mode(self, mode: str):
        self._split_mode = mode.upper()

    def clear_manual_teams(self):
        self._manual_teams.clear()
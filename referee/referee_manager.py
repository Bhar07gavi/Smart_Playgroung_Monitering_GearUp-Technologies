# referee/referee_manager.py
import numpy as np
from config import config # Import config

from referee.cricket_referee import CricketReferee
from referee.football_referee import FootballReferee
from referee.basketball_referee import BasketballReferee
from referee.badminton_referee import BadmintonReferee


class RefereeManager:
    def __init__(self):
        print("[RefereeManager] Loading sport analytics modules...")

        self.modules = {
            "cricket": CricketReferee(),
            "football": FootballReferee(),
            "basketball": BasketballReferee(),
            "badminton": BadmintonReferee()
        }

        self.active_sport = None

        print(f"[RefereeManager] Ready: {list(self.modules.keys())}")

    def update(self, sport, frame, players, speeds, sport_confidence=0.0):
        if not sport or sport == "unknown":
            return {
                "mode": "WAITING",
                "sport": None,
                "message": "Waiting for confident sport detection",
                "event": None
            }

        # Use config.detection.SPORT_CONF_REFEREE for this threshold
        if sport_confidence < config.detection.SPORT_CONF_REFEREE:
            return {
                "mode": "WAITING",
                "sport": sport,
                "message": f"Analyzing {sport}... confidence {sport_confidence:.0%}",
                "event": None
            }

        if self.active_sport != sport:
            self.reset()
            self.active_sport = sport

        module = self.modules.get(sport)

        if module:
            try:
                return module.analyze(
                    frame,
                    players,
                    speeds,
                    sport_confidence
                )
            except Exception as e:
                print(f"[RefereeManager] Module error for {sport}: {e}")

        return self._fallback(sport, players, speeds)

    def _fallback(self, sport, players, speeds):
        avg = 0.0
        if speeds:
            vals = [s.get("speed_kmh", 0.0) for s in speeds.values()]
            if vals:
                avg = sum(vals) / len(vals)

        return {
            "mode": "ANALYTICS",
            "sport": sport,
            "message": f"{sport.upper()} analytics | Players:{len(players)} | Avg:{avg:.1f} km/h",
            "event": None
        }

    def reset(self):
        for m in self.modules.values():
            if hasattr(m, "reset"):
                m.reset()
# referee/cricket_referee.py
from config import config # Import config

class CricketReferee:
    def __init__(self):
        self.sport = "cricket"

    def analyze(self, frame, players, speeds, sport_confidence):
        msg = f"Cricket play detected | Players:{len(players)}"

        if players: # Only analyze speed if players are detected
            fastest_player_id = None
            max_speed_kmh = 0.0
            for pid, speed_data in speeds.items():
                if speed_data.get("speed_kmh", 0.0) > max_speed_kmh:
                    max_speed_kmh = speed_data["speed_kmh"]
                    fastest_player_id = pid
            
            if fastest_player_id is not None:
                if max_speed_kmh >= config.referee.CRICKET_FAST_RUN_KMH:
                    msg = f"Fast run between wickets possible | P{fastest_player_id} {max_speed_kmh:.1f} km/h"
                elif max_speed_kmh >= config.referee.CRICKET_ACTIVE_KMH:
                    msg = f"Bowler-striker motion active | P{fastest_player_id} {max_speed_kmh:.1f} km/h"
        else: # No players detected
            msg = "Cricket pitch is empty."

        return {
            "mode": "ANALYTICS",
            "sport": self.sport,
            "message": msg,
            "event": None 
        }
        
    def reset(self):
        pass
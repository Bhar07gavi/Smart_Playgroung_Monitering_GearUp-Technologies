# referee/basketball_referee.py
from config import config # Import config

class BasketballReferee:
    def __init__(self):
        self.sport = "basketball"
        
    def analyze(self, frame, players, speeds, sport_confidence):
        msg = f"Basketball play detected | Players:{len(players)}"

        if players: # Only analyze speed if players are detected
            fastest_player_id = None
            max_speed_kmh = 0.0
            for pid, speed_data in speeds.items():
                if speed_data.get("speed_kmh", 0.0) > max_speed_kmh:
                    max_speed_kmh = speed_data["speed_kmh"]
                    fastest_player_id = pid
            
            if fastest_player_id is not None:
                if max_speed_kmh >= config.referee.BASKETBALL_FAST_DRIVE_KMH:
                    msg = f"Fast player drive detected | P{fastest_player_id} {max_speed_kmh:.1f} km/h"
                elif max_speed_kmh >= config.referee.BASKETBALL_TRANSITION_KMH:
                    msg = f"Basketball transition movement | P{fastest_player_id} {max_speed_kmh:.1f} km/h"
        else: # No players detected
            msg = "Basketball court is empty."

        return {
            "mode": "ANALYTICS",
            "sport": self.sport,
            "message": msg,
            "event": None 
        }
        
    def reset(self):
        pass
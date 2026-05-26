# referee/football_referee.py
from config import config # Import config

class FootballReferee:
    def __init__(self):
        self.sport = "football"

    def analyze(self, frame, players, speeds, sport_confidence):
        msg = f"Football play detected | Players:{len(players)}"

        if players: # Only analyze speed if players are detected
            # Get fastest player
            fastest_player_id = None
            max_speed_kmh = 0.0
            for pid, speed_data in speeds.items():
                if speed_data.get("speed_kmh", 0.0) > max_speed_kmh:
                    max_speed_kmh = speed_data["speed_kmh"]
                    fastest_player_id = pid
            
            if fastest_player_id is not None:
                if max_speed_kmh >= config.referee.FOOTBALL_FAST_BREAK_KMH:
                    msg = f"Fast attack movement | P{fastest_player_id} {max_speed_kmh:.1f} km/h"
                elif max_speed_kmh >= config.referee.FOOTBALL_ACTIVE_KMH:
                    msg = f"Active football movement | P{fastest_player_id} {max_speed_kmh:.1f} km/h"
        else: # No players detected
            msg = "Football field is empty."

        return {
            "mode": "ANALYTICS",
            "sport": self.sport,
            "message": msg,
            "event": None
        }
        
    def reset(self):
        pass
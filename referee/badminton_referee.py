# referee/badminton_referee.py
from config import config # Import config

class BadmintonReferee:
    def __init__(self):
        self.sport = "badminton"
        
    def analyze(self, frame, players, speeds, sport_confidence):
        match_type = "Singles" if len(players) <= 2 else "Doubles"
        msg = f"Badminton rally monitoring | {match_type}"

        if players: # Only analyze speed if players are detected
            fastest_player_id = None
            max_speed_kmh = 0.0
            for pid, speed_data in speeds.items():
                if speed_data.get("speed_kmh", 0.0) > max_speed_kmh:
                    max_speed_kmh = speed_data["speed_kmh"]
                    fastest_player_id = pid
            
            if fastest_player_id is not None:
                if max_speed_kmh >= config.referee.BADMINTON_QUICK_EXCHANGE_KMH:
                    msg = f"Quick exchange detected | P{fastest_player_id} {max_speed_kmh:.1f} km/h"
        else: # No players detected
            msg = "Badminton court is empty."

        return {
            "mode": "ANALYTICS",
            "sport": self.sport,
            "message": msg,
            "event": None 
        }
        
    def reset(self):
        pass
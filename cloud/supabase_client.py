# cloud/supabase_client.py
# ============================================================
# Supabase Cloud Integration
# ============================================================

import os
import threading
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

try:
    from supabase import create_client, Client
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")   # ← Better for server # ← Use ANON_KEY from your .env
    
    supabase: Client = None
    connected = False
    
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            connected = True
            print(f"[Supabase] Connected: {SUPABASE_URL[:30]}...")
        except Exception as e:
            print(f"[Supabase ERROR] Connection failed: {e}")
            connected = False
    else:
        print("[Supabase] Missing credentials in .env file")
        connected = False
        
except ImportError:
    print("[Supabase] Package not installed (pip install supabase)")
    supabase = None
    connected = False


class SupabaseClient:
    """
    Supabase cloud client for live status and event uploads.
    """
    
    def __init__(self):
        self.connected = connected
        self.table_live = "live_status"
        self.table_events = "events"
        self.playground_id = "playground_1"
    
    def push_live_status(self, data):
        """Push live detection status to Supabase."""
        if not self.connected or supabase is None:
            print("[Supabase] Not connected, skipping live status push.")
            return False
        
        try:
            data["updated_at"] = datetime.utcnow().isoformat()
            
            supabase.table(self.table_live).upsert(data).execute()
            return True
        except Exception as e:
            print(f"[Supabase ERROR] Live status push failed: {e}")
            return False
    
    def process_event(self, clip_path, metadata):
        """Upload event clip to Supabase Storage and create record."""
        if not self.connected or supabase is None:
            print("[Supabase] Not connected, skipping event upload.")
            return None
        
        try:
            # Upload video file to storage
            with open(clip_path, "rb") as f:
                file_name = f"{self.playground_id}_{int(time.time())}.mp4"
                supabase.storage.from_("event-clips").upload(file_name, f)
            
            # Get public URL
            video_url = supabase.storage.from_("event-clips").get_public_url(file_name)
            
            # Create event record
            event_data = {
                "event_type": metadata.get("event_type", "unknown"),
                "video_url": video_url,
                "sport": metadata.get("sport", ""),
                "mode": metadata.get("mode", "SAFETY"),
                "playground_id": self.playground_id,
                "created_at": datetime.utcnow().isoformat()
            }
            
            result = supabase.table(self.table_events).insert(event_data).execute()
            
            if result.data:
                print(f"[Supabase] Event uploaded: {metadata.get('event_type')}")
                return result.data[0]["id"]
            
            return None
            
        except Exception as e:
            print(f"[Supabase ERROR] Event upload failed: {e}")
            return None
    
    def get_recent_events(self, limit=10):
        """Get recent event clips from Supabase."""
        if not self.connected or supabase is None:
            return []
        
        try:
            result = supabase.table(self.table_events)\
                .select("*")\
                .order("created_at", desc=True)\
                .limit(limit)\
                .execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            print(f"[Supabase ERROR] Get events failed: {e}")
            return []
    
    def push_esp32_status(self, esp32_data):
        """Push ESP32 status to Supabase."""
        if not self.connected or supabase is None:
            return False
        
        try:
            data = {
                "ip_address": esp32_data.get("ip", ""),
                "rssi": esp32_data.get("rssi", 0),
                "heap_free": esp32_data.get("heap_free", 0),
                "uptime_sec": esp32_data.get("uptime_sec", 0),
                "frame_count": esp32_data.get("frame_count", 0),
                "connected": esp32_data.get("connected", False),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            supabase.table("esp32_status").upsert(data).execute()
            return True
            
        except Exception as e:
            print(f"[Supabase ERROR] ESP32 status push failed: {e}")
            return False
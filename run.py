# run.py
# ============================================================
# SINGLE COMMAND LAUNCHER
# Starts everything: ESP32 reader + AI pipeline + API server
#
# Usage:
#   python run.py
# ============================================================

import threading
import sys
import os
import time
import socket

import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import SmartPlaygroundSystem
from config import config


def get_local_ip():
    """Get this machine's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def check_config():
    """Warn about common config issues."""
    issues = []

    if config.esp32.IP_ADDRESS == "192.168.1.45":
        issues.append(
            "ESP32 IP is still the default placeholder.\n"
            "  → Open config.py and set ESP32Config.IP_ADDRESS\n"
            "    to the IP shown in Arduino Serial Monitor."
        )

    supabase_url = os.getenv("SUPABASE_URL", "")
    if not supabase_url:
        issues.append(
            "SUPABASE_URL missing in .env file.\n"
            "  → System will run without cloud features.\n"
            "  → Add SUPABASE_URL and SUPABASE_SERVICE_KEY to .env to enable."
        )

    if not os.path.exists("models/sports_v2.tflite"):
        issues.append(
            "sports_v2.tflite not found in models/ folder.\n"
            "  → Copy your trained model there."
        )

    if not os.path.exists("models/uniform_detection_final.tflite"):
        issues.append(
            "uniform_detection_final.tflite not found in models/ folder.\n"
            "  → Copy your trained model there."
        )

    return issues


def print_banner(pc_ip):
    port = config.api.PORT
    print("\n" + "═" * 62)
    print("  SMART PLAYGROUND MONITOR  |  ESP32-CAM Edition")
    print("═" * 62)
    print(f"\n  ESP32-CAM  : {config.esp32.STREAM_URL}")
    print(f"\n  Your PC IP : {pc_ip}")
    print(f"  Dashboard  : http://{pc_ip}:{port}/   ← OPEN THIS")
    print(f"  Live Stream: http://{pc_ip}:{port}/stream")
    print(f"  Status API : http://{pc_ip}:{port}/api/status")
    print(f"  WebSocket  : ws://{pc_ip}:{port}/ws")
    print("\n" + "─" * 62 + "\n")


def main():
    pc_ip = get_local_ip()

    # Print banner
    print_banner(pc_ip)

    # Config checks
    issues = check_config()
    if issues:
        print("⚠  CONFIGURATION WARNINGS:")
        for i, issue in enumerate(issues, 1):
            print(f"\n  {i}. {issue}")
        print()

    # Confirm start
    try:
        input("Press Enter to start the system  (Ctrl+C to cancel)...\n")
    except KeyboardInterrupt:
        print("\nCancelled.")
        return

    # Create system
    system = SmartPlaygroundSystem()

    # Start API server (server.py) in background thread
    api_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={
            "app"       : "server:app",
            "host"      : config.api.HOST,
            "port"      : config.api.PORT,
            "log_level" : "warning",
            "access_log": False
        },
        daemon=True,
        name="APIServer"
    )
    api_thread.start()

    # Give server time to bind port
    time.sleep(2)

    print(f"✓ API server running on port {config.api.PORT}")
    print(f"✓ Open browser: http://{pc_ip}:{config.api.PORT}/\n")

    # Run AI processing loop (blocking — runs until Ctrl+C)
    system.run()


if __name__ == "__main__":
    main()
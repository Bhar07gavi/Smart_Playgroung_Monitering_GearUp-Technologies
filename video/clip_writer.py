# video/clip_writer.py
# ============================================================
import cv2
import os
import time
import threading
from pathlib import Path
from config import config


class ClipWriter:

    def __init__(self):
        self.save_dir = Path(config.buffer.SAVE_DIR)
        self.save_dir.mkdir(exist_ok=True)
        self._lock  = threading.Lock()
        self._clips = []
        self._scan_existing()
        print(f"[ClipWriter] Ready. Saving to '{self.save_dir}'")

    def _scan_existing(self):
        """Load clips already on disk at startup."""
        files = sorted(
            self.save_dir.glob("*.mp4"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        for f in files[:config.buffer.MAX_CLIPS_STORED]:
            self._clips.append(self._make_meta(f))
        if self._clips:
            print(f"[ClipWriter] Loaded {len(self._clips)} existing clips")

    def _make_meta(self, path: Path) -> dict:
        """Build metadata dict for a clip file."""
        try:
            stat = path.stat()
        except Exception:
            return {}

        # Parse event_type and sport from filename
        # Format: eventtype_sport_YYYYMMDD_HHMMSS.mp4
        #      OR: eventtype_YYYYMMDD_HHMMSS.mp4
        parts      = path.stem.split("_")
        event_type = parts[0] if parts else "event"
        sport      = parts[1] if len(parts) > 3 else None

        return {
            "filename":   path.name,
            "local_url":  f"/clips/{path.name}",
            "event_type": event_type,
            "sport":      sport,
            "timestamp":  stat.st_mtime * 1000,   # ms for JS
            "size_kb":    round(stat.st_size / 1024, 1),
        }

    def write(self, clip: dict) -> str | None:
        """Write frames to disk. Returns file path or None."""
        frames     = clip.get("frames", [])
        event_type = clip.get("event_type", "event")
        sport      = clip.get("sport", "")
        fps        = clip.get("fps", config.buffer.FPS)

        if not frames:
            return None

        ts       = time.strftime("%Y%m%d_%H%M%S")
        sport_tag = f"_{sport}" if sport else ""
        filename  = f"{event_type}{sport_tag}_{ts}.mp4"
        filepath  = self.save_dir / filename

        try:
            h, w = frames[0].shape[:2]

            # ── Try H.264 first (browser-compatible) ──────────
            # avc1 = H.264, plays in all browsers natively
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(
                str(filepath), fourcc, fps, (w, h))

            # If avc1 not supported, fall back to mp4v
            if not writer.isOpened():
                writer.release()
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(filepath), fourcc, fps, (w, h))

            if not writer.isOpened():
                print(f"[ClipWriter] ERROR: Cannot open writer for "
                      f"{filename}")
                return None

            for frame in frames:
                writer.write(frame)
            writer.release()

            # Verify file was actually written
            if not filepath.exists() or filepath.stat().st_size < 1000:
                print(f"[ClipWriter] ERROR: File too small or missing: "
                      f"{filename}")
                return None

            meta = self._make_meta(filepath)
            if not meta:
                return None

            with self._lock:
                self._clips.insert(0, meta)
                # Trim oldest clips
                while len(self._clips) > config.buffer.MAX_CLIPS_STORED:
                    old = self._clips.pop()
                    try:
                        old_path = self.save_dir / old["filename"]
                        old_path.unlink(missing_ok=True)
                    except Exception:
                        pass

            size = meta.get("size_kb", 0)
            print(f"[ClipWriter] ✓ Saved: {filename} "
                  f"({len(frames)}f, {size}KB)")
            return str(filepath)

        except Exception as e:
            print(f"[ClipWriter] ERROR: {e}")
            try:
                filepath.unlink(missing_ok=True)
            except Exception:
                pass
            return None

    def get_all_local_clips(self) -> list:
        """Return list of clip metadata dicts, newest first."""
        with self._lock:
            # Re-verify files still exist
            valid = []
            for c in self._clips:
                p = self.save_dir / c.get("filename", "")
                if p.exists():
                    valid.append(c)
            self._clips = valid
            return list(self._clips)
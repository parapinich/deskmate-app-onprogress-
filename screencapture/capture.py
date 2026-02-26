"""
DeskMate Screen Capture Sidecar
Native OS-level screenshot capture using mss library.
Called by Tauri as a sidecar process — outputs base64 JPEG to stdout.
"""

import mss
import base64
import io
import sys
import json
from PIL import Image


def capture_screen() -> str:
    """
    Capture the primary monitor and return as base64-encoded JPEG.
    Image is resized to max 1280x1280 to reduce payload size.
    """
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary monitor
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        img.thumbnail((1280, 1280), Image.LANCZOS)  # Resize to reduce payload
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=70)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


if __name__ == "__main__":
    try:
        screenshot_b64 = capture_screen()
        print(json.dumps({"screenshot": screenshot_b64}))
        sys.stdout.flush()
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

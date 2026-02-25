"""
Question Generator — Handles question generation with dedup and retry logic.
MD5 hash deduplication to avoid repeat questions on the same slide.
Exponential backoff on rate limits.
"""

import asyncio
import hashlib
import time
from collections import defaultdict
from .deskmate_agent import analyze_screen_content


# Per-session screenshot hash history for deduplication
_screenshot_hashes: dict[str, list[str]] = defaultdict(list)
MAX_HASH_HISTORY = 3  # Keep last N hashes per session

# Rate limit retry config
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds


def _compute_hash(image_base64: str) -> str:
    """Compute MD5 hash of the image data for deduplication."""
    return hashlib.md5(image_base64.encode("utf-8")).hexdigest()


def _is_duplicate(session_id: str, image_hash: str) -> bool:
    """Check if the same screenshot was seen in the last N captures for this session."""
    return image_hash in _screenshot_hashes[session_id]


def _record_hash(session_id: str, image_hash: str) -> None:
    """Record a screenshot hash, keeping only the last MAX_HASH_HISTORY entries."""
    hashes = _screenshot_hashes[session_id]
    hashes.append(image_hash)
    if len(hashes) > MAX_HASH_HISTORY:
        _screenshot_hashes[session_id] = hashes[-MAX_HASH_HISTORY:]


async def generate_questions(image_base64: str, session_id: str) -> dict:
    """
    Generate study questions from a screenshot with dedup and retry.
    
    Args:
        image_base64: Base64-encoded JPEG screenshot.
        session_id: UUID string for the current study session.
        
    Returns:
        dict with question data, or skip/error indicators.
    """
    # --- Deduplication ---
    image_hash = _compute_hash(image_base64)

    if _is_duplicate(session_id, image_hash):
        return {
            "is_study_material": False,
            "skipped": True,
            "reason": "duplicate_screenshot"
        }

    # --- Call Gemini with exponential backoff ---
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            result = await analyze_screen_content(image_base64)

            # Success — record hash so future identical screenshots are skipped
            _record_hash(session_id, image_hash)
            return result

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)

    # All retries exhausted
    return {
        "is_study_material": False,
        "error": f"Failed after {MAX_RETRIES} retries: {str(last_error)}"
    }


def clear_session_hashes(session_id: str) -> None:
    """Clear hash history for a session (called on session end)."""
    _screenshot_hashes.pop(session_id, None)

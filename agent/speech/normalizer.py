from __future__ import annotations
import re

def normalize_for_tts(text: str) -> str:
    if not text:
        return ""
    # Strip markdown links [text](url) -> text
    cleaned = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Strip URLs
    cleaned = re.sub(r'https?://\S+|www\.\S+', '', cleaned)
    # Strip code blocks and markdown symbols
    cleaned = re.sub(r'```[\s\S]*?```', '', cleaned)
    cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)
    cleaned = re.sub(r'[#*_~>|]', '', cleaned)
    # Normalize currency
    cleaned = cleaned.replace('₹', ' రూపాయలు ').replace('$', ' డాలర్లు ')
    # Collapse extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

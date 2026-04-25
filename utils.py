"""Utility helpers for TalentScout."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

# Accepts digits with optional separators and country code.
PHONE_PATTERN = re.compile(r"^\+?[0-9\s\-()]{7,20}$")

EXIT_KEYWORDS = {"exit", "quit", "bye"}


def normalize_text(value: str) -> str:
    """Normalize whitespace and trim surrounding spaces."""
    return " ".join((value or "").strip().split())


def validate_email(email: str) -> bool:
    """Validate email address format."""
    return bool(EMAIL_PATTERN.match(normalize_text(email)))


def validate_phone(phone: str) -> bool:
    """Validate phone number format and ensure enough digits exist."""
    cleaned = normalize_text(phone)
    if not PHONE_PATTERN.match(cleaned):
        return False
    digits = re.sub(r"\D", "", cleaned)
    return 7 <= len(digits) <= 15


def validate_experience(experience: str) -> Tuple[bool, float | None]:
    """Validate years of experience as a non-negative number."""
    value = normalize_text(experience)
    try:
        years = float(value)
    except ValueError:
        return False, None

    if years < 0 or years > 60:
        return False, None
    return True, years


def parse_tech_stack(raw_input: str) -> List[str]:
    """Parse technology text into a de-duplicated list."""
    text = normalize_text(raw_input)
    if not text:
        return []

    tokens = re.split(r"[,/;|]", text)
    cleaned = []
    seen = set()

    for token in tokens:
        item = token.strip()
        if not item:
            continue
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            cleaned.append(item)

    return [item for item in cleaned if is_reasonable_technology_name(item)]


def parse_desired_positions(raw_input: str) -> List[str]:
    """Parse desired positions into a normalized de-duplicated list."""
    text = normalize_text(raw_input)
    if not text:
        return []

    tokens = re.split(r"[,/;|]", text)
    positions: List[str] = []
    seen = set()

    for token in tokens:
        item = normalize_text(token)
        if len(item) < 2 or not has_alpha_content(item):
            continue
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            positions.append(item)

    return positions


def has_alpha_content(value: str) -> bool:
    """Return True when the text contains at least one alphabet character."""
    return bool(re.search(r"[A-Za-z]", normalize_text(value)))


def is_reasonable_technology_name(value: str) -> bool:
    """Validate that a token resembles a real technology name."""
    item = normalize_text(value)
    if len(item) < 2 or len(item) > 32:
        return False

    # Tech names should have letters and should not start with a digit-only style token.
    if not has_alpha_content(item):
        return False
    if item[0].isdigit():
        return False

    # Allow common symbols used in tech names.
    allowed_pattern = re.compile(r"^[A-Za-z.#+][A-Za-z0-9 .#+_-]*$")
    return bool(allowed_pattern.match(item))


def is_exit_intent(user_input: str) -> bool:
    """Return True if the user intends to stop the conversation."""
    normalized = normalize_text(user_input).casefold()
    return normalized in EXIT_KEYWORDS


def detect_sentiment(user_input: str) -> str:
    """Return a basic sentiment label for optional UI hinting."""
    text = normalize_text(user_input).casefold()
    if not text:
        return "neutral"

    positive_words = {"great", "good", "awesome", "excited", "happy", "confident"}
    negative_words = {"bad", "nervous", "stressed", "worried", "anxious", "upset"}

    if any(word in text for word in positive_words):
        return "positive"
    if any(word in text for word in negative_words):
        return "negative"
    return "neutral"


def mask_email(email: str) -> str:
    """Mask email for safe display in UI previews."""
    value = normalize_text(email)
    if "@" not in value:
        return "***"

    local, domain = value.split("@", 1)
    if len(local) <= 2:
        masked_local = "*" * len(local)
    else:
        masked_local = local[0] + ("*" * (len(local) - 2)) + local[-1]
    return f"{masked_local}@{domain}"


def mask_phone(phone: str) -> str:
    """Mask phone number for safe display in UI previews."""
    digits = re.sub(r"\D", "", normalize_text(phone))
    if len(digits) <= 4:
        return "*" * max(len(digits), 1)
    return ("*" * (len(digits) - 4)) + digits[-4:]


def mask_profile_data(profile: Dict[str, object]) -> Dict[str, object]:
    """Return a masked profile dictionary for safe display."""
    masked = dict(profile)

    email_value = str(masked.get("email") or "")
    if email_value:
        masked["email"] = mask_email(email_value)

    phone_value = str(masked.get("phone_number") or "")
    if phone_value:
        masked["phone_number"] = mask_phone(phone_value)

    full_name = str(masked.get("full_name") or "")
    if full_name:
        parts = full_name.split()
        if parts:
            masked["full_name"] = f"{parts[0]} {'*' * 3}"

    return masked

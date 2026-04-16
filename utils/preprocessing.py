"""Common preprocessing helpers used by training and inference scripts."""

from __future__ import annotations


def clean_text(text: str) -> str:
    """Basic text cleaning helper."""
    return " ".join(text.strip().split()).lower()

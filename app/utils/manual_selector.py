
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Optional, Tuple

from app.utils.manual_registry import ManualEntry


def _normalize_question(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def _question_words(q: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", q.lower())


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def select_manual_from_question(
    question: str,
    registry: Dict[str, ManualEntry],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Manual selection by:
      1) exact whole-word token match
      2) fuzzy match (handles misspellings like vezzel -> vezel)
    Returns: (file_name, matched_token)
    """
    q = _normalize_question(question)
    words = _question_words(q)

    best_file = None
    best_token = None
    best_score = 0.0

    for file_name, entry in registry.items():
        score = 0.0
        matched = None

        for token in entry.tokens:
            # ---- exact whole-word match
            if re.search(rf"\b{re.escape(token)}\b", q):
                score += 1.0
                matched = token
                continue

            # ---- fuzzy match against individual words (misspelling tolerance)
            # Only fuzzy-match reasonably long words
            if len(token) >= 4:
                for w in words:
                    if len(w) >= 4 and _similar(w, token) >= 0.88:
                        score += 0.9  # slightly lower than exact match
                        matched = token
                        break

        # tie-break: higher score wins
        if score > best_score:
            best_score = score
            best_file = file_name
            best_token = matched

    if best_score <= 0:
        return None, None
    return best_file, best_token

# app/utils/manual_selector.py
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from app.utils.manual_registry import ManualEntry


def _normalize_question(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def select_manual_from_question(
    question: str,
    registry: Dict[str, ManualEntry],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to pick a manual purely by matching filename tokens inside the user question.

    Returns:
      (selected_file_name, matched_token)

    Example:
      question: "how to control headlights in outlander?"
      -> ("2014_outlander.pdf", "outlander")
    """
    q = _normalize_question(question)

    best_file = None
    best_token = None
    best_score = 0

    for file_name, entry in registry.items():
        # score = number of tokens from this manual that appear in the question
        def _has_word(q: str, token: str) -> bool:
            return re.search(rf"\b{re.escape(token)}\b", q) is not None

        hits = [t for t in entry.tokens if _has_word(q, t)]
        score = len(hits)

        # tie-break by longest matched token (more specific)
        best_len = len(best_token) if best_token else 0
        new_len = max((len(t) for t in hits), default=0)

        if score > best_score or (score == best_score and new_len > best_len):
            best_score = score
            best_file = file_name
            best_token = max(hits, key=len) if hits else None


    if best_score == 0:
        return None, None

    return best_file, best_token

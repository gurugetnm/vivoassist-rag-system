from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from llama_index.core import VectorStoreIndex


SYSTEM_PROMPT = """
You are a technical assistant for VivoAssist.

Rules:
- Answer ONLY using the provided PDF manual content.
- If the answer is not explicitly found in the manual, say exactly:
  "Not found in the manual."
- Do NOT guess or use external knowledge.
- Keep answers clear, concise, and technical.
- When possible, support answers with page numbers.
"""

NOT_FOUND = "Not found in the manual."


# -----------------------------
# Helpers
# -----------------------------
def _extract_sources(resp) -> List[Tuple[str, Optional[str]]]:
    source_nodes = getattr(resp, "source_nodes", None)
    if source_nodes is None:
        source_nodes = getattr(resp, "sources", None)
    if not source_nodes:
        return []

    out: List[Tuple[str, Optional[str]]] = []
    for sn in source_nodes:
        node = getattr(sn, "node", sn)
        meta = getattr(node, "metadata", {}) or {}
        file_name = meta.get("file_name", "unknown_file")
        page = meta.get("page_label") or meta.get("page_number") or meta.get("page")
        out.append((file_name, str(page) if page is not None else None))
    return out


def _is_models_question(q: str) -> bool:
    qn = q.lower().strip()

    # 1) Hard block: page-specific / content questions should NEVER hit fast-path
    page_patterns = [
        "page ", "page:", "on page", "say on page", "says on page", "what does", "what is on page"
    ]
    if any(p in qn for p in page_patterns):
        return False
    # also block patterns like "p. 1", "p 1"
    if re.search(r"\bp\.?\s*\d+\b", qn):
        return False

    # 2) Inventory/list intent (must be present)
    list_intent = any(
        k in qn for k in ["list", "show", "what", "which", "available", "all", "do you have"]
    )

    # 3) Inventory subject (models/manuals)
    subject = any(
        k in qn for k in [
            "models", "model", "vehicle models",
            "manuals", "manual list", "documents", "pdfs"
        ]
    )

    # 4) Light typo tolerance only around "models"
    typo_model = any(k in qn for k in ["modesl", "modles"])

    return (list_intent and subject) or (list_intent and typo_model)



def _split_models_and_rest(q: str) -> Tuple[bool, str]:
    """
    If query contains a models request PLUS another question joined by 'and',
    return (True, remainder_question). Otherwise (False, '').

    Examples:
      "what models do you have and what is 4wd lock"
        -> (True, "what is 4wd lock")
      "list manuals and how to connect bluetooth"
        -> (True, "how to connect bluetooth")
    """
    qn = q.strip()

    # Only split if it looks like it's asking for models/manuals
    if not _is_models_question(qn):
        return False, ""

    # Split on common conjunctions. Keep it conservative.
    parts = re.split(r"\s+\band\b\s+|\s+\bthen\b\s+", qn, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) < 2:
        return False, ""

    remainder = parts[1].strip()
    # If remainder is too short, ignore
    if len(remainder) < 5:
        return False, ""

    return True, remainder


def _print_models_from_cache(models_cache: Dict, *, debug: bool) -> bool:
    """
    Prints the model list from cache.
    Returns True if anything printed, else False.
    """
    if debug:
        print("✅ FAST PATH: using models_cache\n")

    print("Assistant: Models found across all manuals:\n")

    found_any = False
    for file_name, data in (models_cache or {}).items():
        for m in data.get("models", []):
            found_any = True
            pages = m.get("pages") or []
            if pages:
                print(f"- {m['name']} | {file_name} (pages: {', '.join(pages)})")
            else:
                print(f"- {m['name']} | {file_name}")

    if not found_any:
        print(NOT_FOUND)

    print()
    return found_any


# -----------------------------
# Chat Engine
# -----------------------------
def run_terminal_chat(
    index: VectorStoreIndex,
    *,
    top_k: int,
    debug: bool,
    data_dir: str,
    models_cache: dict,
):
    chat_engine = index.as_chat_engine(
        similarity_top_k=top_k,
        system_prompt=SYSTEM_PROMPT,
    )

    print("Chat ready. Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        # -------------------------------------------------
        # MIXED QUESTION: models list + another query
        # -------------------------------------------------
        is_mixed, remainder = _split_models_and_rest(q)
        if is_mixed:
            _print_models_from_cache(models_cache, debug=debug)

            # Now answer the remainder using normal RAG
            resp = chat_engine.chat(remainder)
            resp_text = str(resp).strip()
            print(f"Assistant: {resp_text}\n")

            if NOT_FOUND.lower() in resp_text.lower():
                continue

            sources = _extract_sources(resp)
            if sources:
                grouped = defaultdict(set)
                for f, p in sources:
                    if p is not None:
                        grouped[f].add(str(p))

                print("Sources:")
                for f, pages in grouped.items():
                    if pages:
                        pages_sorted = sorted(pages, key=lambda x: int(x) if x.isdigit() else x)
                        print(f"- {f} (pages: {', '.join(pages_sorted)})")
                    else:
                        print(f"- {f}")
                print()
            continue

        # -------------------------------------------------
        # FAST PATH: only models/manuals list
        # -------------------------------------------------
        if _is_models_question(q):
            _print_models_from_cache(models_cache, debug=debug)
            continue

        # -------------------------------------------------
        # NORMAL RAG CHAT
        # -------------------------------------------------
        resp = chat_engine.chat(q)
        resp_text = str(resp).strip()
        print(f"\nAssistant: {resp_text}\n")

        if NOT_FOUND.lower() in resp_text.lower():
            continue

        sources = _extract_sources(resp)
        if sources:
            grouped = defaultdict(set)
            for f, p in sources:
                if p is not None:
                    grouped[f].add(str(p))

            print("Sources:")
            for f, pages in grouped.items():
                if pages:
                    pages_sorted = sorted(pages, key=lambda x: int(x) if x.isdigit() else x)
                    print(f"- {f} (pages: {', '.join(pages_sorted)})")
                else:
                    print(f"- {f}")
            print()

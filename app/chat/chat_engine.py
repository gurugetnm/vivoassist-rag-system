from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Tuple

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
    qn = q.lower()

    model_like = any(
        k in qn
        for k in [
            "model",
            "models",
            "vehicle models",
            "all models",
            "all vehicle models",
            "available models",
            "list models",
        ]
    )

    intent = any(
        k in qn
        for k in [
            "what",
            "which",
            "list",
            "all",
            "available",
            "show",
            "give",
        ]
    )

    return model_like and intent



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
    """
    models_cache format:
    {
      "file.pdf": {
        "models": [
          {"name": "Model A", "pages": ["1","2"]},
          {"name": "Model B", "pages": ["1"]}
        ]
      }
    }
    """
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
        # FAST PATH: Models/manuals list from cache
        # -------------------------------------------------
        if _is_models_question(q):
            if debug:
                print("✅ FAST PATH: using models_cache")

            print("\nAssistant: Models found across all manuals:\n")

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
            continue

        # -------------------------------------------------
        # NORMAL RAG CHAT
        # -------------------------------------------------
        resp = chat_engine.chat(q)
        resp_text = str(resp).strip()
        print(f"\nAssistant: {resp_text}\n")

        # Do NOT show sources for refusals
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

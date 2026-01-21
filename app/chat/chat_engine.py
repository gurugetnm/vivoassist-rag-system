from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

from app.utils.manual_registry import ManualEntry
from app.utils.manual_selector import select_manual_from_question


SYSTEM_PROMPT = """
You are a technical assistant for VivoAssist.

Rules:
- Answer ONLY using the provided PDF manual content.
- If the answer is not explicitly found in the manual, say:
  "Not found in the manual."
- Do NOT guess or use external knowledge.
- Keep answers clear, concise, and technical.
- When possible, support answers with page numbers.
"""


def _extract_sources(resp):
    source_nodes = getattr(resp, "source_nodes", None)
    if source_nodes is None:
        source_nodes = getattr(resp, "sources", None)
    if not source_nodes:
        return []

    out = []
    for sn in source_nodes:
        node = getattr(sn, "node", sn)
        meta = getattr(node, "metadata", {}) or {}
        file_name = meta.get("file_name", "unknown_file")
        page = meta.get("page_label") or meta.get(
            "page_number") or meta.get("page")
        out.append((file_name, page))
    return out


def _print_sources(sources):
    if not sources:
        return

    grouped = defaultdict(set)
    for f, p in sources:
        if p is not None:
            grouped[f].add(str(p))

    print("Sources:")
    for f, pages in grouped.items():
        if pages:
            pages_sorted = sorted(
                pages, key=lambda x: int(x) if x.isdigit() else x)
            print(f"- {f} (pages: {', '.join(pages_sorted)})")
        else:
            print(f"- {f}")
    print()


def _make_filtered_retriever(index: VectorStoreIndex, *, top_k: int, allowed_file: Optional[str]):
    """
    Build a retriever hard-filtered to a single PDF (file_name) if allowed_file is set.
    """
    if allowed_file:
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="file_name", value=allowed_file)])
        return index.as_retriever(similarity_top_k=top_k, filters=filters)

    # No manual locked: global retriever (or you can choose to refuse)
    return index.as_retriever(similarity_top_k=top_k)


def _make_chat_engine(index: VectorStoreIndex, *, top_k: int, allowed_file: Optional[str]):
    """
    CondensePlusContextChatEngine does:
      - rewrite follow-ups ("explain more about it") into a full question
      - retrieve with context
      - answer grounded in retrieved nodes
    """
    retriever = _make_filtered_retriever(
        index, top_k=top_k, allowed_file=allowed_file)

    # This engine is STATEFUL: it keeps chat history so follow-ups get condensed properly.
    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        system_prompt=SYSTEM_PROMPT,
    )
    return chat_engine


def run_terminal_chat(
    index: VectorStoreIndex,
    *,
    top_k: int,
    debug: bool,
    manual_registry: Dict[str, ManualEntry],
):
    print("Chat ready. Type 'exit' to quit.")
    print("Commands:")
    print("  /manual <file.pdf>  -> force manual scope")
    print("  /clear              -> clear manual scope (unlock)")
    print()

    locked_manual: Optional[str] = None
    chat_engine = _make_chat_engine(
        index, top_k=top_k, allowed_file=locked_manual)
    manual_lock_mode: str = "auto"

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        # ---- Commands
        if q.startswith("/manual"):
            parts = q.split(maxsplit=1)
            if len(parts) == 2:
                wanted = parts[1].strip()
                if wanted in manual_registry:
                    locked_manual = wanted
                    manual_lock_mode = "forced"
                    chat_engine = _make_chat_engine(index, top_k=top_k, allowed_file=locked_manual)
                    print(f"\n[manual locked] {locked_manual}\n")
                else:
                    print("\nUnknown manual filename. Available manuals:")
                    for fn in sorted(manual_registry.keys()):
                        print(" -", fn)
                    print()
            else:
                print("\nUsage: /manual <file.pdf>\n")
            continue

        if q.startswith("/clear"):
            locked_manual = None
            manual_lock_mode = "auto"
            chat_engine = _make_chat_engine(index, top_k=top_k, allowed_file=locked_manual)
            print("\n[manual unlocked]\n")
            continue

        # ---- Auto-select / Auto-switch manual (unless user forced /manual)
        selected, token = select_manual_from_question(q, manual_registry)

        if manual_lock_mode != "forced" and selected:
            if locked_manual != selected:
                locked_manual = selected
                # IMPORTANT: recreate engine to reset history when switching manuals
                chat_engine = _make_chat_engine(index, top_k=top_k, allowed_file=locked_manual)
                if debug:
                    print(f"\n[auto-switched manual] {locked_manual} (matched: {token})\n")

        # ---- Chat (this is the important part)
        resp = chat_engine.chat(q)
        print(f"\nAssistant: {resp}\n")

        sources = _extract_sources(resp)

        # ---- Hard guard: if anything leaks outside locked manual, reject
        if locked_manual:
            bad = [f for (f, _) in sources if f != locked_manual]
            if bad:
                print("Assistant: Not found in the requested manual.\n")
                if debug:
                    print("[GUARD] Retrieved from wrong manuals:",
                          sorted(set(bad)))
                continue

        _print_sources(sources)

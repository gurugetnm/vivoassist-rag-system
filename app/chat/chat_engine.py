from collections import defaultdict
from llama_index.core import VectorStoreIndex


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


def run_terminal_chat(index: VectorStoreIndex, *, top_k: int, debug: bool):
    chat_engine = index.as_chat_engine(
        similarity_top_k=top_k,
        system_prompt=SYSTEM_PROMPT,
    )

    print("Chat ready. Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        # Normal RAG chat
        resp = chat_engine.chat(q)
        print(f"\nAssistant: {resp}\n")

        sources = _extract_sources(resp)
        if sources:
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

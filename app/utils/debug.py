from typing import List
from collections import Counter

from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.schema import BaseNode


# -----------------------------
# Chunk statistics
# -----------------------------
def print_chunk_counts(
    big_nodes: List[BaseNode],
    mid_nodes: List[BaseNode],
    small_nodes: List[BaseNode],
    all_nodes: List[BaseNode] | None = None,
):
    """
    Print chunk counts for hierarchical chunking.
    Optionally includes diagram node counts if all_nodes is provided.
    """
    print("Chunk counts:")
    print(f"  big      : {len(big_nodes)}")
    print(f"  mid      : {len(mid_nodes)}")
    print(f"  small    : {len(small_nodes)}")

    if all_nodes is not None:
        diagram_nodes = [
            n for n in all_nodes
            if n.metadata.get("chunk_level") == "diagram"
        ]
        print(f"  diagram  : {len(diagram_nodes)}")

    print()


# -----------------------------
# Preview nodes
# -----------------------------
def preview_nodes(nodes: List[BaseNode], label: str, n: int = 2):
    """
    Preview the first N nodes with character counts.
    """
    print(f"--- {label} CHUNK PREVIEW ---")

    for i, node in enumerate(nodes[:n], 1):
        text = node.get_content()
        print(f"[{label} #{i}] chars={len(text)}")
        print(
            text[:400].replace("\n", " ")
            + ("..." if len(text) > 400 else "")
        )
        print()


def preview_diagram_nodes(nodes: List[BaseNode], n: int = 2):
    """
    Preview diagram nodes with metadata.
    """
    diagrams = [
        n for n in nodes
        if n.metadata.get("chunk_level") == "diagram"
    ]

    print("--- DIAGRAM NODE PREVIEW ---")

    if not diagrams:
        print("No diagram nodes found.\n")
        return

    for i, node in enumerate(diagrams[:n], 1):
        meta = node.metadata
        print(
            f"[DIAGRAM #{i}] page={meta.get('page_number')} file={meta.get('file_name')}")
        print(f"  diagram_type : {meta.get('diagram_type', 'unknown')}")
        print(f"  manual_title : {meta.get('manual_title')}")
        print(
            node.get_content()[:300].replace("\n", " ")
            + ("..." if len(node.get_content()) > 300 else "")
        )
        print()


# -----------------------------
# Diagram statistics
# -----------------------------
def print_diagram_stats(nodes: List[BaseNode]):
    """
    Print statistics about diagram nodes.
    """
    diagrams = [
        n for n in nodes
        if n.metadata.get("chunk_level") == "diagram"
    ]

    print("--- DIAGRAM STATS ---")

    if not diagrams:
        print("No diagram nodes detected.\n")
        return

    types = Counter(
        n.metadata.get("diagram_type", "unknown")
        for n in diagrams
    )

    print(f"Total diagram nodes: {len(diagrams)}")
    for t, count in types.items():
        print(f"  {t:<12}: {count}")

    print()


# -----------------------------
# Retrieval debug
# -----------------------------
def debug_retrieval(
    index: VectorStoreIndex,
    query: str,
    top_k: int = 8,
):
    """
    Debug vector retrieval results with chunk level visibility.
    """
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(QueryBundle(query))

    print("\n--- RETRIEVAL DEBUG ---")
    print(f"Query: {query}\n")

    for i, r in enumerate(results, 1):
        node = r.node
        level = node.metadata.get("chunk_level", "unknown")
        page = node.metadata.get("page_number")
        file = node.metadata.get("file_name")
        dtype = node.metadata.get("diagram_type")

        snippet = node.get_content()[:160].replace("\n", " ")

        extra = f" diagram={dtype}" if level == "diagram" else ""
        print(
            f"{i}. level={level:<7} score={r.score:.4f} "
            f"page={page} file={file}{extra}"
        )
        print(f"   {snippet}...")
    print()

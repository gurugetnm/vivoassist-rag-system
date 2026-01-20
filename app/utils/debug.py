from typing import List
from llama_index.core import QueryBundle, VectorStoreIndex


def print_chunk_counts(big_nodes: List, mid_nodes: List, small_nodes: List):
    print("Chunk counts:")
    print(f"  big   : {len(big_nodes)}")
    print(f"  mid   : {len(mid_nodes)}")
    print(f"  small : {len(small_nodes)}\n")


def preview_nodes(nodes: List, label: str, n: int = 2):
    print(f"--- {label} CHUNK PREVIEW ---")
    for i, node in enumerate(nodes[:n], 1):
        text = node.get_content()
        print(f"[{label} #{i}] chars={len(text)}")
        print(text[:400].replace("\n", " ") + ("..." if len(text) > 400 else ""))
        print()


def debug_retrieval(index: VectorStoreIndex, query: str, top_k: int = 8):
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(QueryBundle(query))

    print("\n--- RETRIEVAL DEBUG ---")
    for i, r in enumerate(results, 1):
        level = r.node.metadata.get("chunk_level", "unknown")
        snippet = r.node.get_content()[:160].replace("\n", " ")
        print(f"{i}. level={level:5} score={r.score:.4f}  {snippet}...")
    print()

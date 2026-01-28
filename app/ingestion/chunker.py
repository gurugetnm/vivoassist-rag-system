from typing import List, Tuple
import json

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, BaseNode, TextNode

from app.ingestion.diagram_extractor import (
    extract_diagram_metadata,
    build_diagram_summary,
)


def hierarchical_chunk(
    documents: List[Document],
    *,
    big_size: int,
    big_overlap: int,
    mid_size: int,
    mid_overlap: int,
    small_size: int,
    small_overlap: int,
) -> Tuple[List[BaseNode], List[BaseNode], List[BaseNode], List[BaseNode]]:
    """
    Creates hierarchical chunks with diagram awareness.

    - Text pages → big / mid / small sentence-based chunks
    - Diagram pages → 1 semantic node per page (no sentence splitting)

    Returns:
      (all_nodes, big_nodes, mid_nodes, small_nodes)

    Diagram nodes are included in all_nodes only
    and tagged with chunk_level="diagram".
    """

    # -----------------------------
    # Split documents by content type
    # -----------------------------
    text_docs: List[Document] = []
    diagram_docs: List[Document] = []

    for d in documents:
        if d.metadata.get("content_type") == "diagram":
            diagram_docs.append(d)
        else:
            text_docs.append(d)

    # -----------------------------
    # Sentence splitters for TEXT pages
    # -----------------------------
    splitter_big = SentenceSplitter(chunk_size=big_size, chunk_overlap=big_overlap)
    splitter_mid = SentenceSplitter(chunk_size=mid_size, chunk_overlap=mid_overlap)
    splitter_small = SentenceSplitter(chunk_size=small_size, chunk_overlap=small_overlap)

    def split_and_tag(
        splitter: SentenceSplitter,
        docs: List[Document],
        level: str,
    ) -> List[BaseNode]:
        nodes = splitter.get_nodes_from_documents(docs)
        for n in nodes:
            n.metadata["chunk_level"] = level
        return nodes

    # -----------------------------
    # Text chunks (unchanged behavior)
    # -----------------------------
    nodes_big = split_and_tag(splitter_big, text_docs, "big")
    nodes_mid = split_and_tag(splitter_mid, text_docs, "mid")
    nodes_small = split_and_tag(splitter_small, text_docs, "small")

    # -----------------------------
    # Diagram nodes (1 node per page)
    # -----------------------------
    diagram_nodes: List[BaseNode] = []

    for d in diagram_docs:
        diagram_meta = extract_diagram_metadata(d.text)

        # ✅ Chroma requires metadata to be FLAT (no lists/dicts).
        # Convert list fields -> JSON strings.
        diagram_meta_flat = {
            k: (json.dumps(v) if isinstance(v, list) else v)
            for k, v in diagram_meta.items()
        }

        node = TextNode(
            text=d.text,
            metadata={
                **d.metadata,
                "chunk_level": "diagram",
                **diagram_meta_flat,
                "diagram_summary": build_diagram_summary(diagram_meta),
            },
        )
        diagram_nodes.append(node)

    # -----------------------------
    # Combine all nodes
    # -----------------------------
    all_nodes: List[BaseNode] = nodes_big + nodes_mid + nodes_small + diagram_nodes

    return all_nodes, nodes_big, nodes_mid, nodes_small

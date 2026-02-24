from typing import List, Tuple
import json

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import (
    Document,
    BaseNode,
    TextNode,
    NodeRelationship,
    RelatedNodeInfo,
)

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
    True hierarchical chunking with diagram awareness.

    - Text pages:
        big chunks (parents)
          -> mid chunks (children)
              -> small chunks (leaf children)
      Relationships are attached using NodeRelationship.PARENT / NodeRelationship.CHILD.

    - Diagram pages:
        1 node per page (no sentence splitting), tagged chunk_level="diagram"
        plus extracted diagram metadata and diagram_summary.

    Returns:
      (all_nodes, big_nodes, mid_nodes, small_nodes)
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
    # Sentence splitters
    # -----------------------------
    splitter_big = SentenceSplitter(chunk_size=big_size, chunk_overlap=big_overlap)
    splitter_mid = SentenceSplitter(chunk_size=mid_size, chunk_overlap=mid_overlap)
    splitter_small = SentenceSplitter(chunk_size=small_size, chunk_overlap=small_overlap)

    # -----------------------------
    # Helper: ensure CHILD relationship is a list
    # -----------------------------
    def _ensure_child_list(node: BaseNode) -> List[RelatedNodeInfo]:
        existing = node.relationships.get(NodeRelationship.CHILD)
        if existing is None:
            node.relationships[NodeRelationship.CHILD] = []
            return node.relationships[NodeRelationship.CHILD]
        # Some versions may store a single RelatedNodeInfo; normalize to list
        if isinstance(existing, RelatedNodeInfo):
            node.relationships[NodeRelationship.CHILD] = [existing]
            return node.relationships[NodeRelationship.CHILD]
        return existing  # assume it's already a list

    # -----------------------------
    # 1) BIG nodes (top level)
    # -----------------------------
    big_nodes: List[BaseNode] = splitter_big.get_nodes_from_documents(text_docs)
    for n in big_nodes:
        n.metadata["chunk_level"] = "big"
        n.metadata["hierarchy_depth"] = 0

    # -----------------------------
    # 2) MID nodes (children of BIG)
    # -----------------------------
    mid_nodes: List[BaseNode] = []
    for big in big_nodes:
        # Create a temporary "document" from the big chunk so we can split it further.
        # Keep metadata from big so page/filename/etc stay consistent.
        tmp_doc = Document(text=big.get_content(), metadata=dict(big.metadata))

        mids = splitter_mid.get_nodes_from_documents([tmp_doc])
        for mid in mids:
            mid.metadata["chunk_level"] = "mid"
            mid.metadata["hierarchy_depth"] = 1

            # link mid -> big (parent)
            mid.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=big.node_id)

            # link big -> mid (child)
            child_list = _ensure_child_list(big)
            child_list.append(RelatedNodeInfo(node_id=mid.node_id))

        mid_nodes.extend(mids)

    # -----------------------------
    # 3) SMALL nodes (children of MID)
    # -----------------------------
    small_nodes: List[BaseNode] = []
    for mid in mid_nodes:
        tmp_doc = Document(text=mid.get_content(), metadata=dict(mid.metadata))

        smalls = splitter_small.get_nodes_from_documents([tmp_doc])
        for sm in smalls:
            sm.metadata["chunk_level"] = "small"
            sm.metadata["hierarchy_depth"] = 2

            # link small -> mid (parent)
            sm.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=mid.node_id)

            # link mid -> small (child)
            child_list = _ensure_child_list(mid)
            child_list.append(RelatedNodeInfo(node_id=sm.node_id))

        small_nodes.extend(smalls)

    # -----------------------------
    # 4) Diagram nodes (1 node per page)
    # -----------------------------
    diagram_nodes: List[BaseNode] = []
    for d in diagram_docs:
        diagram_meta = extract_diagram_metadata(d.text)

        diagram_meta_flat = {
            k: (json.dumps(v) if isinstance(v, list) else v)
            for k, v in diagram_meta.items()
        }

        node = TextNode(
            text=d.text,
            metadata={
                **d.metadata,
                "chunk_level": "diagram",
                "hierarchy_depth": 0,  # standalone
                **diagram_meta_flat,
                "diagram_summary": build_diagram_summary(diagram_meta),
            },
        )
        diagram_nodes.append(node)

    # -----------------------------
    # Combine all nodes
    # -----------------------------
    all_nodes: List[BaseNode] = big_nodes + mid_nodes + small_nodes + diagram_nodes

    return all_nodes, big_nodes, mid_nodes, small_nodes
from typing import List, Tuple
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, BaseNode


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
    Creates 3 levels of chunks (big/mid/small) and returns:
      (all_nodes, big_nodes, mid_nodes, small_nodes)
    Each node is tagged with node.metadata["chunk_level"].
    """
    splitter_big = SentenceSplitter(chunk_size=big_size, chunk_overlap=big_overlap)
    splitter_mid = SentenceSplitter(chunk_size=mid_size, chunk_overlap=mid_overlap)
    splitter_small = SentenceSplitter(chunk_size=small_size, chunk_overlap=small_overlap)

    def split_and_tag(splitter: SentenceSplitter, level: str) -> List[BaseNode]:
        nodes = splitter.get_nodes_from_documents(documents)
        for n in nodes:
            n.metadata["chunk_level"] = level
        return nodes

    nodes_big = split_and_tag(splitter_big, "big")
    nodes_mid = split_and_tag(splitter_mid, "mid")
    nodes_small = split_and_tag(splitter_small, "small")

    all_nodes = nodes_big + nodes_mid + nodes_small
    return all_nodes, nodes_big, nodes_mid, nodes_small

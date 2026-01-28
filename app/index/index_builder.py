import time
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore


def build_and_persist_index(
    nodes,
    vector_store: ChromaVectorStore,
    *,
    throttle_every: int = 10,  
    throttle_sleep: float = 1.5 
) -> VectorStoreIndex:
    """
    Build a VectorStoreIndex from nodes and persist into Chroma,
    with simple throttling to reduce Azure 429 rate limits.

    We build/insert in batches and sleep between batches.
    """
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = None
    batch = []

    for i, node in enumerate(nodes, start=1):
        batch.append(node)

        if i % throttle_every == 0:
            if index is None:
                index = VectorStoreIndex(batch, storage_context=storage_context)
            else:
                index.insert_nodes(batch)

            batch.clear()
            time.sleep(throttle_sleep)

    if batch:
        if index is None:
            index = VectorStoreIndex(batch, storage_context=storage_context)
        else:
            index.insert_nodes(batch)

    return index


def load_index_from_chroma(vector_store: ChromaVectorStore) -> VectorStoreIndex:
    """
    Load an index wrapper that uses existing vectors/documents already stored in Chroma.
    """
    return VectorStoreIndex.from_vector_store(vector_store)

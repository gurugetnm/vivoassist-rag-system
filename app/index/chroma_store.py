import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore


def get_chroma_vector_store(chroma_dir: str, collection_name: str):
    """
    Creates/opens a persistent Chroma DB at `chroma_dir`,
    and returns (vector_store, collection).
    """
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return vector_store, collection

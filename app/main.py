import argparse
import shutil
from pathlib import Path

from app.config.settings import load_config, configure_llamaindex
from app.ingestion.pdf_loader import load_pdfs
from app.ingestion.chunker import hierarchical_chunk
from app.index.chroma_store import get_chroma_vector_store
from app.index.index_builder import build_and_persist_index, load_index_from_chroma
from app.utils.debug import print_chunk_counts, preview_nodes
from app.chat.chat_engine import run_terminal_chat


def main():
    # -----------------------------
    # CLI arguments
    # -----------------------------
    parser = argparse.ArgumentParser(description="VivoAssist RAG Assistant")

    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Delete ChromaDB and rebuild the vector index",
    )

    args = parser.parse_args()

    # -----------------------------
    # Load config + configure LlamaIndex (Azure)
    # -----------------------------
    cfg = load_config()
    configure_llamaindex()

    # -----------------------------
    # Handle rebuild flags
    # -----------------------------
    if args.rebuild_index:
        chroma_path = Path(cfg.chroma_dir)
        if chroma_path.exists():
            print("🔄 Rebuilding vector index (deleting ChromaDB)...")
            shutil.rmtree(chroma_path)

    # -----------------------------
    # Chroma setup (persistent on disk)
    # -----------------------------
    vector_store, collection = get_chroma_vector_store(
        cfg.chroma_dir,
        cfg.chroma_collection,
    )

    # -----------------------------
    # Build or load index
    # -----------------------------
    existing = collection.count()

    if existing > 0:
        print(
            f"Chroma already has {existing} items. Loading index from disk...\n")
        index = load_index_from_chroma(vector_store)

    else:
        print("Chroma collection is empty. Building index (first run)...\n")

        docs = load_pdfs(cfg.data_dir)
        print("\n--- LOADED FILES ---")
        files = sorted({d.metadata.get("file_name", "unknown") for d in docs})
        for f in files:
            print(" -", f)
        print("Total docs/pages loaded:", len(docs), "\n")

        all_nodes, big_nodes, mid_nodes, small_nodes = hierarchical_chunk(
            docs,
            big_size=cfg.big_chunk_size,
            big_overlap=cfg.big_chunk_overlap,
            mid_size=cfg.mid_chunk_size,
            mid_overlap=cfg.mid_chunk_overlap,
            small_size=cfg.small_chunk_size,
            small_overlap=cfg.small_chunk_overlap,
        )

        if cfg.debug:
            print_chunk_counts(big_nodes, mid_nodes, small_nodes)
            preview_nodes(big_nodes, "BIG", n=2)
            preview_nodes(mid_nodes, "MID", n=2)
            preview_nodes(small_nodes, "SMALL", n=2)

        index = build_and_persist_index(all_nodes, vector_store)

        print(
            f"Saved to Chroma at: {cfg.chroma_dir} "
            f"(collection: {cfg.chroma_collection})\n"
        )

    # -----------------------------
    # Chat
    # -----------------------------
    run_terminal_chat(index, top_k=cfg.top_k, debug=cfg.debug)


if __name__ == "__main__":
    main()

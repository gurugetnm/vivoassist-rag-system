# streamlit_app.py
from __future__ import annotations

import os
from pathlib import Path
import streamlit as st

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore

from app.config.settings import load_config, configure_llamaindex
from app.index.chroma_store import get_chroma_vector_store
from app.index.index_builder import load_index_from_chroma
from app.utils.models_registry import load_models_cache, build_models_cache

from app.chat.chat_engine import (
    _build_engine,
    _extract_sources,
    best_manual_match_with_score,
    NOT_FOUND,
    PDF_BASE_URL,
    FRIENDLY_INTRO,
)

# If your project has these (based on your earlier codebase style)
# If not, remove and replace with your own PDF loader logic.
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import Document

st.set_page_config(page_title="VivoAssist Demo", page_icon="📘", layout="wide")


def sources_markdown(resp) -> str:
    sources = _extract_sources(resp)
    if not sources:
        return ""

    grouped = {}
    for f, p in sources:
        if not f or not p:
            continue
        grouped.setdefault(f, set()).add(str(p))

    lines = []
    for f, pages in grouped.items():

        def _key(x: str):
            return int(x) if x.isdigit() else 10**9

        pages_sorted = sorted(pages, key=_key)

        lines.append(f"- **{f}** (pages: {', '.join(pages_sorted)})")
        for p in pages_sorted:
            lines.append(f"  - {PDF_BASE_URL}/{f}#page={p}")

    return "\n".join(lines)


def get_engine(index, top_k: int, manual_id: str | None):
    engine_cache = st.session_state.setdefault("engine_cache", {})
    key = (manual_id, int(top_k))
    if key not in engine_cache:
        engine_cache[key] = _build_engine(index, top_k=int(top_k), manual_id=manual_id)
    return engine_cache[key]


@st.cache_resource
def load_all():
    cfg = load_config()
    configure_llamaindex()

    vector_store, collection = get_chroma_vector_store(
        cfg.chroma_dir, cfg.chroma_collection
    )

    # If empty, return "no index"
    if collection.count() <= 0:
        return cfg, None, None, []

    index = load_index_from_chroma(vector_store)

    cache_path = f"{cfg.chroma_dir}/models_cache.json"
    models_cache = load_models_cache(cache_path)
    if not models_cache:
        models_cache = build_models_cache(
            index=index,
            data_dir=cfg.data_dir,
            cache_path=cache_path,
            per_manual_top_k=40,
        )

    manuals = sorted((models_cache or {}).keys())
    return cfg, index, models_cache, manuals


def _save_uploaded_pdfs(uploaded_files, target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in uploaded_files:
        p = target_dir / f.name
        p.write_bytes(f.read())
        saved.append(p)
    return saved


def _load_pdfs_as_documents(pdf_paths: list[Path]) -> list[Document]:
    reader = PyMuPDFReader()
    docs: list[Document] = []

    for p in pdf_paths:
        pages = reader.load(file_path=p)
        text = "\n\n".join(pg.get_content() for pg in pages)
        docs.append(
            Document(
                text=text,
                metadata={"filename": p.name, "source": str(p), "page_count": len(pages)},
            )
        )
    return docs


def build_index_in_streamlit(cfg) -> None:
    """
    Build Chroma index inside Streamlit.
    - Uses cfg.data_dir as PDF storage
    - Writes embeddings into cfg.chroma_dir / cfg.chroma_collection
    - Builds models_cache.json
    """
    configure_llamaindex()

    vector_store, collection = get_chroma_vector_store(cfg.chroma_dir, cfg.chroma_collection)

    # Load PDFs from cfg.data_dir
    pdf_dir = Path(cfg.data_dir)
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"No PDFs found in {pdf_dir}. Upload PDFs first.")

    # Load documents
    docs = _load_pdfs_as_documents(pdf_paths)

    # Build index (writes into Chroma via vector_store)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # LlamaIndex will chunk internally (unless you configured a node parser in configure_llamaindex)
    VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        show_progress=False,
    )

    # Now build models cache used by your manual matching logic
    index = load_index_from_chroma(vector_store)
    cache_path = f"{cfg.chroma_dir}/models_cache.json"
    build_models_cache(
        index=index,
        data_dir=cfg.data_dir,
        cache_path=cache_path,
        per_manual_top_k=40,
    )


def main():
    cfg, index, models_cache, manuals = load_all()

    # Session defaults
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("sticky_manual", None)
    st.session_state.setdefault("debug", getattr(cfg, "debug", False))
    st.session_state.setdefault("top_k", getattr(cfg, "top_k", 8))

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.title("📘 VivoAssist")
        st.caption("Answers only from your uploaded manuals (with page links).")

        st.toggle("Debug", key="debug")
        st.slider("top_k", 3, 20, key="top_k")

        st.divider()
        st.subheader("Upload manuals (PDF)")
        uploads = st.file_uploader(
            "Upload PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save PDFs", use_container_width=True):
                if not uploads:
                    st.error("Upload at least one PDF.")
                else:
                    saved = _save_uploaded_pdfs(uploads, Path(cfg.data_dir))
                    st.success(f"Saved {len(saved)} PDF(s) to {cfg.data_dir}")

        with c2:
            if st.button("⚙️ Build Index", use_container_width=True, type="primary"):
                try:
                    with st.spinner("Building index (Chroma) ..."):
                        build_index_in_streamlit(cfg)

                    # Clear caches so load_all() re-reads new data
                    load_all.clear()
                    st.session_state["engine_cache"] = {}
                    st.success("Index built ✅ Reloading...")
                    st.rerun()
                except Exception as e:
                    st.error(f"Build failed: {type(e).__name__}: {e}")

        st.divider()

        # If no index yet, show quick hint
        if index is None:
            st.warning("No index yet. Upload PDFs → Save PDFs → Build Index.")
            st.stop()

        st.subheader("Manual scope")

        current = st.session_state["sticky_manual"]
        selected = st.selectbox(
            "Select manual",
            options=[None] + manuals,
            index=([None] + manuals).index(current) if current in manuals else 0,
            format_func=lambda x: "(no lock)" if x is None else x,
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔒 Lock", use_container_width=True):
                st.session_state["sticky_manual"] = selected
        with c2:
            if st.button("🔓 Unlock", use_container_width=True):
                st.session_state["sticky_manual"] = None

        st.divider()
        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["engine_cache"] = {}

        st.divider()
        st.subheader("Manuals available")
        for m in manuals[:60]:
            st.write("•", m)
        if len(manuals) > 60:
            st.caption(f"...and {len(manuals) - 60} more")

    # ---------------- Main ----------------
    st.header("Chat")
    st.markdown(FRIENDLY_INTRO)

    lock = st.session_state["sticky_manual"]
    if lock:
        st.info(f"🔒 Locked to: **{lock}**")
    else:
        st.caption("No manual lock. I’ll auto-pick a manual if confidence is high.")

    # Render previous messages FIRST
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources_md"):
                with st.expander("Sources"):
                    st.markdown(msg["sources_md"])

    raw = st.chat_input("Ask a question…")
    if raw is None:
        return

    q = (raw or "").strip()
    if not q:
        st.warning("Type a question first.")
        return

    with st.chat_message("user"):
        st.markdown(q)
    st.session_state["messages"].append({"role": "user", "content": q})

    # Decide active manual
    active_manual = lock
    if not active_manual:
        matched, score = best_manual_match_with_score(q, manuals)
        if matched and score >= 0.72:
            active_manual = matched
            if st.session_state["debug"]:
                st.toast(f"Auto-selected: {matched} ({score:.2f})", icon="✅")
        elif matched and score >= 0.55 and st.session_state["debug"]:
            st.toast(
                f"Possible: {matched} ({score:.2f}) — lock it in sidebar", icon="💡"
            )

    if st.session_state["debug"]:
        st.caption(f"ACTIVE MANUAL: {active_manual or '(none)'}")

    engine = get_engine(index, top_k=int(st.session_state["top_k"]), manual_id=active_manual)

    with st.chat_message("assistant"):
        try:
            resp = engine.chat(q)
            text = str(resp).strip() or NOT_FOUND
        except Exception as e:
            text = f"⚠️ Error: {type(e).__name__}: {e}"
            resp = None

        st.markdown(text)

        sources_md = ""
        if resp is not None and NOT_FOUND.lower() not in text.lower():
            sources_md = sources_markdown(resp)
            if sources_md:
                with st.expander("Sources"):
                    st.markdown(sources_md)

    st.session_state["messages"].append(
        {"role": "assistant", "content": text, "sources_md": sources_md}
    )


if __name__ == "__main__":
    main()

# streamlit_app.py
from __future__ import annotations

import streamlit as st

from app.config.settings import load_config, configure_llamaindex
from app.index.chroma_store import get_chroma_vector_store
from app.index.index_builder import load_index_from_chroma
from app.utils.models_registry import load_models_cache, build_models_cache

# Reuse your project helpers
from app.chat.chat_engine import (
    _build_engine,
    _extract_sources,
    best_manual_match_with_score,
    NOT_FOUND,
    PDF_BASE_URL,
    FRIENDLY_INTRO,
)

st.set_page_config(page_title="VivoAssist Demo", page_icon="📘", layout="wide")


@st.cache_resource
def load_all():
    cfg = load_config()
    configure_llamaindex()

    vector_store, collection = get_chroma_vector_store(
        cfg.chroma_dir, cfg.chroma_collection
    )

    # ✅ Cloud-safe: don't crash if DB is empty
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
    # Keep engines cached per (manual_id, top_k) so chat memory is stable
    engine_cache = st.session_state.setdefault("engine_cache", {})
    key = (manual_id, int(top_k))

    if key not in engine_cache:
        engine_cache[key] = _build_engine(index, top_k=int(top_k), manual_id=manual_id)

    return engine_cache[key]


def main():
    cfg, index, models_cache, manuals = load_all()

    # ✅ If no index yet, show friendly instructions (don’t crash)
    if index is None:
        st.title("📘 VivoAssist Demo")
        st.warning("Chroma collection is empty — no index found.")
        st.markdown(
            """
### What to do next

This deployment starts with an empty database.

**Option A (recommended for demo):**
- Run indexing locally:
  - `python -m app.main`
- Then deploy with a persistent Chroma directory (Render/VPS), **or** commit a small prebuilt index (not ideal).

**Option B (Cloud-friendly):**
- Add a Streamlit “Build Index” button that uploads manuals and builds Chroma inside the app.

If you want, tell me which option you want and I’ll wire the “Build Index” button into this UI.
"""
        )
        st.stop()

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
            # optional: also reset engines to clear memory
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

    # Input
    raw = st.chat_input("Ask a question…")

    # If user didn’t submit, stop here
    if raw is None:
        return

    q = (raw or "").strip()
    if not q:
        st.warning("Type a question first.")
        return

    # Show user message immediately
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

    # Chat with engine
    engine = get_engine(
        index, top_k=int(st.session_state["top_k"]), manual_id=active_manual
    )

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

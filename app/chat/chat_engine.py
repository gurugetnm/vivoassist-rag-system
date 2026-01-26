from __future__ import annotations

import re
from difflib import SequenceMatcher
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

# LlamaIndex chat engine (Condense+Context)
try:
    # Newer layout
    from llama_index.core.chat_engine import CondensePlusContextChatEngine
except Exception:
    # Fallback for some versions
    from llama_index.core.chat_engine.condense_plus_context import (  # type: ignore
        CondensePlusContextChatEngine,
    )


# =========================================================
# SYSTEM CONFIG
# =========================================================

SYSTEM_PROMPT = """
You are a technical assistant for VivoAssist.

Rules:
- Answer ONLY using the provided PDF manual content.
- If the answer is not explicitly found in the manual, say exactly:
  "Not found in the manual."
- Do NOT guess or use external knowledge.
- Keep answers clear, concise, and technical.
- When possible, support answers with page numbers.
""".strip()

NOT_FOUND = "Not found in the manual."
PDF_BASE_URL = "http://localhost:8000/data/manuals"

# Confidence thresholds for auto manual selection
AUTO_LOCK_THRESHOLD = 0.72   # strong match → auto-select
SUGGEST_THRESHOLD = 0.55     # weak match → suggest only


# =========================================================
# FRIENDLY TEXT
# =========================================================

FRIENDLY_INTRO = (
    "Hi! I’m **VivoAssist** 👋\n"
    "I answer questions **only from the manuals you uploaded**.\n\n"
    "Try:\n"
    "- `list manuals`\n"
    "- `use gmdss`\n"
    "- `use starlink`\n"
    "- `unlock`\n"
)


# =========================================================
# SOURCE HANDLING
# =========================================================

def _extract_sources(resp) -> List[Tuple[str, Optional[str]]]:
    nodes = getattr(resp, "source_nodes", None) or getattr(resp, "sources", None)
    if not nodes:
        return []

    out: List[Tuple[str, Optional[str]]] = []
    for sn in nodes:
        node = getattr(sn, "node", sn)
        meta = getattr(node, "metadata", {}) or {}
        out.append(
            (
                meta.get("file_name", "unknown"),
                meta.get("page_label") or meta.get("page_number") or meta.get("page"),
            )
        )
    return out


def _print_sources_with_links(sources):
    grouped = defaultdict(set)
    for f, p in sources:
        if p:
            grouped[f].add(str(p))

    if not grouped:
        return

    print("Sources:")
    for f, pages in grouped.items():
        pages = sorted(pages, key=lambda x: int(x) if x.isdigit() else x)
        print(f"- {f} (pages: {', '.join(pages)})")
        for p in pages:
            print(f"  • {PDF_BASE_URL}/{f}#page={p}")
    print()


# =========================================================
# MANUAL MATCHING (WITH CONFIDENCE)
# =========================================================

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def best_manual_match_with_score(q: str, manuals: List[str]) -> Tuple[Optional[str], float]:
    qn = _normalize(q)
    if not qn:
        return None, 0.0

    q_tokens = [t for t in qn.split() if len(t) >= 4]  # >=4 reduces noise like "iom"
    if not q_tokens:
        q_tokens = qn.split()

    best_manual = None
    best_score = 0.0

    for m in manuals:
        mn = _normalize(m)

        # 1) overall similarity
        s_full = _similar(qn, mn)

        # 2) token coverage: average of best matches per query token
        m_tokens = [t for t in mn.split() if len(t) >= 3]
        if not m_tokens:
            m_tokens = mn.split()

        per_token_best = []
        for qt in q_tokens:
            best_t = 0.0
            for mt in m_tokens:
                best_t = max(best_t, _similar(qt, mt))
            per_token_best.append(best_t)

        s_tokens_avg = sum(per_token_best) / max(len(per_token_best), 1)

        # 3) hard bonus when a strong unique token appears literally in filename
        literal_bonus = 0.0
        for qt in q_tokens:
            if qt in mn and len(qt) >= 6:   # "starlink" triggers, "system" doesn't
                literal_bonus += 0.08
        literal_bonus = min(literal_bonus, 0.24)

        # Final weighted score (keeps 0..1-ish)
        score = 0.55 * s_full + 0.40 * s_tokens_avg + literal_bonus

        if score > best_score:
            best_score = score
            best_manual = m

    # clamp to 1.0
    best_score = min(best_score, 1.0)
    return best_manual, best_score



# =========================================================
# CONDENSE PROMPT (THIS RESTORES "Condensed question" QUALITY)
# =========================================================

def _make_condense_prompt(manual_name: Optional[str]) -> PromptTemplate:
    """
    Custom prompt used by CondensePlusContextChatEngine to rewrite follow-ups.
    This is the key to getting the "Condensed question" behavior like before.
    """
    if manual_name:
        tmpl = f"""
You rewrite a user's follow-up question into a standalone question.

Context:
- The user is asking about the PDF manual: "{manual_name}"
- Make the standalone question explicitly refer to that manual.
- Keep it short and factual.
- Do NOT answer the question.
- Do NOT add new information.

Chat History:
{{chat_history}}

Follow-up Question:
{{question}}

Standalone question:
""".strip()
    else:
        tmpl = """
You rewrite a user's follow-up question into a standalone question.

Rules:
- Keep it short and factual.
- Do NOT answer the question.
- Do NOT add new information.

Chat History:
{chat_history}

Follow-up Question:
{question}

Standalone question:
""".strip()

    return PromptTemplate(tmpl)


def _build_engine(
    index: VectorStoreIndex,
    *,
    top_k: int,
    manual_id: Optional[str] = None,
) -> CondensePlusContextChatEngine:
    """
    Build a Condense+Context engine with:
    - optional manual filter at retriever level
    - custom condense prompt (manual-aware)
    """
    if manual_id:
        filters = MetadataFilters(filters=[MetadataFilter(key="manual_id", value=manual_id)])
    else:
        filters = None

    retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)

    return CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        llm=Settings.llm,
        system_prompt=SYSTEM_PROMPT,
        condense_prompt=_make_condense_prompt(manual_id),
    )


# =========================================================
# CHAT ENGINE (TERMINAL)
# =========================================================

def run_terminal_chat(
    index: VectorStoreIndex,
    *,
    top_k: int,
    debug: bool,
    data_dir: str,
    models_cache: dict,
    manual_id: Optional[str] = None,
):
    manuals: List[str] = sorted((models_cache or {}).keys())

    # Sticky lock ONLY set by CLI manual_id or explicit `use/lock`
    sticky_manual: Optional[str] = manual_id

    # Base engine (no manual filter)
    base_engine = _build_engine(index, top_k=top_k, manual_id=None)

    # Cache engines per manual_id (so chat history stays stable per scope)
    engine_cache: Dict[Optional[str], CondensePlusContextChatEngine] = {None: base_engine}

    print("Chat ready. Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        # ------------------ small talk ------------------
        if q.lower() in {"hi", "hello", "hey", "who are you", "help"}:
            print(f"\nAssistant: {FRIENDLY_INTRO}\n")
            continue

        # ------------------ list manuals ------------------
        if q.lower().startswith("list manuals"):
            print("\nAssistant: Manuals available:\n")
            for m in manuals:
                print(f"- {m}")
            print()
            continue

        # ------------------ unlock ------------------
        if q.lower() in {"unlock", "clear lock"}:
            sticky_manual = None
            print("\nAssistant: Manual lock cleared.\n")
            continue

        # ------------------ explicit lock ------------------
        if q.lower().startswith(("use ", "lock ")):
            target = q.split(maxsplit=1)[1]
            matched, score = best_manual_match_with_score(target, manuals)

            if matched and score >= SUGGEST_THRESHOLD:
                sticky_manual = matched
                print(
                    f"\nAssistant: 🔒 Locked to manual: **{matched}** "
                    f"(confidence {score:.2f})\n"
                )
            else:
                print("\nAssistant: Could not match that manual.\n")
            continue

        # =================================================
        # DECIDE MANUAL FOR THIS QUESTION
        # =================================================
        active_manual: Optional[str] = sticky_manual

        if not sticky_manual:
            matched, score = best_manual_match_with_score(q, manuals)

            if matched and score >= AUTO_LOCK_THRESHOLD:
                active_manual = matched
                if debug:
                    print(
                        f"\nAssistant: Using **{matched}** "
                        f"(confidence {score:.2f}) for this question.\n"
                    )
            elif matched and score >= SUGGEST_THRESHOLD:
                if debug:
                    print(
                        f"\nAssistant: Possible manual match: **{matched}** "
                        f"(confidence {score:.2f})."
                    )
                    print("Assistant: Tip: use `use <manual>` to lock it.\n")

        # =================================================
        # GET / BUILD ENGINE FOR THIS MANUAL
        # =================================================
        if active_manual not in engine_cache:
            engine_cache[active_manual] = _build_engine(
                index,
                top_k=top_k,
                manual_id=active_manual,
            )
        engine = engine_cache[active_manual]

        # NOTE:
        # We DO NOT need to append "Context: ..." into the user text anymore,
        # because the condense prompt + retriever filter already enforce scope.
        # (Leaving the raw question improves condensation quality.)
        if debug:
            print("✅ ACTIVE MANUAL:", active_manual or "(none)")
            print("✅ RAW USER QUESTION:", q)

        # ------------------ RAG ------------------
        resp = engine.chat(q)

        text = str(resp).strip()
        print(f"\nAssistant: {text}\n")

        if NOT_FOUND.lower() in text.lower():
            print("Assistant: Try asking something that exists in the manual.\n")
            continue

        sources = _extract_sources(resp)
        _print_sources_with_links(sources)

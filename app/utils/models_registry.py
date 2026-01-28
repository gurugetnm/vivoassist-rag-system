from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llama_index.core import VectorStoreIndex

# LlamaIndex filter imports (version-safe)
try:
    from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
except Exception:
    from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter


NOT_FOUND = "Not found in the manual."


# =========================================================
# Helpers
# =========================================================

def _extract_sources(resp) -> List[Tuple[str, Optional[str]]]:
    srcs = getattr(resp, "source_nodes", None) or getattr(resp, "sources", None) or []
    out: List[Tuple[str, Optional[str]]] = []
    for sn in srcs:
        node = getattr(sn, "node", sn)
        meta = getattr(node, "metadata", {}) or {}
        file_name = meta.get("file_name", "unknown_file")
        page = meta.get("page_label") or meta.get("page_number") or meta.get("page")
        out.append((file_name, str(page) if page is not None else None))
    return out


# Generic junk filters (works for vehicle + telco manuals)
_DENY_KEYWORDS = [
    "appendix", "table", "figure", "revision", "rev.", "copyright",
    "all rights reserved", "contents", "index",
    "part number", "p/n", "serial", "firmware", "software version",
    "specification", "specifications", "dimensions",
    "compatible", "compatibility",
]


def _is_valid_subject(name: str) -> bool:
    n = (name or "").strip().lower()
    if not n:
        return False
    if any(bad in n for bad in _DENY_KEYWORDS):
        return False
    # avoid very short junk like "tv", "pc"
    if len(n) < 4:
        return False
    # avoid pure numbers
    if re.fullmatch(r"\d+", n):
        return False
    return True


def _parse_subjects(text: str) -> List[str]:
    """
    Parse a comma/newline separated list of subject/product names.
    """
    t = (text or "").strip()
    if not t or NOT_FOUND.lower() in t.lower():
        return []

    parts = re.split(r",|\n|;|\u2022", t)
    subjects: List[str] = []

    for p in parts:
        s = re.sub(r"^[-•\*]\s*", "", p.strip())
        if not s:
            continue

        # strip common prefixes
        s = re.sub(r"^(model|vehicle|manual|product|system)\s*:\s*", "", s, flags=re.I).strip()
        s = re.sub(r"\s+", " ", s).strip()

        if not s:
            continue

        if _is_valid_subject(s) and s not in subjects:
            subjects.append(s)

    return subjects


def load_models_cache(cache_path: str) -> Dict:
    """
    Backwards compatible: keep name 'models_cache.json' if your app expects it.
    """
    p = Path(cache_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_query(qe, prompt: str, *, max_retries: int = 8, base_sleep: float = 2.0):
    """
    Retry wrapper for Azure 429 + transient errors.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return qe.query(prompt)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            retryable = (
                "429" in msg
                or "rate limit" in msg
                or "too many requests" in msg
                or "throttle" in msg
            )
            if not retryable and attempt >= 2:
                raise

            sleep_s = base_sleep * (2 ** (attempt - 1))
            sleep_s = min(sleep_s, 60.0)
            print(f"[MODELS CACHE] Retry {attempt}/{max_retries} after {sleep_s:.1f}s due to: {e}")
            time.sleep(sleep_s)
    raise last_err


def _title_from_filename(file_name: str) -> str:
    """
    Convert filename to a readable title:
      lancer_2012.pdf -> Lancer 2012
      Telecom System IOM Procedure - Starlink System.pdf -> Telecom System IOM Procedure - Starlink System
    """
    stem = Path(file_name).stem
    s = stem.replace("-", " ").replace("_", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# =========================================================
# Cache Builder
# =========================================================

def build_models_cache(
    index: VectorStoreIndex,
    *,
    data_dir: str,
    cache_path: str,
    per_manual_top_k: int = 60,
    throttle_every: int = 2,
    throttle_sleep: float = 1.5,
) -> Dict:
    """
    Resume-safe incremental cache builder.

    NEW behavior (generic, works for  telco manuals):
    - Tries to extract the PRIMARY product/system this manual is for
    - If nothing explicit found, falls back to filename (marked inferred)
    - Writes cache after each PDF (resume-safe)

    Output format stays compatible with your existing code:
      cache[file_name]["models"] = [{name, pages, inferred}]
    (You can rename later, but this avoids breaking other files.)
    """
    pdfs = sorted(Path(data_dir).glob("*.pdf"))
    cache: Dict = load_models_cache(cache_path) or {}

    for i, pdf in enumerate(pdfs, start=1):
        file_name = pdf.name

        # skip if already cached
        if file_name in cache:
            continue

        print(f"[MODELS CACHE] Scanning {file_name}")

        filters = MetadataFilters(filters=[ExactMatchFilter(key="file_name", value=file_name)])
        qe = index.as_query_engine(similarity_top_k=per_manual_top_k, filters=filters)

        prompt = (
            "You are analyzing a PDF manual.\n\n"
            "Task: Identify ONLY the PRIMARY product/system that this manual is written for.\n"
            'Examples:\n'
            '- "GMDSS System"\n'
            '- "Starlink System"\n'
            '- "Inmarsat FleetBroadband"\n\n'
            "Look specifically at cover/title pages and headings.\n\n"
            "Do NOT return:\n"
            "- tables of contents\n"
            "- section titles\n"
            "- part numbers / serial numbers\n"
            "- firmware/software version strings\n"
            "- compatible devices lists\n\n"
            f"If the primary subject is not explicitly stated, say: {NOT_FOUND}\n\n"
            "Return ONLY the name(s) as a comma-separated list."
        )

        resp = _safe_query(qe, prompt, max_retries=8, base_sleep=2.0)

        txt = str(resp).strip()
        names = _parse_subjects(txt)

        # Collect pages only if we got manual-explicit names
        if names:
            pages = sorted(
                {p for f, p in _extract_sources(resp) if f == file_name and p},
                key=lambda x: int(x) if x.isdigit() else x,
            )
            cache[file_name] = {
                "models": [{"name": n, "pages": pages, "inferred": False} for n in names]
            }
        else:
            inferred_name = _title_from_filename(file_name)
            cache[file_name] = {
                "models": [{"name": f"{inferred_name} (inferred from filename)", "pages": [], "inferred": True}]
            }

        # write after each PDF (resume-safe)
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(cache, indent=2), encoding="utf-8")

        # throttle to reduce 429s
        if i % throttle_every == 0:
            time.sleep(throttle_sleep)

    print(f"[MODELS CACHE] Saved → {cache_path}")
    return cache

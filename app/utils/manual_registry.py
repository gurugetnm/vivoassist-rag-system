from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set


_STOPWORDS = {
    "pdf", "manual", "owner", "owners", "guide", "handbook", "user", "users",
    "edition", "ver", "version", "rev",
}


def _tokenize(text: str) -> List[str]:
    parts = re.split(r"[^a-zA-Z0-9]+", text.lower())
    toks: List[str] = []
    for p in parts:
        if not p:
            continue
        if p in _STOPWORDS:
            continue
        
        if p.isdigit() and len(p) == 4:
            continue
      
        if len(p) < 3:
            continue
        toks.append(p)
    return toks



@dataclass(frozen=True)
class ManualEntry:
    file_name: str          
    stem: str               
    tokens: Set[str]       


def build_manual_registry(data_dir: str) -> Dict[str, ManualEntry]:
    """
    Builds a registry from PDFs in data_dir.
    Keyed by file_name for easy lookup.
    """
    base = Path(data_dir)
    registry: Dict[str, ManualEntry] = {}

    for pdf in sorted(base.glob("*.pdf")):
        stem = pdf.stem
        tokens = set(_tokenize(stem))
        registry[pdf.name] = ManualEntry(
            file_name=pdf.name,
            stem=stem,
            tokens=tokens,
        )

    return registry

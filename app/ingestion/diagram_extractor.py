from __future__ import annotations

import re
from typing import Dict, List


# -----------------------------
# Regex patterns (tuned for telco / marine manuals)
# -----------------------------
PART_NUMBER_PATTERN = re.compile(
    r"\b\d{2}-[A-Z]{2}-\d{3}\b"
)

ALT_PART_NUMBER_PATTERN = re.compile(
    r"\b[A-Z]{2,}-\d{2,}\b"
)

CABLE_PATTERN = re.compile(
    r"\b(CAT\d|FIBER|COAX|ETHERNET)\b.*?\b(\d+\s?m|\d+\s?mm²)?",
    re.IGNORECASE,
)

DEVICE_KEYWORDS = [
    "ANTENNA",
    "SWITCH",
    "ROUTER",
    "MODEM",
    "JUNCTION BOX",
    "JB",
    "CABINET",
    "SERVER",
    "TERMINAL",
    "PANEL",
    "RADAR",
    "CONTROLLER",
]

CONNECTION_WORDS = [
    "TO",
    "FROM",
    "VIA",
    "CONNECT",
    "CONNECTED",
    "LINK",
]


# -----------------------------
# Core extractor
# -----------------------------
def extract_diagram_metadata(text: str) -> Dict[str, List[str]]:
    """
    Extract structured metadata from diagram-heavy text.

    Returns:
        {
          "components": [...],
          "part_numbers": [...],
          "cables": [...],
          "connections": [...]
        }
    """

    if not text:
        return {
            "components": [],
            "part_numbers": [],
            "cables": [],
            "connections": [],
        }

    t = text.upper()

    # -----------------------------
    # Part numbers
    # -----------------------------
    part_numbers = set(PART_NUMBER_PATTERN.findall(t))
    part_numbers.update(ALT_PART_NUMBER_PATTERN.findall(t))

    # -----------------------------
    # Cable information
    # -----------------------------
    cables = set()
    for match in CABLE_PATTERN.findall(t):
        cable_type = match[0]
        length = match[1]
        if length:
            cables.add(f"{cable_type} {length}")
        else:
            cables.add(cable_type)

    # -----------------------------
    # Components / devices
    # -----------------------------
    components = set()
    lines = [l.strip() for l in t.splitlines() if l.strip()]

    for line in lines:
        for kw in DEVICE_KEYWORDS:
            if kw in line:
                # Clean excessive symbols
                cleaned = re.sub(r"[^A-Z0-9 \-]", "", line)
                components.add(cleaned.strip())
                break

    # -----------------------------
    # Connection hints (lightweight)
    # -----------------------------
    connections = set()

    for line in lines:
        if any(w in line for w in CONNECTION_WORDS):
            cleaned = re.sub(r"[^A-Z0-9 \-]", "", line)
            connections.add(cleaned.strip())

    return {
        "components": sorted(components),
        "part_numbers": sorted(part_numbers),
        "cables": sorted(cables),
        "connections": sorted(connections),
    }


# -----------------------------
# Optional: diagram summary helper
# -----------------------------
def build_diagram_summary(metadata: Dict[str, List[str]]) -> str:
    """
    Build a short human-readable summary from extracted metadata.
    Safe to store in node.metadata["diagram_summary"].
    """
    parts = []

    if metadata.get("components"):
        parts.append(
            "Components: " + ", ".join(metadata["components"][:6])
        )

    if metadata.get("cables"):
        parts.append(
            "Cables: " + ", ".join(metadata["cables"])
        )

    if metadata.get("part_numbers"):
        parts.append(
            "Part numbers: " + ", ".join(metadata["part_numbers"])
        )

    if metadata.get("connections"):
        parts.append(
            "Connections: " + ", ".join(metadata["connections"][:4])
        )

    return " | ".join(parts)

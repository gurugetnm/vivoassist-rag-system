from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import os
import re
import shutil
import platform

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.readers.file import PDFReader

# -----------------------------
# Optional OCR deps (graceful)
# -----------------------------
try:
    import fitz  # PyMuPDF
    from PIL import Image
    import pytesseract

    OCR_DEPS_AVAILABLE = True
except Exception:
    fitz = None  # type: ignore
    Image = None  # type: ignore
    pytesseract = None  # type: ignore
    OCR_DEPS_AVAILABLE = False


def _configure_tesseract() -> bool:
    """
    Configure pytesseract in a cross-platform way.

    Priority:
    1) TESSERACT_CMD env var (explicit override; can be a full path or command name)
    2) tesseract in PATH (Linux/Docker/Azure usually)
    3) Windows default install locations fallback

    Returns True if found/configured, else False.
    """
    if not OCR_DEPS_AVAILABLE:
        return False

    assert pytesseract is not None

    # 1) explicit override
    env_cmd = (os.getenv("TESSERACT_CMD") or "").strip()
    if env_cmd:
        # allow either full path or a command name
        if Path(env_cmd).exists():
            pytesseract.pytesseract.tesseract_cmd = env_cmd
            return True
        detected_env = shutil.which(env_cmd)
        if detected_env:
            pytesseract.pytesseract.tesseract_cmd = detected_env
            return True
        return False

    # 2) available in PATH
    detected = shutil.which("tesseract")
    if detected:
        pytesseract.pytesseract.tesseract_cmd = detected
        return True

    # 3) Windows common locations
    if platform.system().lower().startswith("win"):
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for c in candidates:
            if Path(c).exists():
                pytesseract.pytesseract.tesseract_cmd = c
                return True

    return False


HAS_TESSERACT = _configure_tesseract()

# -----------------------------
# Manual metadata helpers
# -----------------------------


def _manual_title_from_filename(file_name: str) -> str:
    return Path(file_name).stem.replace("__", "_").strip()


def _classify_manual_type(title: str) -> str:
    t = (title or "").lower()
    system_keywords = [
        "iom",
        "installation",
        "operation",
        "maintenance",
        "system",
        "equipment",
        "commissioning",
        "configuration",
        "user manual",
        "service manual",
    ]
    if any(k in t for k in system_keywords):
        return "system"
    return "vehicle"


# -----------------------------
# Diagram detection helpers
# -----------------------------
def _is_diagram_page(text: str) -> bool:
    """
    Heuristic diagram detector:
    - many uppercase labels
    - engineering keywords
    - part-number patterns
    """
    if not text:
        return False

    t = text.upper()

    keywords = [
        "DRAWING",
        "DIAGRAM",
        "SCHEMATIC",
        "TERMINATION",
        "LAYOUT",
        "WIRING",
        "CONNECTION",
    ]

    keyword_hits = sum(1 for k in keywords if k in t)
    part_numbers = re.findall(r"\b\d{2}-[A-Z]{2}-\d{3}\b", t)
    cable_hits = re.findall(r"\bCAT\d\b|\bCABLE\b", t)

    lines = [l for l in t.splitlines() if l.strip()]
    uppercase_lines = [l for l in lines if l.isupper() and len(l) > 6]

    score = 0
    if keyword_hits >= 2:
        score += 1
    if len(part_numbers) >= 2:
        score += 1
    if len(cable_hits) >= 2:
        score += 1
    if len(uppercase_lines) >= 4:
        score += 1

    return score >= 2


def _diagram_type(text: str) -> str:
    t = (text or "").upper()
    if "WIRING" in t or "CABLE" in t:
        return "wiring"
    if "TERMINATION" in t:
        return "termination"
    if "LAYOUT" in t:
        return "layout"
    return "diagram"


# -----------------------------
# OCR helpers
# -----------------------------
def _ocr_page(doc: "fitz.Document", page_index: int, dpi: int = 200) -> str:
    assert OCR_DEPS_AVAILABLE and fitz is not None and Image is not None and pytesseract is not None

    page = doc.load_page(page_index)
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return (pytesseract.image_to_string(img) or "").strip()


def _looks_useful(text: str, min_chars: int = 40) -> bool:
    t = (text or "").strip()
    if len(t) < min_chars:
        return False
    alnum = sum(ch.isalnum() for ch in t)
    return alnum / max(len(t), 1) > 0.2


def _needs_ocr(page: "fitz.Page", extracted_text: str, *, min_chars: int = 60, min_images: int = 1) -> bool:
    # If we already got enough text, skip OCR
    if len((extracted_text or "").strip()) >= min_chars:
        return False
    # If page has images and text is weak, OCR likely helps
    try:
        imgs = page.get_images(full=True)
    except Exception:
        imgs = []
    return len(imgs) >= min_images


# -----------------------------
# Main loader
# -----------------------------
def load_pdfs(data_dir: str, *, ocr_dpi: int = 200) -> List[Document]:
    """
    Loads PDFs per-page and applies:
    - manual-level metadata
    - OCR only where needed (embedded text first)
    - diagram detection & tagging

    Returns list[Document] with consistent metadata:
      file_name, page_number, page_label, manual_id, manual_title, manual_type,
      is_ocr, content_type, (diagram_type optional)
    """

    # 1) Embedded text extraction (fast path)
    pdf_reader = PDFReader(return_full_document=False)
    reader = SimpleDirectoryReader(
        input_dir=data_dir,
        recursive=True,
        required_exts=[".pdf"],
        file_extractor={".pdf": pdf_reader},
    )

    docs: List[Document] = reader.load_data()

    # 2) Tag extracted pages
    for d in docs:
        # SimpleDirectoryReader metadata may include a path; normalize to basename for matching
        raw_file_name = d.metadata.get("file_name", "unknown.pdf")
        file_name = Path(raw_file_name).name

        title = _manual_title_from_filename(file_name)
        manual_type = _classify_manual_type(title)

        is_diagram = _is_diagram_page(d.text)
        content_type = "diagram" if is_diagram else "text"

        d.metadata.update(
            {
                "file_name": file_name,  # normalize
                "manual_id": file_name,
                "manual_title": title,
                "manual_type": manual_type,
                "is_ocr": False,
                "content_type": content_type,
            }
        )

        if is_diagram:
            d.metadata["diagram_type"] = _diagram_type(d.text)

    # 3) Build extracted text lookup for OCR decision
    extracted_text_map: Dict[Tuple[str, int], str] = {}
    for d in docs:
        fn = d.metadata.get("file_name")
        pn = d.metadata.get("page_number")

        if fn and pn is not None:
            try:
                extracted_text_map[(str(fn), int(pn))] = d.text or ""
            except Exception:
                pass

    # 4) OCR augmentation (only if deps + tesseract available)
    if not (OCR_DEPS_AVAILABLE and HAS_TESSERACT):
        return docs

    assert fitz is not None

    ocr_docs: List[Document] = []

    for pdf_path in sorted(Path(data_dir).rglob("*.pdf")):
        file_name = pdf_path.name
        title = _manual_title_from_filename(file_name)
        manual_type = _classify_manual_type(title)

        try:
            pdf = fitz.open(str(pdf_path))
        except Exception as e:
            print(f"[OCR] Failed to open {file_name}: {e}")
            continue

        try:
            for i in range(len(pdf)):
                page_no = i + 1
                page = pdf.load_page(i)

                extracted_text = extracted_text_map.get(
                    (file_name, page_no), "")

                if not _needs_ocr(page, extracted_text):
                    continue

                try:
                    text = _ocr_page(pdf, i, dpi=ocr_dpi)
                except Exception as e:
                    print(f"[OCR] Error OCR {file_name} page {page_no}: {e}")
                    continue

                if not _looks_useful(text):
                    continue

                is_diagram = _is_diagram_page(text)
                content_type = "diagram" if is_diagram else "text"

                meta = {
                    "file_name": file_name,
                    "page_label": str(page_no),
                    "page_number": page_no,
                    "is_ocr": True,
                    "manual_id": file_name,
                    "manual_title": title,
                    "manual_type": manual_type,
                    "content_type": content_type,
                }

                if is_diagram:
                    meta["diagram_type"] = _diagram_type(text)

                ocr_docs.append(Document(text=text, metadata=meta))
        finally:
            try:
                pdf.close()
            except Exception:
                pass

    if ocr_docs:
        print(f"[OCR] Added {len(ocr_docs)} OCR pages into ingestion.")
        docs.extend(ocr_docs)

    return docs

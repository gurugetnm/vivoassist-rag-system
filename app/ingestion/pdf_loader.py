from __future__ import annotations

from pathlib import Path
from typing import List

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.readers.file import PDFReader

# OCR deps
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# ✅ Explicit Tesseract path (Windows-safe, avoids PATH refresh issues)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Fail fast if the exe path is wrong
if not Path(pytesseract.pytesseract.tesseract_cmd).exists():
    raise RuntimeError(
        "Tesseract executable not found at "
        f"{pytesseract.pytesseract.tesseract_cmd}. "
        "Install Tesseract or update this path in pdf_loader.py."
    )


# -----------------------------
# Manual metadata helpers
# -----------------------------
def _manual_title_from_filename(file_name: str) -> str:
    """Turn a PDF filename into a clean human title."""
    # "GMDSS System - IOM Manual__.pdf" -> "GMDSS System - IOM Manual"
    return Path(file_name).stem.replace("__", "_").strip()


def _classify_manual_type(title: str) -> str:
    """
    Simple classifier (tune later).
    - "system" for IOM/installation/operation/maintenance/equipment-type manuals
    - otherwise "vehicle"
    """
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
# OCR helpers
# -----------------------------
def _ocr_page(doc: fitz.Document, page_index: int, dpi: int = 200) -> str:
    """Render a PDF page to an image and OCR it using Tesseract."""
    page = doc.load_page(page_index)
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = pytesseract.image_to_string(img)

    return (text or "").strip()


def _looks_useful(text: str, min_chars: int = 40) -> bool:
    """Ignore empty/noisy OCR."""
    t = (text or "").strip()
    if len(t) < min_chars:
        return False
    alnum = sum(ch.isalnum() for ch in t)
    return alnum / max(len(t), 1) > 0.2


def _needs_ocr(page: fitz.Page, extracted_text: str, *, min_chars: int = 60, min_images: int = 1) -> bool:
    """
    OCR only if:
    - extracted text is thin/empty AND
    - page contains images (likely scanned / image-based text)
    """
    text_len = len((extracted_text or "").strip())
    if text_len >= min_chars:
        return False

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
    Load PDFs per-page using LlamaIndex PDFReader, and then OCR ONLY pages
    that look image-based (thin extracted text + has images).

    ✅ Adds manual-level metadata to every per-page document:
      - manual_id: stable scope key (uses file_name)
      - manual_title: cleaned title from filename
      - manual_type: system/vehicle (basic heuristic)

    ✅ Adds OCR pages as extra Documents (same page_number/page_label), flagged with is_ocr=True.
    """

    # 1) Normal text extraction (per-page)
    pdf_reader = PDFReader(return_full_document=False)
    reader = SimpleDirectoryReader(
        input_dir=data_dir,
        recursive=True,
        required_exts=[".pdf"],
        file_extractor={".pdf": pdf_reader},
    )
    docs: List[Document] = reader.load_data()

    # ✅ Tag normal extracted docs with manual-level metadata
    for d in docs:
        file_name = d.metadata.get("file_name", "unknown.pdf")
        title = _manual_title_from_filename(file_name)
        manual_type = _classify_manual_type(title)

        d.metadata["manual_id"] = file_name
        d.metadata["manual_title"] = title
        d.metadata["manual_type"] = manual_type
        d.metadata.setdefault("is_ocr", False)

    # Build lookup: (file_name, page_number) -> extracted text
    extracted_text_map = {}
    for d in docs:
        fn = d.metadata.get("file_name")
        pn = d.metadata.get("page_number")
        if fn and pn:
            try:
                extracted_text_map[(fn, int(pn))] = d.text or ""
            except Exception:
                # if pn isn't int-like for some reason, skip
                pass

    # 2) OCR augmentation (ONLY image-like pages)
    pdf_paths = sorted(Path(data_dir).glob("*.pdf"))
    ocr_docs: List[Document] = []

    for pdf_path in pdf_paths:
        file_name = pdf_path.name
        title = _manual_title_from_filename(file_name)
        manual_type = _classify_manual_type(title)

        try:
            pdf = fitz.open(str(pdf_path))
        except Exception as e:
            print(f"[OCR] Failed to open {file_name}: {e}")
            continue

        for i in range(len(pdf)):
            page_no = i + 1

            try:
                page = pdf.load_page(i)
            except Exception as e:
                print(f"[OCR] Failed to load {file_name} page {page_no}: {e}")
                continue

            extracted_text = extracted_text_map.get((file_name, page_no), "")

            # ✅ OCR only if it looks like an image/scanned page
            if not _needs_ocr(page, extracted_text, min_chars=60, min_images=1):
                continue

            try:
                text = _ocr_page(pdf, i, dpi=ocr_dpi)
            except Exception as e:
                print(f"[OCR] Error OCR {file_name} page {page_no}: {e}")
                continue

            if not _looks_useful(text):
                continue

            meta = {
                "file_name": file_name,
                "page_label": str(page_no),
                "page_number": page_no,
                "is_ocr": True,
                "manual_id": file_name,
                "manual_title": title,
                "manual_type": manual_type,
            }
            ocr_docs.append(Document(text=text, metadata=meta))

        pdf.close()

    # Combine normal + OCR docs
    if ocr_docs:
        print(f"[OCR] Added {len(ocr_docs)} OCR pages into ingestion.")
        docs.extend(ocr_docs)

    return docs

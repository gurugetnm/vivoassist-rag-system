from __future__ import annotations

import os
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


def _ocr_page(doc: fitz.Document, page_index: int, dpi: int = 200) -> str:
    """Render a PDF page to an image and OCR it using Tesseract."""
    page = doc.load_page(page_index)
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = pytesseract.image_to_string(img)

    return (text or "").strip()


def _looks_useful(text: str, min_chars: int = 40) -> bool:
    """Simple heuristic: ignore empty/noisy OCR."""
    t = (text or "").strip()
    if len(t) < min_chars:
        return False

    # If it’s mostly non-alphanumeric, it’s likely garbage
    alnum = sum(ch.isalnum() for ch in t)
    return alnum / max(len(t), 1) > 0.2


def load_pdfs(data_dir: str, *, ocr_first_pages: int = 3) -> List[Document]:
    """
    Load PDFs per-page using LlamaIndex PDFReader AND OCR first N pages
    to capture cover/title text for scanned manuals.
    """

    # 1) Normal text extraction
    pdf_reader = PDFReader(return_full_document=False)
    reader = SimpleDirectoryReader(
        input_dir=data_dir,
        recursive=True,
        required_exts=[".pdf"],
        file_extractor={".pdf": pdf_reader},
    )
    docs: List[Document] = reader.load_data()

    # 2) OCR augmentation (first N pages per PDF)
    pdf_paths = sorted(Path(data_dir).glob("*.pdf"))
    ocr_docs: List[Document] = []

    for pdf_path in pdf_paths:
        file_name = pdf_path.name

        try:
            pdf = fitz.open(str(pdf_path))
        except Exception as e:
            print(f"[OCR] Failed to open {file_name}: {e}")
            continue

        max_pages = min(ocr_first_pages, len(pdf))

        for i in range(max_pages):
            try:
                text = _ocr_page(pdf, i, dpi=200)
            except Exception as e:
                print(f"[OCR] Error OCR {file_name} page {i+1}: {e}")
                continue

            if not _looks_useful(text):
                continue

            # Create a Document that looks like a normal per-page doc
            meta = {
                "file_name": file_name,
                "page_label": str(i + 1),
                "page_number": i + 1,
                "is_ocr": True,
            }
            ocr_docs.append(Document(text=text, metadata=meta))

        pdf.close()

    # Combine normal + OCR docs
    if ocr_docs:
        print(f"[OCR] Added {len(ocr_docs)} OCR pages into ingestion.")
        docs.extend(ocr_docs)

    return docs

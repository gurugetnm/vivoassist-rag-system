from typing import List
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.readers.file import PDFReader


def load_pdfs(data_dir: str) -> List[Document]:
    """
    Load ALL PDFs from a directory, but parse each PDF per-page
    (so metadata like page_label is available).
    """
    pdf_reader = PDFReader(return_full_document=False) 

    reader = SimpleDirectoryReader(
        input_dir=data_dir,
        recursive=True,
        required_exts=[".pdf"],
        file_extractor={".pdf": pdf_reader},
    )

    return reader.load_data()

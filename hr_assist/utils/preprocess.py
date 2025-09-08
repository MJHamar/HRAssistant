"""
Utility module for preprocessing user-supplied inputs.
"""

from pathlib import Path
import pymupdf4llm
import pymupdf


def doc_to_md(name, input_bin: bytes):
    print(f"Converting document {name} to markdown...")
    input_document = pymupdf.open(stream=input_bin, filetype=Path(name).suffix[1:] if isinstance(name, str) else "pdf")
    page_chunked = pymupdf4llm.to_markdown(input_document, filename=name, page_chunks=True)
    
    page_chunked = list(page_chunked)
    converted_doc = "\n\n".join([chunk['text'] for chunk in page_chunked])
    metadata = page_chunked[0].get('metadata', {}) if page_chunked else {}
    metadata = metadata.copy(); metadata.pop('page', None)

    return converted_doc, page_chunked, metadata

def doc_file_to_md(file_path: Path):
    with open(file_path, "rb") as f:
        return doc_to_md(f.read())
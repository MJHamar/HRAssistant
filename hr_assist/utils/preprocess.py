"""
Utility module for preprocessing user-supplied inputs.
"""

from pathlib import Path
import pymupdf4llm


def doc_to_md(input_bin: bytes):
    page_chunked = pymupdf4llm.to_markdown(input_bin, page_chunks=True)
    converted_doc = "\n\n".join([chunk['text'] for chunk in page_chunked])
    metadata = page_chunked[0].get('metadata', {}) if page_chunked else {}
    metadata = metadata.copy(); metadata.pop('page', None)

    return converted_doc, page_chunked, metadata

def doc_file_to_md(file_path: Path):
    with open(file_path, "rb") as f:
        return doc_to_md(f.read())
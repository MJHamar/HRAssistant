"""
Utility module for preprocessing user-supplied inputs.
"""

import os
import pymupdf4llm
import pymupdf
import docx2pdf
import logging
import platform
import tempfile
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def pdf_to_md(name, input_bin: bytes):
    logger.debug(f"Converting PDF document {name} to markdown...")
    input_document = pymupdf.open(stream=input_bin, filetype=Path(name).suffix[1:] if isinstance(name, str) else "pdf")
    page_chunked = pymupdf4llm.to_markdown(input_document, filename=name, page_chunks=True)

    page_chunked = list(page_chunked)
    converted_doc = "\n\n".join([chunk['text'] for chunk in page_chunked])
    metadata = page_chunked[0].get('metadata', {}) if page_chunked else {}
    metadata = metadata.copy(); metadata.pop('page', None)

    return converted_doc, page_chunked, metadata

def pdf_file_to_md(file_path: Path):
    with open(file_path, "rb") as f:
        return pdf_to_md(f.read())

def docx_to_pdf(docx_bytes: bytes) -> bytes:
    """
    Convert a DOCX binary to PDF binary.
    Uses docx2pdf (MS Word, macOS) or LibreOffice (Linux) as backend.
    """
    system = platform.system()
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.docx")
        output_path = os.path.join(tmpdir, "input.pdf")
        # Write input file
        with open(input_path, "wb") as f:
            f.write(docx_bytes)

        if system == "Windows" or system == "Darwin":
            # Use docx2pdf if available
            try:
                from docx2pdf import convert
            except ImportError as ie:
                raise ImportError("docx2pdf must be installed for Windows or MacOS usage.") from ie
            convert(input_path, output_path)
        elif system == "Linux":
            # Use libreoffice CLI
            # Check if libreoffice is installed
            if not any(os.access(os.path.join(path, 'libreoffice'), os.X_OK)
                       for path in os.environ["PATH"].split(os.pathsep)):
                raise EnvironmentError("libreoffice must be installed on Linux for conversion.")
            result = subprocess.run([
                "libreoffice", "--headless", "--convert-to", "pdf", "--outdir", tmpdir, input_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # LibreOffice outputs to the same directory
            if result.returncode != 0 or not os.path.isfile(output_path):
                raise RuntimeError(f"LibreOffice conversion failed: {result.stderr.decode()}")
        else:
            raise NotImplementedError(f"Unsupported OS: {system}")

        with open(output_path, "rb") as f:
            pdf_bytes = f.read()
    return pdf_bytes

def docx_to_md(name, input_bin: bytes):
    """
    Convert a DOCX document to markdown format.

    For standardization purposes, we first convert DOCX to PDF using pdf_to_md, and then extract the markdown.
    """
    logger.debug(f"Converting DOCX document {name} to markdown via PDF...")
    pdf_bytes = docx_to_pdf(input_bin)
    return pdf_to_md(name, pdf_bytes)

def docx_file_to_md(file_path: Path):
    with open(file_path, "rb") as f:
        return docx_to_md(f.read())

def convert_to_md(name: str, input_path: bytes):
    """
    Convert a document (PDF or DOCX) to markdown format.

    Args:
        name: The name of the document file (used to infer file type).
        input_path: The binary content of the document.

    Returns:
        A tuple of (converted_markdown_str, list_of_page_chunks, metadata_dict).
    """
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        return pdf_to_md(name, input_path)
    elif suffix == ".docx" or suffix == ".doc":
        return docx_to_md(name, input_path)
    elif suffix == ".txt":
        text = input_path.decode('utf-8', errors='ignore')
        return text, [{"text": text}], {}
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Only PDF, DOCX, and TXT are supported.")

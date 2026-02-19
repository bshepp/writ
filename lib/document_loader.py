"""
Shared document loading and chunking utilities.

Provides PDF text extraction and paragraph-aware chunking used by both
the entity extraction pipeline and the vector embedding pipeline.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4  # rough approximation


# ---------------------------------------------------------------------------
# PDF reading
# ---------------------------------------------------------------------------

def read_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF, cleaning common artifacts.

    Returns the full document text as a single string.  For page-level
    tracking (needed by the embedding pipeline), use ``read_pdf_pages``.
    """
    pages = read_pdf_pages(pdf_path)
    return "\n\n".join(text for _, text in pages)


def read_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """Extract text from a PDF with page-number tracking.

    Returns a list of ``(page_number, text)`` tuples (1-indexed pages).
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader  # type: ignore[no-redef]

    try:
        reader = PdfReader(str(pdf_path))
        pages: List[Tuple[int, str]] = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as page_err:
                logger.warning("  Page %d extraction failed: %s", i, page_err)
                continue
            text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # rejoin hyphenated words
            text = re.sub(r"\n{3,}", "\n\n", text)          # collapse excessive newlines
            pages.append((i + 1, text))
        return pages
    except Exception as e:
        logger.error("Failed to read PDF %s: %s", pdf_path.name, e)
        return []


# ---------------------------------------------------------------------------
# Chunking (for entity extraction)
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 200,
) -> List[str]:
    """Split text into overlapping chunks of approximately *chunk_size* tokens.

    Splits on paragraph boundaries where possible, with a character-level
    overlap derived from ``overlap * CHARS_PER_TOKEN``.
    """
    char_chunk = chunk_size * CHARS_PER_TOKEN
    char_overlap = overlap * CHARS_PER_TOKEN

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 <= char_chunk:
            current_chunk += ("\n\n" + para) if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if current_chunk and char_overlap > 0:
                overlap_text = current_chunk[-char_overlap:]
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = para

            while len(current_chunk) > char_chunk * 1.5:
                split_point = current_chunk.rfind(". ", 0, char_chunk)
                if split_point == -1:
                    split_point = char_chunk
                chunks.append(current_chunk[: split_point + 1])
                current_chunk = current_chunk[split_point + 1 :].strip()

    if current_chunk.strip():
        chunks.append(current_chunk)

    return chunks


# ---------------------------------------------------------------------------
# Chunking with page tracking (for vector embeddings)
# ---------------------------------------------------------------------------

def chunk_document(
    pages: List[Tuple[int, str]],
    doc_id: str,
    chunk_size: int = 500,
    overlap: int = 100,
    min_chunk_chars: int = 80,
) -> List[Dict]:
    """Chunk a document into overlapping segments with page metadata.

    Each returned dict has keys: ``chunk_id``, ``document_id``, ``text``,
    ``page_number``, ``char_offset``.
    """
    char_chunk = chunk_size * CHARS_PER_TOKEN
    char_overlap = overlap * CHARS_PER_TOKEN

    full_text = ""
    page_offsets: List[Tuple[int, int]] = []
    for page_num, text in pages:
        page_offsets.append((len(full_text), page_num))
        full_text += text + "\n\n"

    if not full_text.strip():
        return []

    paragraphs = re.split(r"\n\s*\n", full_text)

    def _find_page(offset: int) -> int:
        page = 1
        for po, pn in page_offsets:
            if offset >= po:
                page = pn
            else:
                break
        return page

    chunks: List[Dict] = []
    current_chunk = ""
    current_start_offset = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 <= char_chunk:
            if not current_chunk:
                current_start_offset = full_text.find(para, current_start_offset)
            current_chunk += ("\n\n" + para) if current_chunk else para
        else:
            if current_chunk and len(current_chunk.strip()) > min_chunk_chars:
                chunks.append({
                    "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                    "document_id": doc_id,
                    "text": current_chunk.strip(),
                    "page_number": _find_page(current_start_offset),
                    "char_offset": current_start_offset,
                })

            if current_chunk and char_overlap > 0:
                overlap_text = current_chunk[-char_overlap:]
                current_start_offset = max(
                    0, current_start_offset + len(current_chunk) - char_overlap
                )
                current_chunk = overlap_text + "\n\n" + para
            else:
                idx = full_text.find(para, current_start_offset)
                current_start_offset = idx if idx >= 0 else 0
                current_chunk = para

            while len(current_chunk) > char_chunk * 1.5:
                split_point = current_chunk.rfind(". ", 0, char_chunk)
                if split_point == -1:
                    split_point = char_chunk
                piece = current_chunk[: split_point + 1]
                if len(piece.strip()) > min_chunk_chars:
                    chunks.append({
                        "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                        "document_id": doc_id,
                        "text": piece.strip(),
                        "page_number": _find_page(current_start_offset),
                        "char_offset": current_start_offset,
                    })
                current_start_offset += split_point + 1
                current_chunk = current_chunk[split_point + 1 :].strip()

    if current_chunk.strip() and len(current_chunk.strip()) > min_chunk_chars:
        chunks.append({
            "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
            "document_id": doc_id,
            "text": current_chunk.strip(),
            "page_number": _find_page(current_start_offset),
            "char_offset": current_start_offset,
        })

    return chunks

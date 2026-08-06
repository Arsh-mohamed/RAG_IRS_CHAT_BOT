from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker


def chunk_pdf(
    pdf_path: str,
    markdown_path: str | None = None,
    max_tokens: int = 1024,
    merge_peers: bool = True,
):
    """Convert a PDF to a document, optionally save markdown, and chunk it.

    Args:
        pdf_path: Path to the input PDF file.
        markdown_path: Optional path to save the converted markdown.
        max_tokens: Maximum tokens per chunk.
        merge_peers: Whether to merge peer chunks.

    Returns:
        List of chunked document segments.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF path does not exist: {pdf_path}")

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    document = result.document

    if markdown_path is not None:
        markdown_path = Path(markdown_path)
        document.save_as_markdown(str(markdown_path))

    chunker = HybridChunker(max_tokens=max_tokens, merge_peers=merge_peers)
    chunks = list(chunker.chunk(document))
    return chunks

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

# 1. Convert PDF (handles multi-column, tables, layout automatically)
converter = DocumentConverter()
result = converter.convert("/Users/sabarisury/Desktop/RAG_IRS_CHAT_BOT/DATA/IRS_Publication15T.pdf")
document = result.document
document.save_as_markdown("irs_publications15t.md")

# # 2. Chunk with hierarchy + layout awareness
# chunker = HybridChunker(max_tokens=512, merge_peers=True)
# chunks = list(chunker.chunk(document))
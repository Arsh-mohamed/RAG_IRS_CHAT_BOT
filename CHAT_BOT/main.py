import sqlite3
import pandas as pd
from chunk import chunk_pdf

conn = sqlite3.connect("db/sqlite.db")

tables = pd.read_sql(
    "SELECT name FROM sqlite_master WHERE type='table';",
    conn
)

print(tables)

# Example PDF chunking call
pdf_path = "../DATA/IRS_Publication15T.pdf"
markdown_path = "irs_publications15t.md"
chunks = chunk_pdf(pdf_path, markdown_path=markdown_path)
print(f"Created {len(chunks)} chunks from {pdf_path}")

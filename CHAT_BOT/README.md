# RAG IRS Chat Bot

This repository contains a simple retrieval-augmented generation (RAG) pipeline for loading structured data into SQLite and chunking PDF documents.

## Files

- `main.py` - entrypoint for the RAG pipeline
- `excel_load.py` - loads CSV / Excel data into `db/sqlite.db`
- `chunk.py` - converts and chunks PDF documents using `docling`
- `db/` - SQLite database output folder
- `DATA/` - sample input data folder

## Usage

Activate your virtual environment, then run:

```bash
cd CHAT_BOT
python main.py --data-path ../DATA/tax_brackets_2026.csv
```

To chunk a PDF:

```bash
python main.py --pdf-path path/to/document.pdf --markdown-path output.md
```

You can also run both together:

```bash
python main.py --data-path ../DATA/tax_brackets_2026.csv --pdf-path path/to/document.pdf
```

## Options

- `--data-path`, `-d`: path to a CSV/XLSX file or a directory of data files
- `--pdf-path`, `-p`: path to a PDF file to convert and chunk
- `--markdown-path`, `-m`: optional path to save converted markdown
- `--max-tokens`, `-t`: max tokens per chunk (default `1024`)
- `--no-merge`: disable peer chunk merging

## Notes

- `excel_load.py` writes loaded tables to `db/sqlite.db`
- `chunk.py` uses `docling` to convert PDFs and chunk text
- Ensure the input paths exist and are correct

import argparse
from pathlib import Path

from chunk import chunk_pdf
from embedding import store_chunks_in_chroma
from excel_load import load_data
from retrival import run_chat


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "DATA"
CHROMA_PATH = BASE_DIR / "db" / "chroma"


def ingest_data() -> bool:
    if not DATA_PATH.exists():
        print(f"Path not found: {DATA_PATH}")
        return False
    if not DATA_PATH.is_dir():
        print(f"Expected a data directory: {DATA_PATH}")
        return False

    print(f"Loading structured data from directory: {DATA_PATH}")
    load_data(str(DATA_PATH))

    pdf_files = sorted(DATA_PATH.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found to chunk in the directory.")
        return False

    for pdf_path in pdf_files:
        print(f"Chunking PDF: {pdf_path}")
        try:
            chunks = chunk_pdf(
                pdf_path=str(pdf_path),
                markdown_path=None,
                max_tokens=1024,
                merge_peers=True,
            )
            print(f"Chunked {len(chunks)} segments from {pdf_path}")
            store_chunks_in_chroma(
                chunks,
                collection_name=pdf_path.stem,
                persist_directory=CHROMA_PATH,
            )
            print(f"Stored chunks in Chroma collection '{pdf_path.stem}'")
        except FileNotFoundError as exc:
            print(f"Could not process {pdf_path}: {exc}")
            return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the IRS retrieval chat.")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="load structured files and rebuild PDF vector collections before chat",
    )
    args = parser.parse_args()

    if args.ingest and not ingest_data():
        return
    run_chat()


if __name__ == "__main__":
    main()

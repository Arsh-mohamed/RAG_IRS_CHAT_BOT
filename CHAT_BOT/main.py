from pathlib import Path

from excel_load import load_data
from chunk import chunk_pdf
from embedding import store_chunks_in_chroma


# Default path points to the workspace DATA folder.
DATA_PATH = Path(__file__).resolve().parent.parent / "DATA"


def main():
    path = DATA_PATH

    if not path.exists():
        print(f"❌ Path not found: {path}")
        return

    suffix = path.suffix.lower()
    if path.is_dir():
        print(f"Loading data from directory: {path}")
        load_data(str(path))

        pdf_files = sorted(path.glob("*.pdf"))
        if not pdf_files:
            print("No PDF files found to chunk in the directory.")
            return

        for pdf_path in pdf_files:
            print(f"Chunking PDF: {pdf_path}")
            try:
                chunks = chunk_pdf(
                    pdf_path=str(pdf_path),
                    markdown_path=None,
                    max_tokens=1024,
                    merge_peers=True,
                )
                print(f"✅ Chunked {len(chunks)} segments from {pdf_path}")
                chroma = store_chunks_in_chroma(
                    chunks,
                    collection_name=pdf_path.stem,
                    persist_directory=Path(__file__).resolve().parent / "db" / "chroma",
                )
                print(f"✅ Stored chunks in Chroma collection '{chroma.collection_name}'")
            except FileNotFoundError as exc:
                print(f"❌ {exc}")
        return

    if suffix in {".csv", ".xlsx", ".xls"}:
        print(f"Loading structured data from file: {path}")
        load_data(str(path))
        return

    if suffix == ".pdf":
        print(f"Chunking PDF: {path}")
        try:
            chunks = chunk_pdf(
                pdf_path=str(path),
                markdown_path=None,
                max_tokens=1024,
                merge_peers=True,
            )
            print(f"✅ Chunked {len(chunks)} segments from {path}")
            chroma = store_chunks_in_chroma(
                chunks,
                collection_name=path.stem,
                persist_directory=Path(__file__).resolve().parent / "db" / "chroma",
            )
            print(f"✅ Stored chunks in Chroma collection '{chroma.collection_name}'")
        except FileNotFoundError as exc:
            print(f"❌ {exc}")
        return

    print(
        f"Unsupported path type: {path}. Provide a directory, CSV/XLSX file, or PDF file."
    )


if __name__ == "__main__":
    main()

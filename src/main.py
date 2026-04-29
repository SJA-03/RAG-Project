import sys

from chunking.basic_chunker import chunk_text
from ingestion.pdf_loader import load_pdf


DEFAULT_PDF_PATH = "data/os/lec1.pdf"


def main() -> int:
    file_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF_PATH

    try:
        text = load_pdf(file_path)
        chunks = chunk_text(text)
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Unexpected error while processing PDF: {error}", file=sys.stderr)
        return 1

    print(f"Loaded PDF: {file_path}")
    print(f"Extracted characters: {len(text)}")
    print(f"Generated chunks: {len(chunks)}")

    if chunks:
        preview = chunks[0][:200].replace("\n", " ")
        print(f"First chunk preview: {preview}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

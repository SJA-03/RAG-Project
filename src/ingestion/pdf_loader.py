from pathlib import Path

import pdfplumber


def load_pdf(file_path: str) -> str:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    return _extract_pdf_text(path)


def load_pdfs_from_folder(folder_path: str) -> list[dict[str, str]]:
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"PDF folder not found: {folder_path}")

    pdf_paths = sorted(
        (path for path in folder.rglob("*.pdf") if path.is_file()),
        key=_pdf_sort_key,
    )
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found under folder: {folder_path}")

    documents: list[dict[str, str]] = []
    for pdf_path in pdf_paths:
        documents.append(
            {
                "source": str(pdf_path),
                "text": _extract_pdf_text(pdf_path),
            }
        )

    return documents


def _extract_pdf_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    pages: list[str] = []

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text is None:
                continue

            cleaned = page_text.strip()
            if cleaned:
                pages.append(cleaned)

    full_text = "\n\n".join(pages).strip()
    if not full_text:
        raise ValueError(f"No extractable text found in PDF: {path}")

    return full_text


def _pdf_sort_key(path: Path) -> tuple[int, str]:
    normalized_path = path.as_posix()
    priority = 0 if normalized_path.endswith("data/os/lec1.pdf") else 1
    return (priority, normalized_path)

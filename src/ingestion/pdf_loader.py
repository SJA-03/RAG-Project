from pathlib import Path

import pdfplumber


def load_pdf(file_path: str) -> str:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

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
        raise ValueError(f"No extractable text found in PDF: {file_path}")

    return full_text

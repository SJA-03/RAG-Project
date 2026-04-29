def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    paragraphs = [paragraph.strip() for paragraph in cleaned_text.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        paragraphs = [cleaned_text]

    chunks: list[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            chunks.extend(_split_long_text(paragraph, chunk_size))
            continue

        if not current_chunk:
            current_chunk = paragraph
            continue

        candidate = f"{current_chunk}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            chunks.append(current_chunk)
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _split_long_text(text: str, chunk_size: int) -> list[str]:
    words = text.split()
    if not words:
        return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]

    chunks: list[str] = []
    current_chunk = ""

    for word in words:
        candidate = word if not current_chunk else f"{current_chunk} {word}"
        if len(candidate) <= chunk_size:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(current_chunk)

        if len(word) <= chunk_size:
            current_chunk = word
        else:
            chunks.extend(word[index : index + chunk_size] for index in range(0, len(word), chunk_size))
            current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

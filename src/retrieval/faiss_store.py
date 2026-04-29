import json
from pathlib import Path

import faiss
import numpy as np


INDEX_FILE_NAME = "faiss.index"
CHUNKS_FILE_NAME = "chunks.json"


def save_faiss_index(
    embeddings: list[list[float]],
    chunks: list[dict[str, str]] | list[str],
    output_dir: str = "embeddings",
) -> tuple[str, str]:
    if not embeddings:
        raise ValueError("Cannot save FAISS index with empty embeddings")
    if len(embeddings) != len(chunks):
        raise ValueError("Embeddings count must match chunks count")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vectors = np.array(embeddings, dtype=np.float32)
    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    index_path = output_path / INDEX_FILE_NAME
    chunks_path = output_path / CHUNKS_FILE_NAME

    faiss.write_index(index, str(index_path))

    normalized_chunks = _normalize_chunks(chunks)
    chunk_metadata = [
        {
            "chunk_id": index,
            "text": chunk["text"],
            "source": chunk["source"],
        }
        for index, chunk in enumerate(normalized_chunks)
    ]
    chunks_path.write_text(
        json.dumps(chunk_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return str(index_path), str(chunks_path)


def _normalize_chunks(chunks: list[dict[str, str]] | list[str]) -> list[dict[str, str]]:
    normalized_chunks: list[dict[str, str]] = []

    for chunk in chunks:
        if isinstance(chunk, str):
            normalized_chunks.append({"text": chunk, "source": "unknown"})
            continue

        normalized_chunks.append(
            {
                "text": chunk["text"],
                "source": chunk.get("source", "unknown"),
            }
        )

    return normalized_chunks

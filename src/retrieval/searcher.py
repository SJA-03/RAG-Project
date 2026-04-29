import json
from pathlib import Path

import faiss
import numpy as np

from embedding.embedder import Embedder
from retrieval.faiss_store import CHUNKS_FILE_NAME, INDEX_FILE_NAME


def search_chunks(
    query: str,
    top_k: int = 3,
    embeddings_dir: str = "embeddings",
    embedder: Embedder | None = None,
) -> list[dict[str, int | float | str]]:
    if not query.strip():
        raise ValueError("Query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    embeddings_path = Path(embeddings_dir)
    index_path = embeddings_path / INDEX_FILE_NAME
    chunks_path = embeddings_path / CHUNKS_FILE_NAME

    if not index_path.is_file():
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunk metadata file not found: {chunks_path}")

    index = faiss.read_index(str(index_path))
    chunk_metadata = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunk_map = {item["chunk_id"]: item["text"] for item in chunk_metadata}

    active_embedder = embedder or Embedder()
    query_vector = np.array(active_embedder.embed_chunks([query]), dtype=np.float32)

    scores, indices = index.search(query_vector, top_k)

    results: list[dict[str, int | float | str]] = []
    for chunk_index, score in zip(indices[0], scores[0]):
        if chunk_index < 0:
            continue

        normalized_chunk_index = int(chunk_index)
        chunk_text = chunk_map.get(normalized_chunk_index)
        if chunk_text is None:
            continue

        results.append(
            {
                "chunk_id": normalized_chunk_index,
                "text": chunk_text,
                "similarity_score": float(score),
            }
        )

    return results

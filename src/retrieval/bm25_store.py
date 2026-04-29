import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from retrieval.faiss_store import CHUNKS_FILE_NAME


def search_bm25(query: str, top_k: int = 3, embeddings_dir: str = "embeddings") -> list[dict[str, int | float | str]]:
    if not query.strip():
        raise ValueError("Query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    embeddings_path = Path(embeddings_dir)
    chunks_path = embeddings_path / CHUNKS_FILE_NAME

    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunk metadata file not found: {chunks_path}")

    chunk_metadata = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not chunk_metadata:
        raise ValueError(f"No chunk metadata found in: {chunks_path}")

    corpus = [item["text"] for item in chunk_metadata]
    tokenized_corpus = [_tokenize(text) for text in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:top_k]

    results: list[dict[str, int | float | str]] = []
    for index in ranked_indices:
        item = chunk_metadata[index]
        results.append(
            {
                "chunk_id": int(item["chunk_id"]),
                "text": item["text"],
                "source": item.get("source", "unknown"),
                "bm25_score": float(scores[index]),
            }
        )

    return results


def _tokenize(text: str) -> list[str]:
    return text.lower().split()

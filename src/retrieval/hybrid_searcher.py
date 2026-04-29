from embedding.embedder import Embedder
from retrieval.bm25_store import search_bm25
from retrieval.searcher import search_chunks


def search_hybrid(
    query: str,
    top_k: int = 3,
    alpha: float = 0.5,
    embeddings_dir: str = "embeddings",
    embedder: Embedder | None = None,
) -> list[dict[str, int | float | str]]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")

    dense_results = search_chunks(
        query,
        top_k=top_k,
        embeddings_dir=embeddings_dir,
        embedder=embedder,
    )
    sparse_results = search_bm25(
        query,
        top_k=top_k,
        embeddings_dir=embeddings_dir,
    )

    normalized_dense_scores = _normalize_scores(dense_results, "similarity_score")
    normalized_sparse_scores = _normalize_scores(sparse_results, "bm25_score")

    combined_results: dict[int, dict[str, int | float | str]] = {}

    for result in dense_results:
        chunk_id = int(result["chunk_id"])
        combined_results[chunk_id] = {
            "chunk_id": chunk_id,
            "text": result["text"],
            "source": result.get("source", "unknown"),
            "dense_score": normalized_dense_scores[chunk_id],
            "sparse_score": 0.0,
            "final_score": alpha * normalized_dense_scores[chunk_id],
        }

    for result in sparse_results:
        chunk_id = int(result["chunk_id"])
        normalized_sparse_score = normalized_sparse_scores[chunk_id]

        if chunk_id in combined_results:
            combined_results[chunk_id]["sparse_score"] = normalized_sparse_score
            combined_results[chunk_id]["final_score"] = float(combined_results[chunk_id]["final_score"]) + (
                (1.0 - alpha) * normalized_sparse_score
            )
            continue

        combined_results[chunk_id] = {
            "chunk_id": chunk_id,
            "text": result["text"],
            "source": result.get("source", "unknown"),
            "dense_score": 0.0,
            "sparse_score": normalized_sparse_score,
            "final_score": (1.0 - alpha) * normalized_sparse_score,
        }

    ranked_results = sorted(
        combined_results.values(),
        key=lambda item: float(item["final_score"]),
        reverse=True,
    )

    return ranked_results[:top_k]


def _normalize_scores(results: list[dict[str, int | float | str]], score_key: str) -> dict[int, float]:
    if not results:
        return {}

    scores = [float(result[score_key]) for result in results]
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return {int(result["chunk_id"]): 1.0 for result in results}

    return {
        int(result["chunk_id"]): (float(result[score_key]) - min_score) / (max_score - min_score)
        for result in results
    }

import os
from copy import deepcopy

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(
            model_name,
            local_files_only=_use_local_files_only(),
        )

    def rerank(self, query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not candidates:
            return []

        query_document_pairs = [(query, candidate["text"]) for candidate in candidates]
        scores = self.model.predict(query_document_pairs)

        reranked_results: list[dict] = []
        for candidate, score in zip(candidates, scores):
            updated_candidate = deepcopy(candidate)
            updated_candidate["reranker_score"] = float(score)
            reranked_results.append(updated_candidate)

        reranked_results.sort(key=lambda item: item["reranker_score"], reverse=True)
        return reranked_results[:top_k]


def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    reranker = Reranker()
    return reranker.rerank(query, candidates, top_k=top_k)


def _use_local_files_only() -> bool:
    return os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"

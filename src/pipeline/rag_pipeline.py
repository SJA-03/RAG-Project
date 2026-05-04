import io
import json
import os
from pathlib import Path

import faiss
import numpy as np
import pdfplumber
from rank_bm25 import BM25Okapi

from chunking.basic_chunker import chunk_text
from embedding.embedder import Embedder
from llm.answer_generator import generate_answer as generate_grounded_answer
from reranker.reranker import Reranker
from retrieval.hybrid_searcher import search_hybrid

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class RAGPipeline:
    def __init__(
        self,
        embeddings_dir: str = "embeddings",
        alpha: float = 0.5,
        reranker_candidate_k: int = 10,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.embeddings_dir = Path(embeddings_dir)
        self.alpha = alpha
        self.reranker_candidate_k = reranker_candidate_k
        self.embedder = embedder or Embedder()
        self.reranker = reranker or Reranker()

        self._uploaded_chunk_records: list[dict[str, str | int]] = []
        self._uploaded_embeddings: np.ndarray | None = None
        self._uploaded_bm25: BM25Okapi | None = None
        self._mode = "existing"

    def build_from_uploaded_files(self, uploaded_files) -> None:
        if not uploaded_files:
            raise ValueError("Please upload at least one PDF file.")

        documents = self._extract_uploaded_documents(uploaded_files)
        chunk_records = self._build_chunk_records(documents)
        embeddings = np.array(
            self.embedder.embed_chunks([chunk["text"] for chunk in chunk_records]),
            dtype=np.float32,
        )
        bm25 = BM25Okapi([self._tokenize(chunk["text"]) for chunk in chunk_records])

        self._uploaded_chunk_records = chunk_records
        self._uploaded_embeddings = embeddings
        self._uploaded_bm25 = bm25
        self._mode = "uploaded"

    def use_existing_index(self) -> None:
        self._mode = "existing"

    def retrieve(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 3,
        domain_filter: str | None = None,
    ) -> list[dict]:
        normalized_method = method.lower()
        if normalized_method not in {"hybrid", "reranker"}:
            raise ValueError("method must be either 'hybrid' or 'reranker'")

        if self._mode == "uploaded":
            contexts = self._search_uploaded_hybrid(
                query,
                top_k=self.reranker_candidate_k,
                domain_filter=domain_filter,
            )
            if domain_filter and not contexts:
                contexts = self._search_uploaded_hybrid(query, top_k=self.reranker_candidate_k)
            if normalized_method == "hybrid":
                return contexts[:top_k]

            return self.reranker.rerank(query, contexts, top_k=top_k)

        self._validate_existing_index()
        contexts = search_hybrid(
            query,
            top_k=self.reranker_candidate_k,
            embeddings_dir=str(self.embeddings_dir),
            embedder=self.embedder,
            domain_filter=domain_filter,
        )
        if domain_filter and not contexts:
            contexts = search_hybrid(
                query,
                top_k=self.reranker_candidate_k,
                embeddings_dir=str(self.embeddings_dir),
                embedder=self.embedder,
            )
        if normalized_method == "hybrid":
            return contexts[:top_k]

        return self.reranker.rerank(query, contexts, top_k=top_k)

    def generate_answer(self, query: str, contexts: list[dict]) -> str:
        return generate_grounded_answer(query, contexts)

    def _extract_uploaded_documents(self, uploaded_files) -> list[dict[str, str]]:
        documents: list[dict[str, str]] = []

        for uploaded_file in uploaded_files:
            text = self._extract_text_from_uploaded_pdf(uploaded_file.name, uploaded_file.getvalue())
            documents.append(
                {
                    "source": uploaded_file.name,
                    "text": text,
                }
            )

        return documents

    def _extract_text_from_uploaded_pdf(self, file_name: str, file_bytes: bytes) -> str:
        pages: list[str] = []

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text is None:
                    continue

                cleaned = page_text.strip()
                if cleaned:
                    pages.append(cleaned)

        full_text = "\n\n".join(pages).strip()
        if not full_text:
            raise ValueError(f"No extractable text found in uploaded PDF: {file_name}")

        return full_text

    def _build_chunk_records(self, documents: list[dict[str, str]]) -> list[dict[str, str | int]]:
        chunk_records: list[dict[str, str | int]] = []

        for document in documents:
            for chunk in chunk_text(document["text"]):
                chunk_records.append(
                    {
                        "chunk_id": len(chunk_records),
                        "text": chunk,
                        "source": document["source"],
                    }
                )

        if not chunk_records:
            raise ValueError("No chunks were generated from the uploaded PDFs.")

        return chunk_records

    def _search_uploaded_hybrid(self, query: str, top_k: int, domain_filter: str | None = None) -> list[dict]:
        if not self._uploaded_chunk_records or self._uploaded_embeddings is None or self._uploaded_bm25 is None:
            raise ValueError("Uploaded PDF corpus has not been built yet.")

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")

        query_vector = np.array(self.embedder.embed_chunks([query]), dtype=np.float32)
        dense_results = self._search_uploaded_faiss(query_vector, top_k=top_k, domain_filter=domain_filter)
        sparse_results = self._search_uploaded_bm25(query, top_k=top_k, domain_filter=domain_filter)

        normalized_dense_scores = self._normalize_scores(dense_results, "similarity_score")
        normalized_sparse_scores = self._normalize_scores(sparse_results, "bm25_score")
        combined_results: dict[int, dict] = {}

        for result in dense_results:
            chunk_id = int(result["chunk_id"])
            combined_results[chunk_id] = {
                "chunk_id": chunk_id,
                "text": result["text"],
                "source": result["source"],
                "dense_score": normalized_dense_scores[chunk_id],
                "sparse_score": 0.0,
                "final_score": self.alpha * normalized_dense_scores[chunk_id],
            }

        for result in sparse_results:
            chunk_id = int(result["chunk_id"])
            normalized_sparse_score = normalized_sparse_scores[chunk_id]

            if chunk_id in combined_results:
                combined_results[chunk_id]["sparse_score"] = normalized_sparse_score
                combined_results[chunk_id]["final_score"] = float(combined_results[chunk_id]["final_score"]) + (
                    (1.0 - self.alpha) * normalized_sparse_score
                )
                continue

            combined_results[chunk_id] = {
                "chunk_id": chunk_id,
                "text": result["text"],
                "source": result["source"],
                "dense_score": 0.0,
                "sparse_score": normalized_sparse_score,
                "final_score": (1.0 - self.alpha) * normalized_sparse_score,
            }

        ranked_results = sorted(
            combined_results.values(),
            key=lambda item: float(item["final_score"]),
            reverse=True,
        )
        return ranked_results[:top_k]

    def _search_uploaded_faiss(
        self,
        query_vector: np.ndarray,
        top_k: int,
        domain_filter: str | None = None,
    ) -> list[dict]:
        assert self._uploaded_embeddings is not None

        dimension = self._uploaded_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(self._uploaded_embeddings)

        search_k = min(len(self._uploaded_chunk_records), max(top_k * 5, top_k))
        scores, indices = index.search(query_vector, search_k)
        chunk_map = {int(chunk["chunk_id"]): chunk for chunk in self._uploaded_chunk_records}
        results: list[dict] = []

        for chunk_index, score in zip(indices[0], scores[0]):
            if chunk_index < 0:
                continue

            normalized_chunk_index = int(chunk_index)
            chunk = chunk_map.get(normalized_chunk_index)
            if chunk is None:
                continue
            if domain_filter and not self._matches_domain(str(chunk["source"]), domain_filter):
                continue

            results.append(
                {
                    "chunk_id": normalized_chunk_index,
                    "text": str(chunk["text"]),
                    "source": str(chunk["source"]),
                    "similarity_score": float(score),
                }
            )
            if len(results) >= top_k:
                break

        return results

    def _search_uploaded_bm25(self, query: str, top_k: int, domain_filter: str | None = None) -> list[dict]:
        assert self._uploaded_bm25 is not None

        scores = self._uploaded_bm25.get_scores(self._tokenize(query))
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: list[dict] = []
        for index in ranked_indices:
            chunk = self._uploaded_chunk_records[index]
            if domain_filter and not self._matches_domain(str(chunk["source"]), domain_filter):
                continue
            results.append(
                {
                    "chunk_id": int(chunk["chunk_id"]),
                    "text": str(chunk["text"]),
                    "source": str(chunk["source"]),
                    "bm25_score": float(scores[index]),
                }
            )
            if len(results) >= top_k:
                break

        return results

    def _normalize_scores(self, results: list[dict], score_key: str) -> dict[int, float]:
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

    def _validate_existing_index(self) -> None:
        index_path = self.embeddings_dir / "faiss.index"
        chunks_path = self.embeddings_dir / "chunks.json"

        if not index_path.is_file() or not chunks_path.is_file():
            raise FileNotFoundError("Embeddings not found. Please run `python src/main.py` first.")

        chunk_metadata = json.loads(chunks_path.read_text(encoding="utf-8"))
        if not chunk_metadata:
            raise ValueError(f"No chunk metadata found in: {chunks_path}")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    @staticmethod
    def _matches_domain(source: str, domain_filter: str) -> bool:
        normalized_source = source.replace("\\", "/").lower()
        normalized_domain = domain_filter.lower()

        if f"/{normalized_domain}/" in normalized_source:
            return True

        file_name = normalized_source.rsplit("/", maxsplit=1)[-1]
        return file_name.startswith(f"{normalized_domain}_") or file_name.startswith(f"{normalized_domain}-")

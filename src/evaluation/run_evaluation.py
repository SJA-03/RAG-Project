import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from embedding.embedder import Embedder
from evaluation.evaluator import evaluate_hit_at_k
from reranker.reranker import Reranker
from retrieval.bm25_store import search_bm25
from retrieval.hybrid_searcher import search_hybrid
from retrieval.searcher import search_chunks


QA_DATASET_PATH = REPO_ROOT / "eval" / "qa_dataset.json"
EMBEDDINGS_DIR = REPO_ROOT / "embeddings"
INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
CHUNKS_PATH = EMBEDDINGS_DIR / "chunks.json"
TOP_K = 3
RERANKER_CANDIDATE_K = 10


def main() -> int:
    if not INDEX_PATH.is_file() or not CHUNKS_PATH.is_file():
        print("Embeddings not found. Please run `python src/main.py` first.")
        return 1

    if not QA_DATASET_PATH.is_file():
        print(f"QA dataset not found: {QA_DATASET_PATH}")
        return 1

    qa_dataset = json.loads(QA_DATASET_PATH.read_text(encoding="utf-8"))

    embedder = Embedder()
    reranker = Reranker()

    faiss_metrics = evaluate_hit_at_k(
        qa_dataset,
        lambda question, top_k: search_chunks(
            question,
            top_k=top_k,
            embeddings_dir=str(EMBEDDINGS_DIR),
            embedder=embedder,
        ),
        top_k=TOP_K,
    )
    bm25_metrics = evaluate_hit_at_k(
        qa_dataset,
        lambda question, top_k: search_bm25(
            question,
            top_k=top_k,
            embeddings_dir=str(EMBEDDINGS_DIR),
        ),
        top_k=TOP_K,
    )
    hybrid_metrics = evaluate_hit_at_k(
        qa_dataset,
        lambda question, top_k: search_hybrid(
            question,
            top_k=top_k,
            embeddings_dir=str(EMBEDDINGS_DIR),
            embedder=embedder,
        ),
        top_k=TOP_K,
    )
    reranker_metrics = evaluate_hit_at_k(
        qa_dataset,
        lambda question, top_k: reranker.rerank(
            question,
            search_hybrid(
                question,
                top_k=RERANKER_CANDIDATE_K,
                embeddings_dir=str(EMBEDDINGS_DIR),
                embedder=embedder,
            ),
            top_k=top_k,
        ),
        top_k=TOP_K,
    )

    print("Retrieval Evaluation Results")
    print(f"Dataset: {QA_DATASET_PATH}")
    print(f"Metric: Hit@{TOP_K}")
    print()

    _print_method_results("FAISS", faiss_metrics)
    _print_method_results("BM25", bm25_metrics)
    _print_method_results("Hybrid", hybrid_metrics)
    _print_method_results("Reranker", reranker_metrics)

    return 0


def _print_method_results(method_name: str, metrics: dict) -> None:
    print(f"{method_name}")
    print(f"  Hit@{metrics['top_k']}: {metrics['hit_at_k']:.2f} ({metrics['hits']}/{metrics['total']})")

    for index, detail in enumerate(metrics["details"], start=1):
        status = "HIT" if detail["hit"] else "MISS"
        print(f"  {index}. {status}")
        print(f"     Question: {detail['question']}")
        print(f"     Expected chunk IDs: {detail['expected_chunk_ids']}")
        print(f"     Retrieved chunk IDs: {detail['retrieved_chunk_ids']}")

    print()


if __name__ == "__main__":
    raise SystemExit(main())

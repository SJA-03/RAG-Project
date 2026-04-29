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
from evaluation.evaluator import evaluate_retrieval
from reranker.reranker import Reranker
from retrieval.bm25_store import search_bm25
from retrieval.hybrid_searcher import search_hybrid
from retrieval.searcher import search_chunks


QA_DATASET_PATH = REPO_ROOT / "eval" / "qa_dataset.json"
EMBEDDINGS_DIR = REPO_ROOT / "embeddings"
INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
CHUNKS_PATH = EMBEDDINGS_DIR / "chunks.json"
EVAL_KS = (1, 3, 5)
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

    faiss_metrics = evaluate_retrieval(
        qa_dataset,
        lambda question, top_k: search_chunks(
            question,
            top_k=top_k,
            embeddings_dir=str(EMBEDDINGS_DIR),
            embedder=embedder,
        ),
        ks=EVAL_KS,
    )
    bm25_metrics = evaluate_retrieval(
        qa_dataset,
        lambda question, top_k: search_bm25(
            question,
            top_k=top_k,
            embeddings_dir=str(EMBEDDINGS_DIR),
        ),
        ks=EVAL_KS,
    )
    hybrid_metrics = evaluate_retrieval(
        qa_dataset,
        lambda question, top_k: search_hybrid(
            question,
            top_k=top_k,
            embeddings_dir=str(EMBEDDINGS_DIR),
            embedder=embedder,
        ),
        ks=EVAL_KS,
    )
    reranker_metrics = evaluate_retrieval(
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
        ks=EVAL_KS,
    )

    print("Retrieval Evaluation Results")
    print(f"Dataset: {QA_DATASET_PATH}")
    print()
    _print_summary_table(
        {
            "FAISS": faiss_metrics,
            "BM25": bm25_metrics,
            "Hybrid": hybrid_metrics,
            "Reranker": reranker_metrics,
        }
    )
    print()

    return 0


def _print_summary_table(metrics_by_method: dict[str, dict]) -> None:
    headers = ("Method", "Hit@1", "Hit@3", "Hit@5", "MRR")
    rows = [
        (
            method_name,
            f"{metrics['hit_at_k'][1]:.2f}",
            f"{metrics['hit_at_k'][3]:.2f}",
            f"{metrics['hit_at_k'][5]:.2f}",
            f"{metrics['mrr']:.2f}",
        )
        for method_name, metrics in metrics_by_method.items()
    ]

    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    header_line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    separator_line = "  ".join("-" * widths[index] for index in range(len(headers)))

    print(header_line)
    print(separator_line)

    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


if __name__ == "__main__":
    raise SystemExit(main())

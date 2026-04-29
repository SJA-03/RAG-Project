from typing import Callable


SearchFn = Callable[[str, int], list[dict]]


def calculate_hit_at_k(retrieved_chunk_ids: list[int], expected_chunk_ids: list[int], top_k: int) -> bool:
    expected_set = set(expected_chunk_ids)
    return not expected_set.isdisjoint(retrieved_chunk_ids[:top_k])


def calculate_reciprocal_rank(retrieved_chunk_ids: list[int], expected_chunk_ids: list[int]) -> float:
    expected_set = set(expected_chunk_ids)

    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in expected_set:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval(qa_dataset: list[dict], search_fn: SearchFn, ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    total_questions = len(qa_dataset)
    hit_counts = {k: 0 for k in ks}
    reciprocal_rank_sum = 0.0
    details: list[dict] = []
    max_k = max(ks)

    for item in qa_dataset:
        question = item["question"]
        expected_chunk_ids = item["expected_chunk_ids"]
        results = search_fn(question, max_k)
        retrieved_chunk_ids = [int(result["chunk_id"]) for result in results if "chunk_id" in result]
        hits = {
            k: calculate_hit_at_k(retrieved_chunk_ids, expected_chunk_ids, k)
            for k in ks
        }
        reciprocal_rank = calculate_reciprocal_rank(retrieved_chunk_ids, expected_chunk_ids)

        for k, hit in hits.items():
            if hit:
                hit_counts[k] += 1

        reciprocal_rank_sum += reciprocal_rank

        details.append(
            {
                "question": question,
                "expected_chunk_ids": expected_chunk_ids,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "hits": hits,
                "reciprocal_rank": reciprocal_rank,
            }
        )

    return {
        "total": total_questions,
        "hit_at_k": {
            k: (hit_counts[k] / total_questions if total_questions else 0.0)
            for k in ks
        },
        "mrr": reciprocal_rank_sum / total_questions if total_questions else 0.0,
        "details": details,
    }

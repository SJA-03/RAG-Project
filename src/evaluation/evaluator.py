from typing import Callable


SearchFn = Callable[[str, int], list[dict]]


def calculate_hit_at_k(results: list[dict], expected_chunk_ids: list[int]) -> bool:
    expected_set = set(expected_chunk_ids)
    retrieved_ids = {
        int(result["chunk_id"])
        for result in results
        if "chunk_id" in result
    }
    return not expected_set.isdisjoint(retrieved_ids)


def evaluate_hit_at_k(qa_dataset: list[dict], search_fn: SearchFn, top_k: int = 3) -> dict:
    total_questions = len(qa_dataset)
    hits = 0
    details: list[dict] = []

    for item in qa_dataset:
        question = item["question"]
        expected_chunk_ids = item["expected_chunk_ids"]
        results = search_fn(question, top_k)
        hit = calculate_hit_at_k(results, expected_chunk_ids)

        if hit:
            hits += 1

        details.append(
            {
                "question": question,
                "expected_chunk_ids": expected_chunk_ids,
                "retrieved_chunk_ids": [int(result["chunk_id"]) for result in results],
                "hit": hit,
            }
        )

    hit_at_k = hits / total_questions if total_questions else 0.0

    return {
        "top_k": top_k,
        "hits": hits,
        "total": total_questions,
        "hit_at_k": hit_at_k,
        "details": details,
    }

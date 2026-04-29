import io
import os
import sys
from pathlib import Path

import faiss
import numpy as np
import pdfplumber
import streamlit as st
from rank_bm25 import BM25Okapi


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
EMBEDDINGS_DIR = REPO_ROOT / "embeddings"
INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
CHUNKS_PATH = EMBEDDINGS_DIR / "chunks.json"
DEFAULT_QUERY = "What is a process in operating systems?"
DISPLAY_TOP_K = 3
RERANKER_CANDIDATE_K = 10

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from chunking.basic_chunker import chunk_text
from embedding.embedder import Embedder
from llm.answer_generator import generate_answer
from reranker.reranker import Reranker
from retrieval.hybrid_searcher import search_hybrid


st.set_page_config(page_title="Context-Aware Academic Assistant", layout="wide")


@st.cache_resource
def get_embedder() -> Embedder:
    return Embedder()


@st.cache_resource
def get_reranker() -> Reranker:
    return Reranker()


def main() -> None:
    st.title("Context-Aware Academic Assistant")

    data_source = st.radio(
        "Corpus source",
        options=["Uploaded PDFs", "Existing indexed corpus"],
        index=0,
    )
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        disabled=data_source != "Uploaded PDFs",
    )
    query = st.text_input("Question", value=DEFAULT_QUERY)
    retrieval_mode = st.selectbox("Search method", options=["Hybrid", "Reranker"])

    if data_source == "Existing indexed corpus" and (not INDEX_PATH.is_file() or not CHUNKS_PATH.is_file()):
        st.warning("Embeddings not found. Please run `python src/main.py` first.")

    if st.button("Generate Answer", type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
            st.stop()

        try:
            with st.spinner("Running retrieval pipeline..."):
                contexts = _retrieve_contexts(query, retrieval_mode, data_source, uploaded_files)
        except FileNotFoundError as error:
            st.error(str(error))
            st.stop()
        except ValueError as error:
            st.error(str(error))
            st.stop()
        except Exception as error:
            st.error(f"Failed to process PDFs: {error}")
            st.stop()

        if not contexts:
            st.error("No retrieval results found.")
            st.stop()

        st.subheader("LLM Final Answer")

        if not os.getenv("OPENAI_API_KEY"):
            st.info("OPENAI_API_KEY is not set. Showing retrieval results only.")
            st.code('export OPENAI_API_KEY="your_api_key_here"', language="bash")
        else:
            try:
                with st.spinner("Generating grounded answer with OpenAI..."):
                    answer = generate_answer(query, contexts)
                st.write(answer)
            except ImportError as error:
                st.warning(str(error))
            except Exception as error:
                st.error(f"LLM answer generation failed: {error}")

        st.subheader("Retrieved Contexts")
        for index, context in enumerate(contexts, start=1):
            score_label, score_value = _get_score(context, retrieval_mode)
            with st.expander(
                f"{index}. Chunk {context['chunk_id']} | {score_label}: {score_value:.4f}",
                expanded=index == 1,
            ):
                st.write(f"**chunk_id:** {context['chunk_id']}")
                st.write(f"**source:** {context.get('source', 'unknown')}")
                st.write(f"**score:** {score_value:.4f}")
                st.write(context["text"])


def _retrieve_contexts(
    query: str,
    retrieval_mode: str,
    data_source: str,
    uploaded_files: list | None,
) -> list[dict]:
    if data_source == "Uploaded PDFs":
        return _retrieve_from_uploaded_pdfs(query, retrieval_mode, uploaded_files or [])

    return _retrieve_from_existing_index(query, retrieval_mode)


def _retrieve_from_existing_index(query: str, retrieval_mode: str) -> list[dict]:
    if not INDEX_PATH.is_file() or not CHUNKS_PATH.is_file():
        raise FileNotFoundError("Embeddings not found. Please run `python src/main.py` first.")

    embedder = get_embedder()

    if retrieval_mode == "Hybrid":
        return search_hybrid(
            query,
            top_k=DISPLAY_TOP_K,
            embeddings_dir=str(EMBEDDINGS_DIR),
            embedder=embedder,
        )

    hybrid_candidates = search_hybrid(
        query,
        top_k=RERANKER_CANDIDATE_K,
        embeddings_dir=str(EMBEDDINGS_DIR),
        embedder=embedder,
    )
    reranker = get_reranker()
    return reranker.rerank(query, hybrid_candidates, top_k=DISPLAY_TOP_K)


def _retrieve_from_uploaded_pdfs(query: str, retrieval_mode: str, uploaded_files: list) -> list[dict]:
    if not uploaded_files:
        raise ValueError("Please upload at least one PDF file.")

    documents = _extract_uploaded_documents(uploaded_files)
    chunk_records = _build_chunk_records(documents)
    embedder = get_embedder()
    hybrid_candidates = _search_uploaded_hybrid(query, chunk_records, embedder, top_k=RERANKER_CANDIDATE_K)

    if retrieval_mode == "Hybrid":
        return hybrid_candidates[:DISPLAY_TOP_K]

    reranker = get_reranker()
    return reranker.rerank(query, hybrid_candidates, top_k=DISPLAY_TOP_K)


def _extract_uploaded_documents(uploaded_files: list) -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []

    for uploaded_file in uploaded_files:
        text = _extract_text_from_uploaded_pdf(uploaded_file.name, uploaded_file.getvalue())
        documents.append(
            {
                "source": uploaded_file.name,
                "text": text,
            }
        )

    return documents


def _extract_text_from_uploaded_pdf(file_name: str, file_bytes: bytes) -> str:
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


def _build_chunk_records(documents: list[dict[str, str]]) -> list[dict[str, str]]:
    chunk_records: list[dict[str, str]] = []

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


def _search_uploaded_hybrid(
    query: str,
    chunk_records: list[dict[str, str]],
    embedder: Embedder,
    top_k: int,
    alpha: float = 0.5,
) -> list[dict]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")

    texts = [chunk["text"] for chunk in chunk_records]
    embeddings = np.array(embedder.embed_chunks(texts), dtype=np.float32)
    query_vector = np.array(embedder.embed_chunks([query]), dtype=np.float32)

    dense_results = _search_uploaded_faiss(chunk_records, embeddings, query_vector, top_k=top_k)
    sparse_results = _search_uploaded_bm25(query, chunk_records, top_k=top_k)

    normalized_dense_scores = _normalize_scores(dense_results, "similarity_score")
    normalized_sparse_scores = _normalize_scores(sparse_results, "bm25_score")
    combined_results: dict[int, dict] = {}

    for result in dense_results:
        chunk_id = int(result["chunk_id"])
        combined_results[chunk_id] = {
            "chunk_id": chunk_id,
            "text": result["text"],
            "source": result["source"],
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
            "source": result["source"],
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


def _search_uploaded_faiss(
    chunk_records: list[dict[str, str]],
    embeddings: np.ndarray,
    query_vector: np.ndarray,
    top_k: int,
) -> list[dict]:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    scores, indices = index.search(query_vector, top_k)
    chunk_map = {int(chunk["chunk_id"]): chunk for chunk in chunk_records}
    results: list[dict] = []

    for chunk_index, score in zip(indices[0], scores[0]):
        if chunk_index < 0:
            continue

        normalized_chunk_index = int(chunk_index)
        chunk = chunk_map.get(normalized_chunk_index)
        if chunk is None:
            continue

        results.append(
            {
                "chunk_id": normalized_chunk_index,
                "text": chunk["text"],
                "source": chunk["source"],
                "similarity_score": float(score),
            }
        )

    return results


def _search_uploaded_bm25(query: str, chunk_records: list[dict[str, str]], top_k: int) -> list[dict]:
    tokenized_corpus = [_tokenize(chunk["text"]) for chunk in chunk_records]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:top_k]

    results: list[dict] = []
    for index in ranked_indices:
        chunk = chunk_records[index]
        results.append(
            {
                "chunk_id": int(chunk["chunk_id"]),
                "text": chunk["text"],
                "source": chunk["source"],
                "bm25_score": float(scores[index]),
            }
        )

    return results


def _normalize_scores(results: list[dict], score_key: str) -> dict[int, float]:
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


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _get_score(context: dict, retrieval_mode: str) -> tuple[str, float]:
    if retrieval_mode == "Reranker":
        return "reranker_score", float(context["reranker_score"])

    return "final_score", float(context["final_score"])


if __name__ == "__main__":
    main()

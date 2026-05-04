import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv
load_dotenv()


REPO_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = REPO_ROOT / "embeddings"
INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
CHUNKS_PATH = EMBEDDINGS_DIR / "chunks.json"
DEFAULT_QUERY = "What is a process in operating systems?"
API_BASE_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
RAG_QUERY_URL = f"{API_BASE_URL}/rag/query"

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


st.set_page_config(page_title="Context-Aware Academic Assistant", layout="wide")


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
    use_query_routing = st.checkbox("Use Query Routing", value=False)

    if data_source == "Existing indexed corpus" and (not INDEX_PATH.is_file() or not CHUNKS_PATH.is_file()):
        st.warning("Embeddings not found. Please run `python src/main.py` first.")

    if st.button("Generate Answer", type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
            st.stop()

        try:
            with st.spinner("Running retrieval pipeline..."):
                response_payload = _request_rag(
                    query,
                    retrieval_mode,
                    data_source,
                    uploaded_files,
                    use_query_routing,
                )
                answer = str(response_payload.get("answer", ""))
                contexts = response_payload.get("contexts", [])
                routed_domain = response_payload.get("routed_domain")
        except ValueError as error:
            st.error(str(error))
            st.stop()
        except RuntimeError as error:
            st.error(str(error))
            st.stop()
        except Exception as error:
            st.error(f"Failed to call FastAPI backend: {error}")
            st.stop()

        if not contexts:
            st.error("No retrieval results found.")
            st.stop()

        if use_query_routing:
            if routed_domain is None:
                st.info("Query routing result: no matching domain, using full corpus search.")
            else:
                st.info(f"Query routing result: `{routed_domain}`")

        st.subheader("LLM Final Answer")
        if answer.strip():
            st.write(answer)
        else:
            st.info("No LLM answer was returned. Check the FastAPI backend configuration if you expected one.")

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


def _request_rag(
    query: str,
    retrieval_mode: str,
    data_source: str,
    uploaded_files: list | None,
    use_query_routing: bool,
) -> dict:
    if data_source == "Uploaded PDFs" and not uploaded_files:
        raise ValueError("Please upload at least one PDF file.")

    form_data = {
        "query": query,
        "method": retrieval_mode.lower(),
        "use_routing": str(use_query_routing).lower(),
    }
    request_files = []

    if data_source == "Uploaded PDFs":
        request_files = [
            ("files", (uploaded_file.name, uploaded_file.getvalue(), "application/pdf"))
            for uploaded_file in (uploaded_files or [])
        ]

    try:
        response = requests.post(
            RAG_QUERY_URL,
            data=form_data,
            files=request_files or None,
            timeout=300,
        )
    except requests.RequestException as error:
        raise RuntimeError(
            f"FastAPI backend is not reachable at {RAG_QUERY_URL}. Start it with `uvicorn api.main:app --reload`."
        ) from error

    if response.ok:
        return response.json()

    detail = _extract_error_detail(response)
    raise RuntimeError(detail)


def _extract_error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"FastAPI backend returned HTTP {response.status_code}."

    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail

    return f"FastAPI backend returned HTTP {response.status_code}."


def _get_score(context: dict, retrieval_mode: str) -> tuple[str, float]:
    if retrieval_mode == "Reranker":
        return "reranker_score", float(context["reranker_score"])

    return "final_score", float(context["final_score"])


if __name__ == "__main__":
    main()

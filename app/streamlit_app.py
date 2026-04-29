import os
import sys
from pathlib import Path

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
EMBEDDINGS_DIR = REPO_ROOT / "embeddings"
INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
CHUNKS_PATH = EMBEDDINGS_DIR / "chunks.json"
DEFAULT_QUERY = "What is a process in operating systems?"
DISPLAY_TOP_K = 3

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pipeline.rag_pipeline import RAGPipeline


st.set_page_config(page_title="Context-Aware Academic Assistant", layout="wide")


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(embeddings_dir=str(EMBEDDINGS_DIR))


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
                    answer = get_pipeline().generate_answer(query, contexts)
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
    pipeline = get_pipeline()

    if data_source == "Uploaded PDFs":
        pipeline.build_from_uploaded_files(uploaded_files or [])
        return pipeline.retrieve(query, method=retrieval_mode, top_k=DISPLAY_TOP_K)

    pipeline.use_existing_index()
    return pipeline.retrieve(query, method=retrieval_mode, top_k=DISPLAY_TOP_K)


def _get_score(context: dict, retrieval_mode: str) -> tuple[str, float]:
    if retrieval_mode == "Reranker":
        return "reranker_score", float(context["reranker_score"])

    return "final_score", float(context["final_score"])


if __name__ == "__main__":
    main()

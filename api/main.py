import os
import sys
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
EMBEDDINGS_DIR = REPO_ROOT / "embeddings"
DISPLAY_TOP_K = 3

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from embedding.embedder import Embedder
from pipeline.rag_pipeline import RAGPipeline
from reranker.reranker import Reranker
from routing.query_router import route_query


class UploadedPDF:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


class RagQueryResponse(BaseModel):
    answer: str
    contexts: list[dict]
    routed_domain: str | None


app = FastAPI(title="RAG API", version="1.0.0")


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()


def build_pipeline() -> RAGPipeline:
    return RAGPipeline(
        embeddings_dir=str(EMBEDDINGS_DIR),
        embedder=get_embedder(),
        reranker=get_reranker(),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/rag/query", response_model=RagQueryResponse)
async def rag_query(request: Request) -> RagQueryResponse:
    try:
        form = await request.form()
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail='Failed to parse multipart form data. Install dependencies with `pip install -r requirements.txt`.',
        ) from error

    normalized_query = str(form.get("query", "")).strip()
    normalized_method = str(form.get("method", "")).lower().strip()
    use_routing = _parse_bool(form.get("use_routing", False))
    files = [file for file in form.getlist("files") if isinstance(file, UploadFile)]

    if not normalized_query:
        raise HTTPException(status_code=400, detail="query must not be empty")
    if normalized_method not in {"hybrid", "reranker"}:
        raise HTTPException(status_code=400, detail="method must be either 'hybrid' or 'reranker'")

    pipeline = build_pipeline()

    try:
        if files:
            uploaded_files = await _read_uploaded_files(files)
            pipeline.build_from_uploaded_files(uploaded_files)
        else:
            pipeline.use_existing_index()

        routed_domain = route_query(normalized_query) if use_routing else None
        contexts = pipeline.retrieve(
            normalized_query,
            method=normalized_method,
            top_k=DISPLAY_TOP_K,
            domain_filter=routed_domain,
        )
        answer = _generate_answer(pipeline, normalized_query, contexts)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Failed to process request: {error}") from error

    return RagQueryResponse(
        answer=answer,
        contexts=contexts,
        routed_domain=routed_domain,
    )


async def _read_uploaded_files(files: list[UploadFile]) -> list[UploadedPDF]:
    uploaded_files: list[UploadedPDF] = []

    for file in files:
        file_name = Path(file.filename or "").name or "uploaded.pdf"
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail=f"Uploaded file is empty: {file_name}")

        uploaded_files.append(UploadedPDF(name=file_name, content=file_bytes))

    if not uploaded_files:
        raise HTTPException(status_code=400, detail="Please upload at least one PDF file.")

    return uploaded_files


def _generate_answer(pipeline: RAGPipeline, query: str, contexts: list[dict]) -> str:
    if not contexts:
        return ""

    if not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY is not set. Retrieved contexts are returned without an LLM answer."

    try:
        return pipeline.generate_answer(query, contexts)
    except ImportError as error:
        return str(error)
    except Exception as error:
        return f"LLM answer generation failed: {error}"


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"1", "true", "yes", "on"}

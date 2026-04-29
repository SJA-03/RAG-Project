import os
import sys
from pathlib import Path

from chunking.basic_chunker import chunk_text
from ingestion.pdf_loader import load_pdf, load_pdfs_from_folder


DEFAULT_INPUT_PATH = "data/"
DEFAULT_QUERY = "What is a process in operating systems?"


def main() -> int:
    input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_PATH
    query = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUERY

    try:
        documents = _load_documents(input_path)
        total_characters = sum(len(document["text"]) for document in documents)
        chunk_records = _build_chunk_records(documents)
        from embedding.embedder import Embedder
        from reranker.reranker import Reranker
        from retrieval.bm25_store import search_bm25
        from retrieval.faiss_store import save_faiss_index
        from retrieval.hybrid_searcher import search_hybrid
        from retrieval.searcher import search_chunks

        embedder = Embedder()
        reranker = Reranker()
        embeddings = embedder.embed_chunks([chunk["text"] for chunk in chunk_records])
        index_path, chunks_path = save_faiss_index(embeddings, chunk_records)
        faiss_results = search_chunks(query, embedder=embedder)
        bm25_results = search_bm25(query)
        hybrid_results = search_hybrid(query, embedder=embedder)
        reranked_results = reranker.rerank(query, hybrid_results)
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Unexpected error while processing PDF: {error}", file=sys.stderr)
        return 1

    print(f"Loaded input: {input_path}")
    print(f"Processed PDFs: {len(documents)}")
    print(f"Extracted characters: {total_characters}")
    print(f"Generated chunks: {len(chunk_records)}")
    print(f"Generated embeddings: {len(embeddings)}")

    if chunk_records:
        preview = chunk_records[0]["text"][:200].replace("\n", " ")
        print(f"First chunk preview: {preview}")
        print(f"First chunk source: {chunk_records[0]['source']}")

    print(f"Saved FAISS index to: {index_path}")
    print(f"Saved chunk metadata to: {chunks_path}")
    print("Embedding and FAISS storage completed successfully.")
    print(f"Query: {query}")
    print("FAISS search results:")

    for index, result in enumerate(faiss_results, start=1):
        print(f"{index}. Chunk ID: {result['chunk_id']}")
        print(f"   Source: {result['source']}")
        print(f"   Similarity score: {result['similarity_score']:.4f}")
        print(f"   Chunk text: {result['text']}")

    print("BM25 search results:")

    for index, result in enumerate(bm25_results, start=1):
        print(f"{index}. Chunk ID: {result['chunk_id']}")
        print(f"   Source: {result['source']}")
        print(f"   BM25 score: {result['bm25_score']:.4f}")
        print(f"   Chunk text: {result['text']}")

    print("Hybrid search results:")

    for index, result in enumerate(hybrid_results, start=1):
        print(f"{index}. Chunk ID: {result['chunk_id']}")
        print(f"   Source: {result['source']}")
        print(f"   Dense score: {result['dense_score']:.4f}")
        print(f"   Sparse score: {result['sparse_score']:.4f}")
        print(f"   Final score: {result['final_score']:.4f}")
        print(f"   Chunk text: {result['text']}")

    print("Reranker results:")

    for index, result in enumerate(reranked_results, start=1):
        print(f"{index}. Chunk ID: {result['chunk_id']}")
        print(f"   Source: {result['source']}")
        print(f"   Dense score: {result['dense_score']:.4f}")
        print(f"   Sparse score: {result['sparse_score']:.4f}")
        print(f"   Final score: {result['final_score']:.4f}")
        print(f"   Reranker score: {result['reranker_score']:.4f}")
        print(f"   Chunk text: {result['text']}")

    if not os.getenv("OPENAI_API_KEY"):
        print("LLM answer generation skipped: OPENAI_API_KEY is not set.")
        print('Set it with: export OPENAI_API_KEY="your_api_key_here"')
        return 0

    try:
        from llm.answer_generator import generate_answer

        final_answer = generate_answer(query, reranked_results)
    except ImportError as error:
        print(f"LLM answer generation skipped: {error}")
        return 0
    except Exception as error:
        print(f"LLM answer generation failed: {error}")
        return 0

    print("LLM final answer:")
    print(final_answer)

    return 0

def _load_documents(input_path: str) -> list[dict[str, str]]:
    path = Path(input_path)

    if path.is_dir():
        return load_pdfs_from_folder(input_path)
    if path.is_file():
        return [{"source": str(path), "text": load_pdf(input_path)}]

    raise FileNotFoundError(f"Input path not found: {input_path}")


def _build_chunk_records(documents: list[dict[str, str]]) -> list[dict[str, str]]:
    chunk_records: list[dict[str, str]] = []

    for document in documents:
        for chunk in chunk_text(document["text"]):
            chunk_records.append(
                {
                    "text": chunk,
                    "source": document["source"],
                }
            )

    if not chunk_records:
        raise ValueError("No chunks were generated from the provided PDFs")

    return chunk_records


if __name__ == "__main__":
    raise SystemExit(main())

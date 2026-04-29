import sys

from chunking.basic_chunker import chunk_text
from ingestion.pdf_loader import load_pdf


DEFAULT_PDF_PATH = "data/os/lec1.pdf"
DEFAULT_QUERY = "What is a process in operating systems?"


def main() -> int:
    file_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF_PATH
    query = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUERY

    try:
        text = load_pdf(file_path)
        chunks = chunk_text(text)
        from embedding.embedder import Embedder
        from retrieval.bm25_store import search_bm25
        from retrieval.faiss_store import save_faiss_index
        from retrieval.hybrid_searcher import search_hybrid
        from retrieval.searcher import search_chunks

        embedder = Embedder()
        embeddings = embedder.embed_chunks(chunks)
        index_path, chunks_path = save_faiss_index(embeddings, chunks)
        faiss_results = search_chunks(query, embedder=embedder)
        bm25_results = search_bm25(query)
        hybrid_results = search_hybrid(query, embedder=embedder)
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Unexpected error while processing PDF: {error}", file=sys.stderr)
        return 1

    print(f"Loaded PDF: {file_path}")
    print(f"Extracted characters: {len(text)}")
    print(f"Generated chunks: {len(chunks)}")
    print(f"Generated embeddings: {len(embeddings)}")

    if chunks:
        preview = chunks[0][:200].replace("\n", " ")
        print(f"First chunk preview: {preview}")

    print(f"Saved FAISS index to: {index_path}")
    print(f"Saved chunk metadata to: {chunks_path}")
    print("Embedding and FAISS storage completed successfully.")
    print(f"Query: {query}")
    print("FAISS search results:")

    for index, result in enumerate(faiss_results, start=1):
        print(f"{index}. Chunk ID: {result['chunk_id']}")
        print(f"   Similarity score: {result['similarity_score']:.4f}")
        print(f"   Chunk text: {result['text']}")

    print("BM25 search results:")

    for index, result in enumerate(bm25_results, start=1):
        print(f"{index}. Chunk ID: {result['chunk_id']}")
        print(f"   BM25 score: {result['bm25_score']:.4f}")
        print(f"   Chunk text: {result['text']}")

    print("Hybrid search results:")

    for index, result in enumerate(hybrid_results, start=1):
        print(f"{index}. Chunk ID: {result['chunk_id']}")
        print(f"   Dense score: {result['dense_score']:.4f}")
        print(f"   Sparse score: {result['sparse_score']:.4f}")
        print(f"   Final score: {result['final_score']:.4f}")
        print(f"   Chunk text: {result['text']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

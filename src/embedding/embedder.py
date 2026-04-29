import os

from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name,
            local_files_only=_use_local_files_only(),
        )

    def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        if not chunks:
            raise ValueError("Cannot embed an empty chunk list")

        embeddings = self.model.encode(
            chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()


def _use_local_files_only() -> bool:
    return os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"

import httpx

from loci.config import OllamaSettings

# The embedding model is effectively part of the schema: vectors from two
# different models aren't comparable, so this version string gets stamped
# on every chunk/entity that used it (see normalize.versioning).


class EmbeddingClient:
    def __init__(self, settings: OllamaSettings):
        self.settings = settings
        self.model_version = settings.embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = httpx.post(
            f"{self.settings.base_url}/api/embed",
            json={"model": self.settings.embedding_model, "input": texts},
            timeout=self.settings.request_timeout_s,
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

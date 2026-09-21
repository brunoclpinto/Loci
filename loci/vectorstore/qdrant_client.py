import re
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from loci.config import QdrantSettings


def collection_name_for_model(embedding_model_version: str) -> str:
    # Chunks embedded by different models live in different collections —
    # vectors from two models are never comparable, so mixing them in one
    # collection would silently corrupt similarity search.
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", embedding_model_version)
    return f"loci_chunks__{safe}"


class VectorStore:
    def __init__(self, settings: QdrantSettings):
        self._client = QdrantClient(host=settings.host, port=settings.port, prefer_grpc=settings.prefer_grpc)

    def ensure_collection(self, embedding_model_version: str, vector_size: int) -> str:
        name = collection_name_for_model(embedding_model_version)
        if not self._client.collection_exists(name):
            self._client.create_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
            )
        return name

    def upsert_chunks(self, embedding_model_version: str, points: list[tuple[uuid.UUID, list[float], dict]]) -> None:
        name = collection_name_for_model(embedding_model_version)
        self._client.upsert(
            collection_name=name,
            points=[
                qmodels.PointStruct(id=str(point_id), vector=vector, payload=payload)
                for point_id, vector, payload in points
            ],
        )

    def collection_info(self, embedding_model_version: str) -> qmodels.CollectionInfo | None:
        name = collection_name_for_model(embedding_model_version)
        if not self._client.collection_exists(name):
            return None
        return self._client.get_collection(name)

    def search(
        self,
        embedding_model_version: str,
        query_vector: list[float],
        top_k: int,
        context_ids: list[uuid.UUID] | None = None,
        segment_types: list[str] | None = None,
    ) -> list[qmodels.ScoredPoint]:
        name = collection_name_for_model(embedding_model_version)
        if not self._client.collection_exists(name):
            return []
        if context_ids is not None and len(context_ids) == 0:
            return []
        must: list[qmodels.FieldCondition] = []
        if context_ids is not None:
            must.append(qmodels.FieldCondition(key="context_id", match=qmodels.MatchAny(any=[str(c) for c in context_ids])))
        if segment_types is not None:
            must.append(qmodels.FieldCondition(key="segment_type", match=qmodels.MatchAny(any=segment_types)))
        query_filter = qmodels.Filter(must=must) if must else None
        result = self._client.query_points(
            collection_name=name, query=query_vector, limit=top_k, query_filter=query_filter, with_payload=True
        )
        return result.points

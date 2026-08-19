import uuid
from collections.abc import Sequence

from sqlalchemy.orm import Session

from loci.config import LociSettings
from loci.db.models import Chunk
from loci.embeddings.client import EmbeddingClient
from loci.ingest.base import ExtractedChunk
from loci.vectorstore.qdrant_client import VectorStore


class QdrantChunkEmbedder:
    """Implements the pipeline's ChunkEmbedder contract: embed the chunk
    texts, upsert vectors into Qdrant, and persist the Postgres-side
    `chunks` rows that link back to context/source/qdrant point."""

    def __init__(self, settings: LociSettings):
        self.embedding_client = EmbeddingClient(settings.ollama)
        self.vector_store = VectorStore(settings.qdrant)

    def __call__(
        self,
        session: Session,
        context_id: uuid.UUID,
        source_adapter: str,
        source_ref: str,
        chunks: Sequence[ExtractedChunk],
    ) -> int:
        if not chunks:
            return 0

        vectors = self.embedding_client.embed([c.text for c in chunks])
        model_version = self.embedding_client.model_version
        self.vector_store.ensure_collection(model_version, vector_size=len(vectors[0]))

        points: list[tuple[uuid.UUID, list[float], dict]] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            point_id = uuid.uuid4()
            row = Chunk(
                context_id=context_id,
                source_adapter=source_adapter,
                source_ref=source_ref,
                text=chunk.text,
                token_count=chunk.token_count,
                embedding_model_version=model_version,
                qdrant_point_id=point_id,
                metadata_=chunk.metadata,
            )
            session.add(row)
            points.append(
                (
                    point_id,
                    vector,
                    {
                        "context_id": str(context_id),
                        "source_adapter": source_adapter,
                        "source_ref": source_ref,
                        "text": chunk.text,
                    },
                )
            )

        self.vector_store.upsert_chunks(model_version, points)
        return len(points)

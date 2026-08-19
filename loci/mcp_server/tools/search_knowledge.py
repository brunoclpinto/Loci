import uuid

from qdrant_client.http import models as qmodels
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from loci.config import LociSettings
from loci.db.engine import session_scope
from loci.db.models import Context, Entity
from loci.embeddings.client import EmbeddingClient
from loci.mcp_server.models import ChunkHit, EntitySummary, SearchResult
from loci.mcp_server.repo import resolve_context_ids, to_citation, to_entity_summary
from loci.vectorstore.qdrant_client import VectorStore


def _chunk_hit_from_point(session: Session, point: qmodels.ScoredPoint) -> ChunkHit:
    context_id = uuid.UUID(point.payload["context_id"])
    context = session.get(Context, context_id)
    return ChunkHit(
        text=point.payload["text"],
        score=point.score,
        context_name=context.name if context else str(context_id),
        citation=to_citation(session, point.payload.get("source_adapter"), point.payload.get("source_ref"), context_id),
    )


def make_search_knowledge(settings: LociSettings):
    embedding_client = EmbeddingClient(settings.ollama)
    vector_store = VectorStore(settings.qdrant)

    def search_knowledge(
        query: str,
        context_names: list[str] | None = None,
        include_descendants: bool = False,
        entity_types: list[str] | None = None,
        top_k: int = 10,
    ) -> SearchResult:
        """Search the knowledge base by natural-language query. Returns both
        semantic chunk hits (from the vector layer) and structured entity
        matches (by name), each carrying a citation back to its source and
        context. Restrict `context_names` to scope results to specific
        contexts (e.g. a particular book or a "real_world" context)."""
        with session_scope(settings) as session:
            context_ids = resolve_context_ids(session, context_names, include_descendants)

            query_vector = embedding_client.embed_one(query)
            points = vector_store.search(settings.ollama.embedding_model, query_vector, top_k, context_ids=context_ids)
            chunk_hits = [_chunk_hit_from_point(session, p) for p in points]

            entity_stmt = select(Entity).where(
                or_(func.similarity(Entity.canonical_name, query) > 0.2, Entity.canonical_name.ilike(f"%{query}%"))
            )
            if entity_types:
                entity_stmt = entity_stmt.where(Entity.type.in_(entity_types))
            if context_ids is not None:
                entity_stmt = entity_stmt.where(
                    or_(Entity.context_id.in_(context_ids), Entity.scope == "cross_context")
                )
            entity_stmt = entity_stmt.limit(top_k)
            entity_hits: list[EntitySummary] = [to_entity_summary(session, e) for e in session.scalars(entity_stmt)]

            return SearchResult(chunk_hits=chunk_hits, entity_hits=entity_hits)

    return search_knowledge

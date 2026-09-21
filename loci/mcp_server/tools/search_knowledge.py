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


def run_search(
    settings: LociSettings,
    session: Session,
    query: str,
    context_names: list[str] | None = None,
    include_descendants: bool = False,
    entity_types: list[str] | None = None,
    top_k: int = 10,
    include_metadata_chunks: bool = False,
    embedding_client: EmbeddingClient | None = None,
    vector_store: VectorStore | None = None,
) -> SearchResult:
    """The actual retrieval logic, as a plain function over an existing
    session — reused by both the search_knowledge MCP tool and the bench
    harness (loci/bench/answer.py), so bench results reflect the real
    product's retrieval, not a parallel reimplementation. Callers that will
    issue many queries should build their own EmbeddingClient/VectorStore
    once and pass them in rather than let this construct fresh ones per call.

    Non-narrative chunks (title pages, tables of contents, license text —
    see loci/ingest/structure.py) are excluded by default so document
    metadata doesn't compete with real content in results; pass
    include_metadata_chunks=True to see them anyway."""
    embedding_client = embedding_client or EmbeddingClient(settings.ollama)
    vector_store = vector_store or VectorStore(settings.qdrant)

    context_ids = resolve_context_ids(session, context_names, include_descendants)
    segment_types = None if include_metadata_chunks else ["narrative"]

    query_vector = embedding_client.embed_one(query)
    points = vector_store.search(
        settings.ollama.embedding_model, query_vector, top_k, context_ids=context_ids, segment_types=segment_types
    )
    chunk_hits = [_chunk_hit_from_point(session, p) for p in points]

    entity_stmt = select(Entity).where(
        or_(func.similarity(Entity.canonical_name, query) > 0.2, Entity.canonical_name.ilike(f"%{query}%")),
        Entity.merged_into.is_(None),
    )
    if entity_types:
        entity_stmt = entity_stmt.where(Entity.type.in_(entity_types))
    if context_ids is not None:
        entity_stmt = entity_stmt.where(or_(Entity.context_id.in_(context_ids), Entity.scope == "cross_context"))
    entity_stmt = entity_stmt.limit(top_k)
    entity_hits: list[EntitySummary] = [to_entity_summary(session, e) for e in session.scalars(entity_stmt)]

    return SearchResult(chunk_hits=chunk_hits, entity_hits=entity_hits)


def make_search_knowledge(settings: LociSettings):
    embedding_client = EmbeddingClient(settings.ollama)
    vector_store = VectorStore(settings.qdrant)

    def search_knowledge(
        query: str,
        context_names: list[str] | None = None,
        include_descendants: bool = False,
        entity_types: list[str] | None = None,
        top_k: int = 10,
        include_metadata_chunks: bool = False,
    ) -> SearchResult:
        """Search the knowledge base by natural-language query. Returns both
        semantic chunk hits (from the vector layer) and structured entity
        matches (by name), each carrying a citation back to its source and
        context. Restrict `context_names` to scope results to specific
        contexts (e.g. a particular book or a "real_world" context).
        Document metadata (title pages, tables of contents, license text) is
        excluded by default — set include_metadata_chunks=True to see it."""
        with session_scope(settings) as session:
            return run_search(
                settings,
                session,
                query,
                context_names=context_names,
                include_descendants=include_descendants,
                entity_types=entity_types,
                include_metadata_chunks=include_metadata_chunks,
                top_k=top_k,
                embedding_client=embedding_client,
                vector_store=vector_store,
            )

    return search_knowledge

import logging
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from loci.config import LociSettings
from loci.db.engine import session_scope
from loci.db.models import Context, Entity, IngestionRun, Relationship
from loci.ingest.base import ExtractedChunk, ExtractionBatch, SourceAdapter, SourceDescriptor
from loci.ingest.coreference import resolve_unresolved_entities
from loci.ingest.relationships import relationship_exists
from loci.ingest.type_consolidation import resolve_type_inconsistencies
from loci.normalize.resolver import EntityResolver
from loci.normalize.versioning import RESOLVER_VERSION
from loci.normalize.vocabulary import VocabularyError, validate_predicate_usage
from loci.schema_registry.loader import SchemaRegistry
from loci.schema_registry.validator import validate_attributes

logger = logging.getLogger(__name__)

# (session, context_id, adapter_name, source_ref, chunks) -> number of chunks embedded
ChunkEmbedder = Callable[[Session, uuid.UUID, str, str, list[ExtractedChunk]], int]


@dataclass
class IngestionStats:
    entities_created: int = 0
    entities_matched: int = 0
    relationships_created: int = 0
    relationships_matched: int = 0
    relationships_skipped_invalid: int = 0
    chunks_embedded: int = 0


class IngestionPipeline:
    """The single adapter-agnostic path every source funnels through:
    validate attributes -> resolve entities -> validate/persist
    relationships -> embed chunks. Adapters never touch the resolver,
    embedder, or Postgres directly, which is what structurally enforces the
    shared-normalization invariant rather than leaving it to convention."""

    def __init__(self, settings: LociSettings, resolver: EntityResolver, chunk_embedder: ChunkEmbedder | None = None):
        self.settings = settings
        self.resolver = resolver
        self.chunk_embedder = chunk_embedder

    def ingest(self, adapter: SourceAdapter, source: SourceDescriptor) -> IngestionStats:
        run_id = self._start_run(adapter.name, str(source.path))
        stats = IngestionStats()
        try:
            with session_scope(self.settings) as session:
                registry = SchemaRegistry.load(session)
                for batch in adapter.extract(source):
                    self._process_batch(session, registry, adapter.name, batch, stats)
                # Whole-document coreference pass: identities revealed only
                # later than their first mention (see loci/ingest/
                # coreference.py) can only be resolved once every window has
                # been extracted. A no-op for adapters that never produce
                # identity_status='unresolved' entities (e.g. structured
                # files).
                context = self._get_or_create_context(session, source.context_name)
                merged = resolve_unresolved_entities(session, context.id, self.settings)
                if merged:
                    logger.info("coreference resolution merged %d entities in %r", merged, source.context_name)
                # Same idea, different axis: the same entity sometimes gets
                # tagged with different types in different windows (e.g.
                # "Jefferson Hope" as both Person and Location), which the
                # resolver can't catch since it only matches within a type.
                type_merged = resolve_type_inconsistencies(session, context.id, self.settings, registry)
                if type_merged:
                    logger.info("type consolidation merged %d entities in %r", type_merged, source.context_name)
            self._finish_run(run_id, "succeeded", stats)
        except Exception as exc:
            self._finish_run(run_id, "failed", stats, error=str(exc))
            raise
        return stats

    def _process_batch(
        self, session: Session, registry: SchemaRegistry, adapter_name: str, batch: ExtractionBatch, stats: IngestionStats
    ) -> None:
        context = self._get_or_create_context(session, batch.context_name)
        local_id_map: dict[str, uuid.UUID] = {}

        for ee in batch.entities:
            schema_version = validate_attributes(registry, ee.type, ee.attributes)
            candidate = self.resolver.find_candidate(session, ee.type, ee.canonical_name, context.id)
            if candidate is not None:
                incoming_aliases = set(ee.aliases)
                if ee.canonical_name != candidate.canonical_name:
                    incoming_aliases.add(ee.canonical_name)
                merged_aliases = sorted(set(candidate.aliases) | incoming_aliases)
                if merged_aliases != candidate.aliases:
                    candidate.aliases = merged_aliases
                local_id_map[ee.local_id] = candidate.id
                stats.entities_matched += 1
            else:
                entity = Entity(
                    type=ee.type,
                    context_id=context.id,
                    scope=ee.scope,
                    canonical_name=ee.canonical_name,
                    identity_status=ee.identity_status,
                    aliases=sorted(set(ee.aliases)),
                    attributes=ee.attributes,
                    attribute_schema_version=schema_version,
                    resolver_version=RESOLVER_VERSION,
                    source_adapter=adapter_name,
                    source_ref=batch.source_ref,
                )
                session.add(entity)
                session.flush()
                local_id_map[ee.local_id] = entity.id
                stats.entities_created += 1

        for er in batch.relationships:
            subject_id = local_id_map.get(er.subject_local_id)
            if subject_id is None:
                raise ValueError(f"relationship references unknown local_id {er.subject_local_id!r}")
            object_id = local_id_map.get(er.object_local_id) if er.object_local_id else None
            if er.object_local_id and object_id is None:
                raise ValueError(f"relationship references unknown local_id {er.object_local_id!r}")

            subject_entity = session.get(Entity, subject_id)
            object_entity = session.get(Entity, object_id) if object_id else None
            try:
                validate_predicate_usage(
                    registry, er.predicate, subject_entity.type, object_entity.type if object_entity else None
                )
            except VocabularyError as exc:
                # A single bad triple (most often an imprecise LLM extraction)
                # shouldn't discard an entire batch of otherwise-good entities
                # and relationships — skip it and keep going.
                logger.warning("skipping invalid relationship in %s batch: %s", adapter_name, exc)
                stats.relationships_skipped_invalid += 1
                continue

            if relationship_exists(session, subject_id, er.predicate, object_id, er.object_literal, context.id):
                stats.relationships_matched += 1
                continue

            session.add(
                Relationship(
                    subject_id=subject_id,
                    predicate=er.predicate,
                    object_id=object_id,
                    object_literal=er.object_literal,
                    context_id=context.id,
                    attributes=er.attributes,
                    confidence=er.confidence,
                    source_adapter=adapter_name,
                    source_ref=batch.source_ref,
                    resolver_version=RESOLVER_VERSION,
                )
            )
            stats.relationships_created += 1

        if batch.chunks and self.chunk_embedder is not None:
            stats.chunks_embedded += self.chunk_embedder(
                session, context.id, adapter_name, batch.source_ref, batch.chunks
            )

        session.flush()

    def _get_or_create_context(self, session: Session, name: str) -> Context:
        context = session.scalars(select(Context).where(Context.name == name)).first()
        if context is None:
            context = Context(name=name)
            session.add(context)
            session.flush()
        return context

    def _start_run(self, adapter_name: str, source_ref: str) -> uuid.UUID:
        with session_scope(self.settings) as session:
            run = IngestionRun(adapter=adapter_name, source_ref=source_ref, status="running")
            session.add(run)
            session.flush()
            return run.id

    def _finish_run(self, run_id: uuid.UUID, status: str, stats: IngestionStats, error: str | None = None) -> None:
        with session_scope(self.settings) as session:
            run = session.get(IngestionRun, run_id)
            run.status = status
            run.stats = asdict(stats)
            run.error = error
            run.completed_at = datetime.now(timezone.utc)

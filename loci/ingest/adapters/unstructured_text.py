import json
import logging
from collections.abc import Iterable
from pathlib import Path

import httpx
import tenacity

from loci.chunking.paragraph_grouping import group_paragraphs
from loci.chunking.splitter import split_text
from loci.config import LociSettings
from loci.db.engine import session_scope
from loci.ingest.base import (
    ExtractedChunk,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionBatch,
    SourceAdapter,
    SourceDescriptor,
)
from loci.ingest.entity_reference import build_name_lookup, resolve_relationship_endpoint
from loci.ingest.entity_registry import EntityRegistry
from loci.ingest.structure import Segment, segment_document
from loci.llm.client import classify_segment, discover_entities, extract_relationships
from loci.normalize.vocabulary import VocabularyError, validate_predicate_usage
from loci.schema_registry.loader import EntityTypeSchema, EnumValueDefinition, PredicateDefinition, SchemaRegistry
from loci.schema_registry.validator import AttributeValidationError, validate_attributes

logger = logging.getLogger(__name__)


class UnstructuredTextAdapter(SourceAdapter):
    """LLM-based extraction for free text (text/markdown/PDF). Fixes the
    known gaps of the old regex/dependency-parsing approach (pronouns,
    passive voice) by prompting a local model constrained to the live
    entity-type/predicate vocabulary, validated and retried on schema
    failure. Chunk embedding proceeds independently of extraction success —
    semantic search isn't blocked by extraction quality.

    Raw text is first split into typed segments (see loci/ingest/structure.py)
    so document metadata (title pages, tables of contents, license text)
    doesn't get extracted as if it were narrative content. Only `narrative`
    segments feed extraction; every segment (narrative or not) is still
    chunked and embedded, tagged with its segment type.

    Extraction is two full sequential passes over the document rather than
    one combined pass per window:

    1. Entity discovery, at window granularity (~2500 words), building one
       complete, permanently-id'd EntityRegistry as it walks every narrative
       segment in order.
    2. Relationship extraction, at finer paragraph-grouped granularity, over
       the same narrative segments, referencing only the now-frozen
       registry. subject_id/object_id are JSON-schema enum-constrained to
       the live registry, and resolve_relationship_endpoint (also used by
       any endpoint that still slips through slightly malformed) resolves
       the rest.

    This structurally prevents the single-pass design's stale-local_id
    relationship drops (a relationship could reference a known entity's
    hint id from an earlier window, which had no guaranteed relationship to
    the *current* window's own id namespace): relationships are now never
    extracted before every entity they could reference already has a
    permanent id, so there's no separate id namespace to go stale against."""

    name = "unstructured_text"

    def __init__(self, settings: LociSettings):
        self.settings = settings

    def can_handle(self, source: SourceDescriptor) -> bool:
        if source.forced_adapter:
            return source.forced_adapter in {"text", "pdf", "md", "unstructured_text"}
        return source.path.suffix.lower() in {".txt", ".md", ".pdf"}

    def extract(self, source: SourceDescriptor) -> Iterable[ExtractionBatch]:
        with session_scope(self.settings) as session:
            registry = SchemaRegistry.load(session)

        entity_types = [registry.entity_types[name] for name in sorted(registry.entity_types)]
        predicates = [registry.predicates[name] for name in sorted(registry.predicates)]
        scope_values = registry.enum_values("entity_scope")
        identity_status_values = registry.enum_values("entity_identity_status")
        segment_type_values = registry.enum_values("segment_type")

        text = self._load_text(source.path)

        def _classify(block_text: str) -> str:
            return classify_segment(
                self.settings.ollama, self.settings.ingest.segment_classifier_model, block_text, segment_type_values
            )

        segments = segment_document(text, classify_fn=_classify)

        entity_registry = EntityRegistry()
        narrative_segments: list[Segment] = []

        # Pass 1: entity discovery. Non-narrative segments are chunked and
        # embedded here (unchanged from the old single-pass design) but
        # never fed to extraction at all.
        for segment in segments:
            embed_chunks = split_text(
                segment.text, self.settings.ingest.chunk_tokens, self.settings.ingest.chunk_overlap_tokens
            )
            chunks = [
                ExtractedChunk(text=c.text, token_count=c.token_count, metadata={"segment_type": segment.type})
                for c in embed_chunks
            ]

            if segment.type != "narrative":
                yield ExtractionBatch(
                    context_name=source.context_name,
                    chunks=chunks,
                    source_ref=str(source.path),
                    debug={"phase": "segment", "segment_type": segment.type},
                )
                continue

            narrative_segments.append(segment)
            windows = split_text(segment.text, self.settings.ingest.extraction_window_tokens, overlap_tokens=0)
            for window in windows:
                known_entities_before = entity_registry.as_hint_list()
                new_entities, call_debug = self._discover_window(
                    window.text, registry, entity_types, scope_values, identity_status_values, entity_registry
                )
                yield ExtractionBatch(
                    context_name=source.context_name,
                    entities=new_entities,
                    chunks=chunks,
                    source_ref=str(source.path),
                    debug={
                        "phase": "discovery",
                        "segment_type": segment.type,
                        "known_entities_before": known_entities_before or None,
                        **call_debug,
                    },
                )
                chunks = []  # only attach each window's chunks to its own batch once

        # Pass 2: relationship extraction, now that every entity the
        # document contains already has a permanent registry id. Skipped
        # entirely if discovery found nothing to relate.
        if not entity_registry.as_hint_list():
            return

        for segment in narrative_segments:
            units = group_paragraphs(segment.paragraphs, self.settings.ingest.relationship_unit_tokens)
            for unit in units:
                relationships, referenced_entities, call_debug = self._extract_relationships_unit(
                    unit.text, registry, predicates, entity_registry
                )
                if not relationships:
                    continue
                yield ExtractionBatch(
                    context_name=source.context_name,
                    entities=referenced_entities,
                    relationships=relationships,
                    source_ref=str(source.path),
                    debug={"phase": "relationship", "segment_type": segment.type, **call_debug},
                )

    def _discover_window(
        self,
        window_text: str,
        registry: SchemaRegistry,
        entity_types: list[EntityTypeSchema],
        scope_values: list[EnumValueDefinition],
        identity_status_values: list[EnumValueDefinition],
        entity_registry: EntityRegistry,
    ) -> tuple[list[ExtractedEntity], dict]:
        retryer = tenacity.Retrying(
            stop=tenacity.stop_after_attempt(self.settings.ingest.extraction_json_retries),
            retry=tenacity.retry_if_exception_type((httpx.HTTPError, json.JSONDecodeError)),
            reraise=True,
        )
        call_debug: dict = {}
        try:
            for attempt in retryer:
                with attempt:
                    raw_entities = discover_entities(
                        self.settings.ollama,
                        self.settings.ollama.chat_model,
                        window_text,
                        entity_types,
                        scope_values,
                        identity_status_values,
                        known_entities=entity_registry.as_hint_list() or None,
                        debug_sink=call_debug.update,
                    )
                    return self._register_discovered(raw_entities, registry, entity_registry), call_debug
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            logger.warning("entity discovery failed after retries for window (len=%d chars): %s", len(window_text), exc)
            return [], call_debug
        return [], call_debug

    def _register_discovered(
        self, raw_entities: list[dict], registry: SchemaRegistry, entity_registry: EntityRegistry
    ) -> list[ExtractedEntity]:
        validated: list[dict] = []
        for e in raw_entities:
            attributes = e.get("attributes") or {}
            try:
                validate_attributes(registry, e["type"], attributes)
            except AttributeValidationError as exc:
                logger.warning("dropping invalid attributes for entity %r: %s", e.get("canonical_name"), exc)
                attributes = {}
            validated.append({**e, "attributes": attributes})

        new_entities = entity_registry.merge(validated)
        return [
            ExtractedEntity(
                local_id=e.id,
                type=e.type,
                canonical_name=e.canonical_name,
                aliases=e.aliases,
                attributes=e.attributes,
                scope=e.scope,
                identity_status=e.identity_status,
            )
            for e in new_entities
        ]

    def _extract_relationships_unit(
        self, unit_text: str, registry: SchemaRegistry, predicates: list[PredicateDefinition], entity_registry: EntityRegistry
    ) -> tuple[list[ExtractedRelationship], list[ExtractedEntity], dict]:
        registry_hint = entity_registry.as_hint_list()

        retryer = tenacity.Retrying(
            stop=tenacity.stop_after_attempt(self.settings.ingest.extraction_json_retries),
            retry=tenacity.retry_if_exception_type((httpx.HTTPError, json.JSONDecodeError)),
            reraise=True,
        )
        call_debug: dict = {}
        try:
            for attempt in retryer:
                with attempt:
                    raw_relationships = extract_relationships(
                        self.settings.ollama,
                        self.settings.ollama.chat_model,
                        unit_text,
                        predicates,
                        registry_hint,
                        debug_sink=call_debug.update,
                    )
                    relationships, referenced = self._resolve_relationships(raw_relationships, registry, registry_hint)
                    return relationships, referenced, call_debug
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            logger.warning(
                "relationship extraction failed after retries for unit (len=%d chars): %s", len(unit_text), exc
            )
            return [], [], call_debug
        return [], [], call_debug

    def _resolve_relationships(
        self, raw_relationships: list[dict], registry: SchemaRegistry, registry_hint: list[dict]
    ) -> tuple[list[ExtractedRelationship], list[ExtractedEntity]]:
        # Pass 2 never introduces brand-new entities — every subject_id/
        # object_id the model emits is schema-enum-constrained to
        # registry_hint's own ids, so `local_ids` (the "this response's own
        # namespace" precedence tier) is deliberately always empty here;
        # resolve_relationship_endpoint's known-id/known-name tiers do all
        # the work, exactly as they do for a known-entity reference in Part
        # 1's fix (see entity_reference.py).
        known_by_id = {e["local_id"]: e for e in registry_hint}
        known_by_name = build_name_lookup(registry_hint)
        type_by_local_id: dict[str, str] = {}
        referenced: list[ExtractedEntity] = []
        synthesized: set[str] = set()

        def _resolve(raw_id: str, label: str) -> str | None:
            resolved_id, known_entity = resolve_relationship_endpoint(raw_id, set(), known_by_id, known_by_name)
            if resolved_id is None:
                logger.warning("skipping relationship with unknown %s %r", label, raw_id)
                return None
            if resolved_id not in synthesized:
                entity = known_entity if known_entity is not None else known_by_id[resolved_id]
                referenced.append(
                    ExtractedEntity(
                        local_id=resolved_id,
                        type=entity["type"],
                        canonical_name=entity["canonical_name"],
                        aliases=entity.get("aliases", []),
                    )
                )
                type_by_local_id[resolved_id] = entity["type"]
                synthesized.add(resolved_id)
            return resolved_id

        relationships: list[ExtractedRelationship] = []
        for r in raw_relationships:
            subject_id = _resolve(r["subject_id"], "subject_id")
            if subject_id is None:
                continue
            raw_object_id = r.get("object_id")
            object_id = _resolve(raw_object_id, "object_id") if raw_object_id else None
            if raw_object_id and object_id is None:
                continue
            if not object_id and not r.get("object_literal"):
                logger.warning("skipping relationship with neither object_id nor object_literal: %r", r)
                continue
            try:
                validate_predicate_usage(
                    registry,
                    r["predicate"],
                    type_by_local_id[subject_id],
                    type_by_local_id.get(object_id) if object_id else None,
                )
            except VocabularyError as exc:
                # The model sometimes gets the relation direction backwards
                # (e.g. emits "Organization works_for Person" instead of
                # "Person works_for Organization") — a deterministic,
                # zero-cost check: if swapping subject/object satisfies the
                # predicate's type constraint, use the swap instead of
                # discarding a fact the model otherwise extracted correctly.
                # Only applicable when both endpoints are entities — an
                # object_literal has no type to swap into subject position.
                if object_id is None:
                    logger.warning("skipping relationship failing vocabulary check: %s", exc)
                    continue
                try:
                    validate_predicate_usage(
                        registry, r["predicate"], type_by_local_id[object_id], type_by_local_id[subject_id]
                    )
                except VocabularyError:
                    logger.warning("skipping relationship failing vocabulary check: %s", exc)
                    continue
                logger.info("swapped reversed subject/object for predicate %r to satisfy vocabulary", r["predicate"])
                subject_id, object_id = object_id, subject_id
            relationships.append(
                ExtractedRelationship(
                    subject_local_id=subject_id,
                    predicate=r["predicate"],
                    object_local_id=object_id,
                    object_literal=r.get("object_literal"),
                )
            )

        return relationships, referenced

    def _load_text(self, path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        return path.read_text()

import json
import logging
from collections.abc import Iterable
from pathlib import Path

import httpx
import tenacity

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
from loci.llm.client import ExtractionLLMClient
from loci.normalize.vocabulary import VocabularyError, validate_predicate_usage
from loci.schema_registry.loader import SchemaRegistry
from loci.schema_registry.validator import AttributeValidationError, validate_attributes

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "llm" / "prompts" / "extract_entities_relationships.md"


class ExtractionValidationError(ValueError):
    pass


class UnstructuredTextAdapter(SourceAdapter):
    """LLM-based extraction for free text (text/markdown/PDF). Fixes the
    known gaps of the old regex/dependency-parsing approach (pronouns,
    passive voice) by prompting a local model constrained to the live
    entity-type/predicate vocabulary, validated and retried on schema
    failure. Chunk embedding proceeds independently of extraction success —
    semantic search isn't blocked by extraction quality."""

    name = "unstructured_text"

    def __init__(self, settings: LociSettings):
        self.settings = settings
        self.llm_client = ExtractionLLMClient(settings.ollama, _PROMPT_PATH.read_text())

    def can_handle(self, source: SourceDescriptor) -> bool:
        if source.forced_adapter:
            return source.forced_adapter in {"text", "pdf", "md", "unstructured_text"}
        return source.path.suffix.lower() in {".txt", ".md", ".pdf"}

    def extract(self, source: SourceDescriptor) -> Iterable[ExtractionBatch]:
        with session_scope(self.settings) as session:
            registry = SchemaRegistry.load(session)

        entity_types = sorted(registry.entity_types)
        predicates = sorted(registry.predicates)

        text = self._load_text(source.path)
        windows = split_text(text, self.settings.ingest.extraction_window_tokens, overlap_tokens=0)

        for window in windows:
            entities, relationships = self._extract_window(window.text, registry, entity_types, predicates)

            embed_chunks = split_text(
                window.text, self.settings.ingest.chunk_tokens, self.settings.ingest.chunk_overlap_tokens
            )

            yield ExtractionBatch(
                context_name=source.context_name,
                entities=entities,
                relationships=relationships,
                chunks=[ExtractedChunk(text=c.text, token_count=c.token_count) for c in embed_chunks],
                source_ref=str(source.path),
            )

    def _extract_window(
        self, window_text: str, registry: SchemaRegistry, entity_types: list[str], predicates: list[str]
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelationship]]:
        # Retries only cover genuinely broken responses (bad JSON, HTTP
        # errors) — an LLM extraction is inherently imperfect, so individual
        # bad items (a dangling relationship reference, an invalid
        # attribute) are skipped in _validate_and_convert rather than
        # discarding an entire window's worth of otherwise-good extraction.
        retryer = tenacity.Retrying(
            stop=tenacity.stop_after_attempt(self.settings.ingest.extraction_json_retries),
            retry=tenacity.retry_if_exception_type((ExtractionValidationError, httpx.HTTPError, json.JSONDecodeError)),
            reraise=True,
        )
        try:
            for attempt in retryer:
                with attempt:
                    raw = self.llm_client.extract(window_text, entity_types, predicates)
                    return self._validate_and_convert(raw, registry)
        except (ExtractionValidationError, httpx.HTTPError, json.JSONDecodeError) as exc:
            logger.warning("extraction failed after retries for window (len=%d chars): %s", len(window_text), exc)
            return [], []
        return [], []

    def _validate_and_convert(
        self, raw: dict, registry: SchemaRegistry
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelationship]]:
        raw_entities = raw.get("entities", [])
        raw_relationships = raw.get("relationships", [])
        if not raw_entities and not raw_relationships:
            raise ExtractionValidationError("response had no entities and no relationships")
        local_ids = {e["local_id"] for e in raw_entities}

        entities: list[ExtractedEntity] = []
        type_by_local_id: dict[str, str] = {}
        for e in raw_entities:
            attributes = e.get("attributes") or {}
            try:
                validate_attributes(registry, e["type"], attributes)
            except AttributeValidationError as exc:
                logger.warning("dropping invalid attributes for entity %r: %s", e.get("canonical_name"), exc)
                attributes = {}
            entities.append(
                ExtractedEntity(
                    local_id=e["local_id"],
                    type=e["type"],
                    canonical_name=e["canonical_name"],
                    aliases=e.get("aliases") or [],
                    attributes=attributes,
                    scope=e.get("scope") or "context_local",
                )
            )
            type_by_local_id[e["local_id"]] = e["type"]

        relationships: list[ExtractedRelationship] = []
        for r in raw_relationships:
            object_local_id = r.get("object_local_id")
            if r["subject_local_id"] not in local_ids:
                logger.warning("skipping relationship with unknown subject_local_id %r", r["subject_local_id"])
                continue
            if object_local_id and object_local_id not in local_ids:
                logger.warning("skipping relationship with unknown object_local_id %r", object_local_id)
                continue
            if not object_local_id and not r.get("object_literal"):
                logger.warning("skipping relationship with neither object_local_id nor object_literal: %r", r)
                continue
            try:
                validate_predicate_usage(
                    registry,
                    r["predicate"],
                    type_by_local_id[r["subject_local_id"]],
                    type_by_local_id.get(object_local_id) if object_local_id else None,
                )
            except VocabularyError as exc:
                logger.warning("skipping relationship failing vocabulary check: %s", exc)
                continue
            relationships.append(
                ExtractedRelationship(
                    subject_local_id=r["subject_local_id"],
                    predicate=r["predicate"],
                    object_local_id=object_local_id,
                    object_literal=r.get("object_literal"),
                )
            )

        return entities, relationships

    def _load_text(self, path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        return path.read_text()

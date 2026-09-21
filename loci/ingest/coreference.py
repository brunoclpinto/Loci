import logging
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from loci.config import LociSettings
from loci.db.models import Entity
from loci.ingest.relationships import reassign_entity_relationships
from loci.llm.client import match_coreferences

logger = logging.getLogger(__name__)


def resolve_unresolved_entities(session: Session, context_id: uuid.UUID, settings: LociSettings) -> int:
    """Whole-document coreference pass: for every entity still marked
    identity_status='unresolved' in this context (a real individual whose
    identity wasn't known when first mentioned — see the extraction prompt),
    ask the model whether the rest of the document later reveals it to be
    one of the context's named entities. On a match, the unresolved
    entity's relationships are reassigned to the canonical named entity,
    its descriptive name is kept as an alias, and its own row is marked
    `merged_into` rather than deleted — it stays auditable, it's just no
    longer resolved/matched as a live entity (see EntityResolver and
    run_search, both of which exclude merged_into IS NOT NULL rows).

    Runs once per ingest call, after the full per-window extraction loop —
    it genuinely needs the whole-document picture, since the identity that
    resolves an early "the mysterious man" mention might not appear until
    many windows later. Returns the number of merges performed."""
    unresolved = session.scalars(
        select(Entity).where(
            Entity.context_id == context_id,
            Entity.identity_status == "unresolved",
            Entity.merged_into.is_(None),
        )
    ).all()
    if not unresolved:
        return 0

    named = session.scalars(
        select(Entity).where(
            Entity.context_id == context_id,
            Entity.identity_status == "named",
            Entity.merged_into.is_(None),
        )
    ).all()
    if not named:
        return 0

    unresolved_payload = [
        {"id": str(e.id), "type": e.type, "canonical_name": e.canonical_name, "attributes": e.attributes}
        for e in unresolved
    ]
    named_payload = [
        {"id": str(e.id), "type": e.type, "canonical_name": e.canonical_name, "aliases": e.aliases} for e in named
    ]

    matches = match_coreferences(settings.ollama, settings.ingest.coreference_model, unresolved_payload, named_payload)
    named_by_id = {str(e.id): e for e in named}

    merged_count = 0
    for entity in unresolved:
        matched_id = matches.get(str(entity.id))
        if not matched_id:
            continue
        canonical = named_by_id.get(matched_id)
        if canonical is None:
            logger.warning("coreference match referenced unknown named entity id %r; skipping", matched_id)
            continue
        if canonical.type != entity.type:
            logger.warning(
                "skipping coreference match with mismatched types: %r (%s) -> %r (%s)",
                entity.canonical_name,
                entity.type,
                canonical.canonical_name,
                canonical.type,
            )
            continue

        reassign_entity_relationships(session, entity.id, canonical.id)
        canonical.aliases = sorted(set(canonical.aliases) | {entity.canonical_name})
        entity.merged_into = canonical.id
        merged_count += 1
        logger.info("coreference merge: %r -> %r", entity.canonical_name, canonical.canonical_name)

    session.flush()
    return merged_count

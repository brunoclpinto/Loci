import logging
import uuid
from difflib import SequenceMatcher

from sqlalchemy import select
from sqlalchemy.orm import Session

from loci.config import LociSettings
from loci.db.models import Entity, Relationship
from loci.ingest.relationships import reassign_entity_relationships
from loci.llm.client import match_type_conflicts
from loci.normalize.vocabulary import VocabularyError, validate_predicate_usage
from loci.schema_registry.loader import SchemaRegistry

logger = logging.getLogger(__name__)

# Looser than the resolver's own match threshold — this only decides which
# pairs are worth putting in front of the model for review, not an
# auto-merge decision, so it can afford to over-include candidates.
_NAME_SIMILARITY_THRESHOLD = 0.6


def _find_cross_type_groups(entities: list[Entity]) -> list[list[Entity]]:
    """Groups entities whose canonical_name is near-identical (case
    -insensitive) but whose type differs — candidates for the same
    real-world entity having been mistyped in some extraction window,
    which the resolver can't catch on its own since it only matches within
    a type."""
    groups: list[list[Entity]] = []
    used: set[uuid.UUID] = set()
    for i, a in enumerate(entities):
        if a.id in used:
            continue
        group = [a]
        for b in entities[i + 1 :]:
            if b.id in used or b.type == a.type:
                continue
            ratio = SequenceMatcher(None, a.canonical_name.lower(), b.canonical_name.lower()).ratio()
            if ratio >= _NAME_SIMILARITY_THRESHOLD:
                group.append(b)
        if len(group) > 1:
            groups.append(group)
            used.update(e.id for e in group)
    return groups


def resolve_type_inconsistencies(
    session: Session, context_id: uuid.UUID, settings: LociSettings, registry: SchemaRegistry
) -> int:
    """Different extraction windows sometimes tag the same real-world
    entity with different types (e.g. "Jefferson Hope" once as Person,
    once as Location) — EntityResolver only matches within a type, so
    these never converge on their own. This whole-document pass groups
    same-context entities with similar names but differing types, asks the
    model which groups are genuinely the same mistyped entity, and merges
    them the same way loci/ingest/coreference.py does: relationships
    reassigned (skipping anything that would duplicate an existing one),
    aliases kept, losing rows marked merged_into rather than deleted.

    Because the surviving entity's type can differ from a merged entity's
    original type, every reassigned relationship is re-validated against
    the predicate vocabulary afterward — one that's no longer valid (e.g. a
    `located_in` edge whose object used to be a Location and is now a
    Person) is removed rather than left corrupt. Returns the number of
    merges performed."""
    named = list(
        session.scalars(
            select(Entity).where(
                Entity.context_id == context_id,
                Entity.identity_status == "named",
                Entity.merged_into.is_(None),
            )
        )
    )
    groups = _find_cross_type_groups(named)
    if not groups:
        return 0

    payload_groups = [
        [{"id": str(e.id), "type": e.type, "canonical_name": e.canonical_name, "aliases": e.aliases} for e in group]
        for group in groups
    ]
    decisions = match_type_conflicts(settings.ollama, settings.ingest.coreference_model, payload_groups)
    entities_by_id = {str(e.id): e for group in groups for e in group}

    reassigned_relationship_ids: set[uuid.UUID] = set()
    merged_count = 0
    for canonical_id, duplicate_ids in decisions.items():
        canonical = entities_by_id.get(canonical_id)
        if canonical is None:
            continue
        for dup_id in duplicate_ids:
            duplicate = entities_by_id.get(dup_id)
            if duplicate is None or duplicate.id == canonical.id:
                continue
            reassigned_relationship_ids |= reassign_entity_relationships(session, duplicate.id, canonical.id)
            canonical.aliases = sorted(set(canonical.aliases) | {duplicate.canonical_name} | set(duplicate.aliases))
            duplicate.merged_into = canonical.id
            merged_count += 1
            logger.info(
                "type-consolidation merge: %r (%s) -> %r (%s)",
                duplicate.canonical_name,
                duplicate.type,
                canonical.canonical_name,
                canonical.type,
            )
    session.flush()

    for rel_id in reassigned_relationship_ids:
        rel = session.get(Relationship, rel_id)
        if rel is None:
            continue
        subject = session.get(Entity, rel.subject_id)
        obj = session.get(Entity, rel.object_id) if rel.object_id else None
        try:
            validate_predicate_usage(registry, rel.predicate, subject.type, obj.type if obj else None)
        except VocabularyError as exc:
            logger.info("dropping relationship invalidated by type-consolidation merge: %s", exc)
            session.delete(rel)
    session.flush()

    return merged_count

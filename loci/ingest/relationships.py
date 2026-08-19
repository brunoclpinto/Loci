import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from loci.db.models import Relationship


def relationship_exists(
    session: Session,
    subject_id: uuid.UUID,
    predicate: str,
    object_id: uuid.UUID | None,
    object_literal: dict | None,
    context_id: uuid.UUID,
) -> bool:
    """Shared by the ingestion pipeline (loci/ingest/pipeline.py) and the
    coreference-merge pass (loci/ingest/coreference.py) — both need the
    same "is this triple already recorded" check, the first to avoid
    duplicating on re-ingest, the second to avoid duplicating when
    reassigning a merged entity's relationships onto its canonical entity."""
    stmt = select(Relationship.id).where(
        Relationship.subject_id == subject_id,
        Relationship.predicate == predicate,
        Relationship.context_id == context_id,
    )
    if object_id is not None:
        stmt = stmt.where(Relationship.object_id == object_id)
    else:
        stmt = stmt.where(Relationship.object_id.is_(None), Relationship.object_literal == object_literal)
    return session.scalars(stmt.limit(1)).first() is not None


def reassign_entity_relationships(session: Session, old_id: uuid.UUID, canonical_id: uuid.UUID) -> set[uuid.UUID]:
    """Move every relationship touching `old_id` onto `canonical_id`,
    skipping any that would duplicate a relationship the canonical entity
    already has. Shared by both the coreference merge pass (an unresolved
    mention turns out to be a named entity) and the type-consolidation pass
    (the same entity was mistyped in different windows). Returns the ids of
    relationships actually reassigned, so a caller that changes an entity's
    effective type can re-validate just those against the vocabulary."""
    reassigned: set[uuid.UUID] = set()
    for rel in session.scalars(select(Relationship).where(Relationship.subject_id == old_id)).all():
        if relationship_exists(session, canonical_id, rel.predicate, rel.object_id, rel.object_literal, rel.context_id):
            continue
        rel.subject_id = canonical_id
        reassigned.add(rel.id)
    for rel in session.scalars(select(Relationship).where(Relationship.object_id == old_id)).all():
        if relationship_exists(session, rel.subject_id, rel.predicate, canonical_id, rel.object_literal, rel.context_id):
            continue
        rel.object_id = canonical_id
        reassigned.add(rel.id)
    return reassigned

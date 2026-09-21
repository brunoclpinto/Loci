import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from loci.db.models import Context, ContextRelationship, Entity, Relationship
from loci.mcp_server.models import Citation, EntitySummary, RelationshipSummary


def get_context_by_name(session: Session, name: str) -> Context | None:
    return session.scalars(select(Context).where(Context.name == name)).first()


def expand_descendant_context_ids(session: Session, root_ids: list[uuid.UUID]) -> set[uuid.UUID]:
    seen: set[uuid.UUID] = set(root_ids)
    frontier = list(root_ids)
    while frontier:
        children = session.scalars(
            select(ContextRelationship.child_context_id).where(ContextRelationship.parent_context_id.in_(frontier))
        ).all()
        new = [c for c in children if c not in seen]
        seen.update(new)
        frontier = new
    return seen


def resolve_context_ids(
    session: Session, context_names: list[str] | None, include_descendants: bool
) -> list[uuid.UUID] | None:
    """Returns None if no filter should be applied (search everything)."""
    if not context_names:
        return None
    ids = []
    for name in context_names:
        ctx = get_context_by_name(session, name)
        if ctx is not None:
            ids.append(ctx.id)
    if include_descendants:
        return list(expand_descendant_context_ids(session, ids))
    return ids


def context_name(session: Session, context_id: uuid.UUID) -> str:
    ctx = session.get(Context, context_id)
    return ctx.name if ctx else str(context_id)


def to_citation(session: Session, source_adapter: str | None, source_ref: str | None, context_id: uuid.UUID) -> Citation:
    return Citation(source_adapter=source_adapter, source_ref=source_ref, context_name=context_name(session, context_id))


def to_entity_summary(session: Session, entity: Entity) -> EntitySummary:
    return EntitySummary(
        id=str(entity.id),
        type=entity.type,
        canonical_name=entity.canonical_name,
        aliases=list(entity.aliases),
        context_name=context_name(session, entity.context_id),
        scope=entity.scope,
        attributes=entity.attributes,
        citation=to_citation(session, entity.source_adapter, entity.source_ref, entity.context_id),
    )


def to_relationship_summary(session: Session, rel: Relationship) -> RelationshipSummary:
    subject = session.get(Entity, rel.subject_id)
    obj = session.get(Entity, rel.object_id) if rel.object_id else None
    return RelationshipSummary(
        id=str(rel.id),
        subject_id=str(rel.subject_id),
        subject_name=subject.canonical_name if subject else str(rel.subject_id),
        predicate=rel.predicate,
        object_id=str(rel.object_id) if rel.object_id else None,
        object_name=obj.canonical_name if obj else None,
        object_literal=rel.object_literal,
        context_name=context_name(session, rel.context_id),
        citation=to_citation(session, rel.source_adapter, rel.source_ref, rel.context_id),
    )

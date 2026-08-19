from sqlalchemy import func, select

from loci.config import LociSettings
from loci.db.engine import session_scope
from loci.db.models import Context, ContextRelationship, Entity, Relationship
from loci.mcp_server.models import ContextEntityView, ContextRelation, ContextSummary, EntitySummary, RelationshipSummary
from loci.mcp_server.repo import get_context_by_name, to_entity_summary, to_relationship_summary
from loci.normalize.resolver import EntityResolver


def _context_summary(session, ctx: Context) -> ContextSummary:
    parents = [
        ContextRelation(relation_type=cr.relation_type, context_name=session.get(Context, cr.parent_context_id).name)
        for cr in session.scalars(select(ContextRelationship).where(ContextRelationship.child_context_id == ctx.id))
    ]
    children = [
        ContextRelation(relation_type=cr.relation_type, context_name=session.get(Context, cr.child_context_id).name)
        for cr in session.scalars(select(ContextRelationship).where(ContextRelationship.parent_context_id == ctx.id))
    ]
    entity_count = session.scalar(select(func.count()).select_from(Entity).where(Entity.context_id == ctx.id)) or 0
    relationship_count = (
        session.scalar(select(func.count()).select_from(Relationship).where(Relationship.context_id == ctx.id)) or 0
    )
    return ContextSummary(
        id=str(ctx.id),
        name=ctx.name,
        description=ctx.description,
        parents=parents,
        children=children,
        entity_count=entity_count,
        relationship_count=relationship_count,
    )


def make_list_contexts(settings: LociSettings):
    def list_contexts(parent_context_name: str | None = None) -> list[ContextSummary]:
        """List contexts. If parent_context_name is given, list only its
        direct children; otherwise list every context."""
        with session_scope(settings) as session:
            if parent_context_name:
                parent = get_context_by_name(session, parent_context_name)
                if parent is None:
                    return []
                child_ids = session.scalars(
                    select(ContextRelationship.child_context_id).where(
                        ContextRelationship.parent_context_id == parent.id
                    )
                ).all()
                contexts = [session.get(Context, cid) for cid in child_ids]
            else:
                contexts = list(session.scalars(select(Context)))
            return [_context_summary(session, c) for c in contexts]

    return list_contexts


def make_get_context(settings: LociSettings):
    def get_context(context_name: str) -> ContextSummary | None:
        """Full detail for one context: description, parent/child context
        relationships, and entity/relationship counts."""
        with session_scope(settings) as session:
            ctx = get_context_by_name(session, context_name)
            if ctx is None:
                return None
            return _context_summary(session, ctx)

    return get_context


def make_compare_contexts(settings: LociSettings):
    resolver = EntityResolver(settings)

    def compare_contexts(entity_name: str, context_names: list[str]) -> list[ContextEntityView]:
        """For a given entity name, show what each named context asserts
        about it side by side — the entity as resolved in that context (or
        the nearest cross-context-shared match) plus the relationships
        scoped to that context. This is how contradictory or
        context-specific claims about the "same" entity stay visible
        instead of silently merging."""
        views: list[ContextEntityView] = []
        with session_scope(settings) as session:
            for name in context_names:
                ctx = get_context_by_name(session, name)
                if ctx is None:
                    views.append(ContextEntityView(context_name=name, entity=None, relationships=[]))
                    continue
                candidates = resolver.find_candidates(session, entity_name, context_id=ctx.id, limit=1)
                if not candidates:
                    views.append(ContextEntityView(context_name=name, entity=None, relationships=[]))
                    continue
                entity = candidates[0]
                rels = session.scalars(
                    select(Relationship).where(
                        Relationship.context_id == ctx.id,
                        (Relationship.subject_id == entity.id) | (Relationship.object_id == entity.id),
                    )
                )
                views.append(
                    ContextEntityView(
                        context_name=name,
                        entity=to_entity_summary(session, entity),
                        relationships=[to_relationship_summary(session, r) for r in rels],
                    )
                )
        return views

    return compare_contexts

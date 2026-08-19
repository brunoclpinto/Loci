import uuid

from sqlalchemy import select

from loci.config import LociSettings
from loci.db.engine import session_scope
from loci.db.models import Relationship
from loci.mcp_server.models import RelationshipSummary
from loci.mcp_server.repo import resolve_context_ids, to_relationship_summary


def make_get_relationships(settings: LociSettings):
    def get_relationships(
        entity_id: str,
        predicate: str | None = None,
        direction: str = "both",
        context_names: list[str] | None = None,
    ) -> list[RelationshipSummary]:
        """Relationships touching a given entity. `direction` is
        'outgoing' (entity is subject), 'incoming' (entity is object), or
        'both'. Optionally filter by predicate and/or by context."""
        eid = uuid.UUID(entity_id)
        with session_scope(settings) as session:
            context_ids = resolve_context_ids(session, context_names, include_descendants=False)

            if direction == "outgoing":
                stmt = select(Relationship).where(Relationship.subject_id == eid)
            elif direction == "incoming":
                stmt = select(Relationship).where(Relationship.object_id == eid)
            else:
                stmt = select(Relationship).where((Relationship.subject_id == eid) | (Relationship.object_id == eid))

            if predicate:
                stmt = stmt.where(Relationship.predicate == predicate)
            if context_ids is not None:
                stmt = stmt.where(Relationship.context_id.in_(context_ids))

            return [to_relationship_summary(session, r) for r in session.scalars(stmt)]

    return get_relationships

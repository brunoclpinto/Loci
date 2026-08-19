import uuid

from loci.config import LociSettings
from loci.db.engine import session_scope
from loci.db.models import Entity
from loci.mcp_server.models import EntitySummary
from loci.mcp_server.repo import get_context_by_name, to_entity_summary
from loci.normalize.resolver import EntityResolver


def make_get_entity(settings: LociSettings):
    def get_entity(entity_id: str) -> EntitySummary | None:
        """Fetch a single entity by id, with its citation. Use
        get_relationships separately for its relationships."""
        with session_scope(settings) as session:
            entity = session.get(Entity, uuid.UUID(entity_id))
            if entity is None:
                return None
            return to_entity_summary(session, entity)

    return get_entity


def make_resolve_entity(settings: LociSettings):
    resolver = EntityResolver(settings)

    def resolve_entity(name: str, context_name: str | None = None, type_hint: str | None = None) -> list[EntitySummary]:
        """Find candidate entities matching `name` by fuzzy canonical-name
        similarity, for disambiguation before referencing an entity. If
        context_name is given, prefers that context but still surfaces
        cross-context-shared entities from elsewhere."""
        with session_scope(settings) as session:
            context_id = None
            if context_name:
                ctx = get_context_by_name(session, context_name)
                context_id = ctx.id if ctx else None
            candidates = resolver.find_candidates(session, name, context_id=context_id, type_name=type_hint)
            return [to_entity_summary(session, e) for e in candidates]

    return resolve_entity

import uuid

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from loci.config import LociSettings
from loci.db.models import Entity


class EntityResolver:
    """Decides whether an incoming (type, canonical_name, context) refers to
    an entity already in the knowledge base. Both ingestion and any future
    query-time disambiguation MUST go through this — it's the enforcement
    point for the shared-normalization invariant on the structured side."""

    def __init__(self, settings: LociSettings):
        self.name_similarity_threshold = settings.resolver.name_similarity_threshold

    def find_candidate(self, session: Session, type_name: str, canonical_name: str, context_id: uuid.UUID) -> Entity | None:
        scope_filter = or_(Entity.context_id == context_id, Entity.scope == "cross_context")

        exact = session.scalars(
            select(Entity).where(
                Entity.type == type_name,
                scope_filter,
                func.lower(Entity.canonical_name) == canonical_name.lower(),
            )
        ).first()
        if exact is not None:
            return exact

        similarity = func.similarity(Entity.canonical_name, canonical_name)
        row = session.execute(
            select(Entity, similarity.label("sim"))
            .where(Entity.type == type_name, scope_filter, similarity >= self.name_similarity_threshold)
            .order_by(similarity.desc())
            .limit(1)
        ).first()
        return row[0] if row else None

    def find_candidates(
        self,
        session: Session,
        canonical_name: str,
        context_id: uuid.UUID | None = None,
        type_name: str | None = None,
        limit: int = 5,
    ) -> list[Entity]:
        """Ranked fuzzy-name candidates, for disambiguation callers (e.g. the
        MCP resolve_entity tool) rather than the create-or-match ingestion
        path. context_id may be None to search across all contexts."""
        similarity = func.similarity(Entity.canonical_name, canonical_name)
        stmt = select(Entity, similarity.label("sim")).where(similarity >= self.name_similarity_threshold)
        if type_name is not None:
            stmt = stmt.where(Entity.type == type_name)
        if context_id is not None:
            stmt = stmt.where(or_(Entity.context_id == context_id, Entity.scope == "cross_context"))
        stmt = stmt.order_by(similarity.desc()).limit(limit)
        return [row[0] for row in session.execute(stmt)]

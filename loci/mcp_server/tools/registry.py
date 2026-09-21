from sqlalchemy import select

from loci.config import LociSettings
from loci.db.engine import session_scope
from loci.db.models import EntityType, EntityTypeSchemaVersion, PredicateVocabulary
from loci.mcp_server.models import EntityTypeSchemaOut


def make_list_entity_types(settings: LociSettings):
    def list_entity_types() -> list[str]:
        """List the entity type names currently registered. Call
        get_entity_type_schema for the fields a given type expects before
        calling search_knowledge / get_entity with that type."""
        with session_scope(settings) as session:
            return list(session.scalars(select(EntityType.name).order_by(EntityType.name)))

    return list_entity_types


def make_get_entity_type_schema(settings: LociSettings):
    def get_entity_type_schema(type_name: str) -> EntityTypeSchemaOut | None:
        """The active JSON Schema for a given entity type's `attributes`."""
        with session_scope(settings) as session:
            et = session.get(EntityType, type_name)
            if et is None:
                return None
            version_row = session.get(
                EntityTypeSchemaVersion, {"entity_type": type_name, "version": et.current_schema_version}
            )
            if version_row is None:
                return None
            return EntityTypeSchemaOut(name=type_name, version=et.current_schema_version, json_schema=version_row.json_schema)

    return get_entity_type_schema


def make_list_predicates(settings: LociSettings):
    def list_predicates() -> list[str]:
        """List the predicate names currently registered."""
        with session_scope(settings) as session:
            return list(session.scalars(select(PredicateVocabulary.name).order_by(PredicateVocabulary.name)))

    return list_predicates

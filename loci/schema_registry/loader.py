from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from loci.db.models import EntityType, EntityTypeSchemaVersion, PredicateVocabulary


@dataclass(frozen=True)
class EntityTypeSchema:
    name: str
    version: int
    json_schema: dict


@dataclass(frozen=True)
class PredicateDefinition:
    name: str
    domain_entity_types: list[str]
    range_entity_types: list[str]


class SchemaRegistry:
    """Loads the live entity-type / predicate vocabulary from Postgres.
    This is the single source of truth both ingestion and the extraction
    prompt must read from — never hardcode types/predicates elsewhere."""

    def __init__(self, entity_types: dict[str, EntityTypeSchema], predicates: dict[str, PredicateDefinition]):
        self.entity_types = entity_types
        self.predicates = predicates

    def entity_schema(self, type_name: str) -> EntityTypeSchema:
        try:
            return self.entity_types[type_name]
        except KeyError:
            raise ValueError(f"Unknown entity type: {type_name!r}") from None

    def predicate(self, name: str) -> PredicateDefinition:
        try:
            return self.predicates[name]
        except KeyError:
            raise ValueError(f"Unknown predicate: {name!r}") from None

    @classmethod
    def load(cls, session: Session) -> "SchemaRegistry":
        entity_types: dict[str, EntityTypeSchema] = {}
        for et in session.scalars(select(EntityType)):
            version_row = session.get(
                EntityTypeSchemaVersion, {"entity_type": et.name, "version": et.current_schema_version}
            )
            if version_row is None:
                continue
            entity_types[et.name] = EntityTypeSchema(
                name=et.name, version=et.current_schema_version, json_schema=version_row.json_schema
            )

        predicates: dict[str, PredicateDefinition] = {}
        for p in session.scalars(select(PredicateVocabulary)):
            predicates[p.name] = PredicateDefinition(
                name=p.name, domain_entity_types=list(p.domain_entity_types), range_entity_types=list(p.range_entity_types)
            )

        return cls(entity_types=entity_types, predicates=predicates)

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from loci.db.models import EntityType, EntityTypeSchemaVersion, EnumVocabulary, PredicateVocabulary


@dataclass(frozen=True)
class EntityTypeSchema:
    name: str
    version: int
    json_schema: dict
    short_description: str | None = None
    long_description: str | None = None


@dataclass(frozen=True)
class PredicateDefinition:
    name: str
    domain_entity_types: list[str]
    range_entity_types: list[str]
    short_description: str | None = None
    long_description: str | None = None


@dataclass(frozen=True)
class EnumValueDefinition:
    domain: str
    value: str
    short_description: str | None = None
    long_description: str | None = None


class SchemaRegistry:
    """Loads the live entity-type / predicate / enum vocabulary from
    Postgres. This is the single source of truth ingestion and every
    extraction prompt must read from — never hardcode types/predicates/
    enum values elsewhere."""

    def __init__(
        self,
        entity_types: dict[str, EntityTypeSchema],
        predicates: dict[str, PredicateDefinition],
        enum_definitions: dict[str, dict[str, EnumValueDefinition]],
    ):
        self.entity_types = entity_types
        self.predicates = predicates
        self.enum_definitions = enum_definitions

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

    def enum_values(self, domain: str) -> list[EnumValueDefinition]:
        return list(self.enum_definitions.get(domain, {}).values())

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
                name=et.name,
                version=et.current_schema_version,
                json_schema=version_row.json_schema,
                short_description=version_row.short_description,
                long_description=version_row.long_description,
            )

        predicates: dict[str, PredicateDefinition] = {}
        for p in session.scalars(select(PredicateVocabulary)):
            predicates[p.name] = PredicateDefinition(
                name=p.name,
                domain_entity_types=list(p.domain_entity_types),
                range_entity_types=list(p.range_entity_types),
                short_description=p.short_description,
                long_description=p.long_description,
            )

        enum_definitions: dict[str, dict[str, EnumValueDefinition]] = {}
        for e in session.scalars(select(EnumVocabulary)):
            enum_definitions.setdefault(e.domain, {})[e.value] = EnumValueDefinition(
                domain=e.domain,
                value=e.value,
                short_description=e.short_description,
                long_description=e.long_description,
            )

        return cls(entity_types=entity_types, predicates=predicates, enum_definitions=enum_definitions)

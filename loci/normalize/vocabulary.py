from loci.schema_registry.loader import SchemaRegistry


class VocabularyError(ValueError):
    pass


def validate_predicate_usage(registry: SchemaRegistry, predicate: str, subject_type: str, object_type: str | None) -> None:
    """Predicates declare an allowed domain/range; an empty list means 'any
    type'. Enforced (not just advisory) — this is what keeps the shared
    vocabulary meaningful rather than decorative."""
    definition = registry.predicate(predicate)  # raises VocabularyError-compatible ValueError if unknown

    if definition.domain_entity_types and subject_type not in definition.domain_entity_types:
        raise VocabularyError(
            f"predicate {predicate!r} does not allow subject type {subject_type!r} "
            f"(allowed: {definition.domain_entity_types})"
        )
    if object_type is not None and definition.range_entity_types and object_type not in definition.range_entity_types:
        raise VocabularyError(
            f"predicate {predicate!r} does not allow object type {object_type!r} "
            f"(allowed: {definition.range_entity_types})"
        )

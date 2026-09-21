import jsonschema

from loci.schema_registry.loader import SchemaRegistry


class AttributeValidationError(ValueError):
    def __init__(self, entity_type: str, errors: list[str]):
        self.entity_type = entity_type
        self.errors = errors
        super().__init__(f"attributes for entity type {entity_type!r} failed validation: {'; '.join(errors)}")


def validate_attributes(registry: SchemaRegistry, type_name: str, attributes: dict) -> int:
    """Validate `attributes` against the active JSON Schema for `type_name`.
    Returns the schema version validated against (to stamp on the row)."""
    schema = registry.entity_schema(type_name)
    validator = jsonschema.Draft7Validator(schema.json_schema)
    errors = [f"{'.'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in validator.iter_errors(attributes)]
    if errors:
        raise AttributeValidationError(type_name, errors)
    return schema.version

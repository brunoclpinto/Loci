from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_COERCERS = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": lambda v: str(v).strip().lower() in {"true", "1", "yes"},
}


@dataclass
class AttributeMapping:
    column: str
    type: str = "string"

    def coerce(self, raw: Any) -> Any:
        if raw is None or raw == "":
            return None
        return _COERCERS[self.type](raw)


@dataclass
class RelationshipMapping:
    predicate: str
    target_type: str
    target_column: str
    target_scope: str = "context_local"


@dataclass
class MappingSpec:
    entity_type: str
    canonical_name_column: str
    attributes: dict[str, AttributeMapping] = field(default_factory=dict)
    relationships: list[RelationshipMapping] = field(default_factory=list)
    context: str | dict[str, str] | None = None
    id_column: str | None = None
    scope: str = "context_local"

    def context_name_for_row(self, row: dict[str, Any]) -> str:
        if isinstance(self.context, dict):
            return str(row[self.context["column"]])
        return self.context

    def natural_key_for_row(self, row: dict[str, Any], context_name: str) -> str:
        if self.id_column:
            return str(row[self.id_column])
        return f"{row[self.canonical_name_column]}::{context_name}"

    @classmethod
    def from_yaml(cls, path: Path) -> "MappingSpec":
        raw = yaml.safe_load(path.read_text())
        attributes = {
            key: AttributeMapping(column=val["column"], type=val.get("type", "string"))
            for key, val in (raw.get("attributes") or {}).items()
        }
        relationships = [
            RelationshipMapping(
                predicate=r["predicate"],
                target_type=r["target_type"],
                target_column=r["target_column"],
                target_scope=r.get("target_scope", "context_local"),
            )
            for r in raw.get("relationships") or []
        ]
        return cls(
            entity_type=raw["entity_type"],
            canonical_name_column=raw["canonical_name_column"],
            attributes=attributes,
            relationships=relationships,
            context=raw.get("context"),
            id_column=raw.get("id_column"),
            scope=raw.get("scope", "context_local"),
        )

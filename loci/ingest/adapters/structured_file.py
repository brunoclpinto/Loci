import csv
import json
from collections.abc import Iterable
from typing import Any

from loci.ingest.adapters.mapping.spec import MappingSpec
from loci.ingest.base import ExtractedEntity, ExtractedRelationship, ExtractionBatch, SourceAdapter, SourceDescriptor


class StructuredFileAdapter(SourceAdapter):
    """Deterministic column/field -> schema mapping for CSV/JSON sources.
    No LLM involved: the mapping spec tells us exactly what each field
    means, so this is pure translation, not inference."""

    name = "structured_file"

    def __init__(self, mapping_spec: MappingSpec):
        self.spec = mapping_spec

    def can_handle(self, source: SourceDescriptor) -> bool:
        if source.forced_adapter:
            return source.forced_adapter in {"csv", "json", "structured_file"}
        return source.path.suffix.lower() in {".csv", ".json"}

    def extract(self, source: SourceDescriptor) -> Iterable[ExtractionBatch]:
        rows = self._read_rows(source)

        batches_by_context: dict[str, ExtractionBatch] = {}
        for i, row in enumerate(rows):
            context_name = self.spec.context_name_for_row(row) if self.spec.context is not None else source.context_name
            batch = batches_by_context.setdefault(
                context_name, ExtractionBatch(context_name=context_name, source_ref=str(source.path))
            )

            primary_local_id = f"row{i}:{self.spec.natural_key_for_row(row, context_name)}"
            attributes = {
                key: mapping.coerce(row.get(mapping.column)) for key, mapping in self.spec.attributes.items()
            }
            attributes = {k: v for k, v in attributes.items() if v is not None}

            batch.entities.append(
                ExtractedEntity(
                    local_id=primary_local_id,
                    type=self.spec.entity_type,
                    canonical_name=str(row[self.spec.canonical_name_column]),
                    attributes=attributes,
                    scope=self.spec.scope,
                )
            )

            for rel in self.spec.relationships:
                target_value = row.get(rel.target_column)
                if not target_value:
                    continue
                target_local_id = f"row{i}:{rel.predicate}:{rel.target_column}"
                batch.entities.append(
                    ExtractedEntity(
                        local_id=target_local_id,
                        type=rel.target_type,
                        canonical_name=str(target_value),
                        scope=rel.target_scope,
                    )
                )
                batch.relationships.append(
                    ExtractedRelationship(
                        subject_local_id=primary_local_id, predicate=rel.predicate, object_local_id=target_local_id
                    )
                )

        yield from batches_by_context.values()

    def _read_rows(self, source: SourceDescriptor) -> list[dict[str, Any]]:
        suffix = source.path.suffix.lower()
        if suffix == ".csv":
            with source.path.open(newline="") as f:
                return list(csv.DictReader(f))
        if suffix == ".json":
            data = json.loads(source.path.read_text())
            if isinstance(data, dict):
                data = data.get("rows", data.get("items", [data]))
            return data
        raise ValueError(f"unsupported structured file extension: {suffix}")

from dataclasses import dataclass, field

from loci.ingest.entity_reference import build_name_lookup


@dataclass
class RegistryEntity:
    id: str
    type: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    scope: str = "context_local"
    identity_status: str = "named"


class EntityRegistry:
    """Document-scoped entity registry built by the two-pass redesign's
    entity-discovery walk (pass 1), then frozen and handed to the
    relationship-extraction walk (pass 2). Ids are permanent for the
    lifetime of one ingest run — unlike the old single-pass adapter's
    per-window local_ids, they never need to be reinterpreted across
    windows/units, which is what structurally prevents the stale-id
    relationship-drop bug Part 1 patched around."""

    def __init__(self) -> None:
        self._by_id: dict[str, RegistryEntity] = {}
        self._by_name: dict[str, RegistryEntity] = {}
        self._seq = 0

    def merge(self, discovered: list[dict]) -> list[RegistryEntity]:
        """Add each newly-discovered raw entity dict, deduping by exact
        canonical_name match (mirrors the old adapter's _remember_entities
        behavior). An entity already in the registry just gets any new
        aliases unioned in and is not re-yielded; only genuinely new
        entities are returned, for the caller to persist via
        ExtractionBatch."""
        newly_registered: list[RegistryEntity] = []
        for raw in discovered:
            existing = self._by_name.get(raw["canonical_name"])
            if existing is not None:
                for alias in raw.get("aliases") or []:
                    if alias not in existing.aliases:
                        existing.aliases.append(alias)
                continue
            self._seq += 1
            entity = RegistryEntity(
                id=f"e{self._seq}",
                type=raw["type"],
                canonical_name=raw["canonical_name"],
                aliases=list(raw.get("aliases") or []),
                attributes=raw.get("attributes") or {},
                scope=raw.get("scope") or "context_local",
                identity_status=raw.get("identity_status") or "named",
            )
            self._by_id[entity.id] = entity
            self._by_name[entity.canonical_name] = entity
            newly_registered.append(entity)
        return newly_registered

    def as_hint_list(self) -> list[dict]:
        """Same shape as the old single-pass adapter's known-entities hint
        (local_id/type/canonical_name/aliases) — consumed both as the LLM
        prompt hint and, via entity_reference.build_name_lookup, as the
        source for resolve_relationship_endpoint's known_by_id/known_by_name
        lookups."""
        return [
            {
                "local_id": e.id,
                "type": e.type,
                "canonical_name": e.canonical_name,
                "aliases": e.aliases,
            }
            for e in self._by_id.values()
        ]

    def name_lookup(self) -> dict[str, dict]:
        return build_name_lookup(self.as_hint_list())

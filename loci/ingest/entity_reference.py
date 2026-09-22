"""Shared relationship-endpoint resolution, used by every extraction pass
that needs to turn a model-supplied id/name string into either a current
window's own local_id or a reference to an already-established entity.

Written once here (Part 1's fix for the stale-local_id relationship-drop
bug) and reused unchanged by the two-pass redesign's relationship pass, so
there is exactly one resolution mechanism in the codebase, not two."""


def normalize_name(name: str) -> str:
    return " ".join(name.split()).casefold()


def build_name_lookup(known_entities: list[dict]) -> dict[str, dict]:
    """canonical_name/alias (normalized) -> known-entity dict. First match
    wins on collision, consistent with _remember_entities' first-seen-wins
    dedup by canonical_name."""
    lookup: dict[str, dict] = {}
    for entity in known_entities:
        for name in (entity["canonical_name"], *entity.get("aliases", [])):
            key = normalize_name(name)
            if key not in lookup:
                lookup[key] = entity
    return lookup


def resolve_relationship_endpoint(
    raw_id: str,
    local_ids: set[str],
    known_by_id: dict[str, dict],
    known_by_name: dict[str, dict],
) -> tuple[str | None, dict | None]:
    """Resolve one relationship endpoint (a subject_local_id or
    object_local_id string from a raw extraction response) in precedence
    order:

    1. The current window's own local_id namespace — a legitimately new
       local entity always wins.
    2. A known entity's stable id (its permanent "k<N>" id from
       _remember_entities) — the natural shortcut a model takes when it
       copies the id straight out of the known-entities hint instead of
       re-emitting that entity in this window's own `entities` array.
    3. A known entity's canonical_name/alias, exact match only (no fuzzy
       matching — a mis-attributed relationship is worse than a dropped
       one) — catches a model that writes a name-like string instead of an
       id.

    Returns (resolved_id, known_entity): `known_entity` is non-None only
    when resolution fell through to (2) or (3), signaling the caller must
    synthesize an ExtractedEntity for it (it wasn't re-emitted in this
    window's own entities). Returns (None, None) when nothing matches —
    caller drops the relationship exactly as before."""
    if raw_id in local_ids:
        return raw_id, None
    if raw_id in known_by_id:
        return raw_id, known_by_id[raw_id]
    matched = known_by_name.get(normalize_name(raw_id))
    if matched is not None:
        return matched["local_id"], matched
    return None, None

import json
from pathlib import Path

import httpx

from loci.config import OllamaSettings

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


_DISCOVERY_SYSTEM_PROMPT = (_PROMPTS_DIR / "discover_entities.md").read_text()
_RELATIONSHIP_SYSTEM_PROMPT = (_PROMPTS_DIR / "extract_relationships.md").read_text()


def _discovery_response_schema(entity_types: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "maxItems": 30,
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": entity_types},
                        "canonical_name": {"type": "string"},
                        "aliases": {"type": "array", "maxItems": 5, "items": {"type": "string"}},
                        "attributes": {"type": "object"},
                        "scope": {"type": "string", "enum": ["context_local", "cross_context"]},
                        "identity_status": {"type": "string", "enum": ["named", "unresolved"]},
                    },
                    "required": ["type", "canonical_name"],
                },
            },
        },
        "required": ["entities"],
    }


def discover_entities(
    settings: OllamaSettings,
    model: str,
    text: str,
    entity_types: list[str],
    known_entities: list[dict] | None = None,
) -> list[dict]:
    """Pass 1 of the two-pass extraction redesign: entity discovery only, no
    relationships. Deliberately doesn't ask the model for a local_id at all
    — the caller's EntityRegistry assigns a permanent id on merge, so
    there's nothing here for a later pass to go stale against (see
    loci/ingest/entity_registry.py)."""
    user_content = text
    if known_entities:
        known_block = json.dumps(known_entities, ensure_ascii=False)
        user_content = (
            f"Entities already established elsewhere in this document:\n{known_block}\n\n"
            f"If anything below refers to one of these, reuse its exact canonical_name rather than "
            f"creating a new entity.\n\n---\n\n{user_content}"
        )
    response = httpx.post(
        f"{settings.base_url}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": _DISCOVERY_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "format": _discovery_response_schema(entity_types),
            "options": {"temperature": 0, "repeat_penalty": 1.3, "num_predict": 16384, "num_ctx": 32768},
        },
        timeout=settings.request_timeout_s,
    )
    response.raise_for_status()
    content = response.json()["message"]["content"]
    return json.loads(content)["entities"]


def _relationship_response_schema(predicates: list[str], registry_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "relationships": {
                "type": "array",
                "maxItems": 30,
                "items": {
                    "type": "object",
                    "properties": {
                        "subject_id": {"type": "string", "enum": registry_ids},
                        "predicate": {"type": "string", "enum": predicates},
                        "object_id": {"type": "string", "enum": registry_ids},
                        "object_literal": {"type": "object"},
                    },
                    "required": ["subject_id", "predicate"],
                },
            },
        },
        "required": ["relationships"],
    }


def extract_relationships(
    settings: OllamaSettings,
    model: str,
    text: str,
    predicates: list[str],
    registry_hint: list[dict],
) -> list[dict]:
    """Pass 2 of the two-pass extraction redesign: relationships only, over
    a paragraph-grouped unit, referencing entities already discovered in
    pass 1. `subject_id`/`object_id` are JSON-schema enum-constrained to the
    live registry's ids — the strongest structural guarantee available that
    the model can't invent a new entity here, on top of (not instead of)
    resolve_relationship_endpoint's fallback matching downstream. Caller
    must not call this with an empty registry_hint (an empty enum is
    meaningless to the schema)."""
    registry_ids = [e["local_id"] for e in registry_hint]
    registry_block = json.dumps(registry_hint, ensure_ascii=False)
    user_content = (
        f"Entities already established in this document (only reference these by id; "
        f"never invent a new entity or id here):\n{registry_block}\n\n---\n\n{text}"
    )
    response = httpx.post(
        f"{settings.base_url}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": _RELATIONSHIP_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "format": _relationship_response_schema(predicates, registry_ids),
            "options": {"temperature": 0, "repeat_penalty": 1.3, "num_predict": 16384, "num_ctx": 32768},
        },
        timeout=settings.request_timeout_s,
    )
    response.raise_for_status()
    content = response.json()["message"]["content"]
    return json.loads(content)["relationships"]


_SEGMENT_CLASSIFICATION_SYSTEM_PROMPT = """Classify the given block of text as exactly one of:
- narrative: actual content — the substance of the document (a story, article, report, etc.).
- front_matter: information about the document/artifact itself that precedes its real content — a title page, table of contents, author byline, publisher/edition info.
- back_matter: information about the document/artifact itself that follows its real content — a colophon, license text, appendix, index.
Respond with only the classification."""


def classify_segment(settings: OllamaSettings, model: str, text: str) -> str:
    """A cheap classification call — deliberately separate from `extract()`
    so it can run on a fast model regardless of which model is doing
    extraction for a given run. Used by loci/ingest/structure.py as the
    fallback for leading/trailing blocks the structural heuristics aren't
    confident about."""
    response = httpx.post(
        f"{settings.base_url}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": _SEGMENT_CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "format": {
                "type": "object",
                "properties": {"segment_type": {"type": "string", "enum": ["narrative", "front_matter", "back_matter"]}},
                "required": ["segment_type"],
            },
            "options": {"temperature": 0, "num_predict": 2048, "num_ctx": 8192},
        },
        timeout=settings.request_timeout_s,
    )
    response.raise_for_status()
    content = response.json()["message"]["content"]
    return json.loads(content)["segment_type"]


_COREFERENCE_SYSTEM_PROMPT = """You are resolving coreferences in a single document's extracted knowledge
graph. You'll be given a list of "unresolved" entities — real individuals whose identity wasn't known when
first mentioned (e.g. "the mysterious man", "the cabman") — and a list of "named" entities established
elsewhere in the same document. For each unresolved entity, decide whether it is later revealed to be one of
the named entities, using the descriptive attributes/context given for each. Only match when the evidence
clearly points to the same individual — an unresolved entity with no clear match should be left unmatched;
it's a legitimate distinct minor character, not a failure to resolve.
Respond with a JSON array: one object per unresolved entity, {"unresolved_id": "...", "matched_id": "..."} or
{"unresolved_id": "...", "matched_id": null} when there's no clear match."""


def match_coreferences(
    settings: OllamaSettings, model: str, unresolved: list[dict], named: list[dict]
) -> dict[str, str | None]:
    """Whole-document resolution pass for identities only revealed later
    than their first mention (see loci/ingest/coreference.py). Returns
    {unresolved_id: matched_named_id_or_None}."""
    if not unresolved:
        return {}
    user_content = (
        f"Unresolved entities:\n{json.dumps(unresolved, ensure_ascii=False)}\n\n"
        f"Named entities:\n{json.dumps(named, ensure_ascii=False)}"
    )
    response = httpx.post(
        f"{settings.base_url}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": _COREFERENCE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "format": {
                "type": "object",
                "properties": {
                    "matches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "unresolved_id": {"type": "string"},
                                "matched_id": {"type": ["string", "null"]},
                            },
                            "required": ["unresolved_id", "matched_id"],
                        },
                    }
                },
                "required": ["matches"],
            },
            "options": {"temperature": 0, "num_predict": 16384, "num_ctx": 32768},
        },
        timeout=settings.request_timeout_s,
    )
    response.raise_for_status()
    content = response.json()["message"]["content"]
    parsed = json.loads(content)
    return {m["unresolved_id"]: m["matched_id"] for m in parsed["matches"]}


_TYPE_CONFLICT_SYSTEM_PROMPT = """You are cleaning up a knowledge graph where the same real-world entity was
sometimes extracted with a different type in different passes (e.g. "Jefferson Hope" once as a Person, once
as a Location — a mistake, not two different things). You'll be given groups of entities whose names are
similar but whose types differ. For each group, decide: are any of these entities actually the same real
thing, just mistyped? If so, pick the ONE entity with the correct type and the most complete/accurate name as
canonical, and list the others as duplicates to merge into it. If the entities are legitimately different
things that happen to share a similar name, leave the group alone (canonical_id null, empty duplicate_ids).
Respond with a JSON array: one object per group, {"group_id": "...", "canonical_id": "..." or null,
"duplicate_ids": ["..."]}."""


def match_type_conflicts(settings: OllamaSettings, model: str, groups: list[list[dict]]) -> dict[str, list[str]]:
    """Whole-document pass for the same entity having been tagged with
    different types in different extraction windows (see
    loci/ingest/type_consolidation.py) — the resolver only matches within a
    type, so these never merge on their own. Returns
    {canonical_id: [duplicate_id, ...]}."""
    if not groups:
        return {}
    payload = [{"group_id": str(i), "entities": g} for i, g in enumerate(groups)]
    response = httpx.post(
        f"{settings.base_url}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": _TYPE_CONFLICT_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "format": {
                "type": "object",
                "properties": {
                    "decisions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "group_id": {"type": "string"},
                                "canonical_id": {"type": ["string", "null"]},
                                "duplicate_ids": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["group_id", "canonical_id", "duplicate_ids"],
                        },
                    }
                },
                "required": ["decisions"],
            },
            "options": {"temperature": 0, "num_predict": 16384, "num_ctx": 32768},
        },
        timeout=settings.request_timeout_s,
    )
    response.raise_for_status()
    parsed = json.loads(response.json()["message"]["content"])

    result: dict[str, list[str]] = {}
    for d in parsed["decisions"]:
        if d.get("canonical_id"):
            result.setdefault(d["canonical_id"], []).extend(d.get("duplicate_ids") or [])
    return result

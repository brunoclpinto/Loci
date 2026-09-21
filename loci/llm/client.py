import json

import httpx

from loci.config import OllamaSettings


def _extraction_response_schema(entity_types: list[str], predicates: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "maxItems": 30,
                "items": {
                    "type": "object",
                    "properties": {
                        "local_id": {"type": "string"},
                        "type": {"type": "string", "enum": entity_types},
                        "canonical_name": {"type": "string"},
                        "aliases": {"type": "array", "maxItems": 5, "items": {"type": "string"}},
                        "attributes": {"type": "object"},
                        "scope": {"type": "string", "enum": ["context_local", "cross_context"]},
                        "identity_status": {"type": "string", "enum": ["named", "unresolved"]},
                    },
                    "required": ["local_id", "type", "canonical_name"],
                },
            },
            "relationships": {
                "type": "array",
                "maxItems": 30,
                "items": {
                    "type": "object",
                    "properties": {
                        "subject_local_id": {"type": "string"},
                        "predicate": {"type": "string", "enum": predicates},
                        "object_local_id": {"type": "string"},
                        "object_literal": {"type": "object"},
                    },
                    "required": ["subject_local_id", "predicate"],
                },
            },
        },
        "required": ["entities", "relationships"],
    }


class ExtractionLLMClient:
    """Wraps Ollama /api/chat for entity/relationship extraction. The
    response `format` is a JSON Schema whose `type`/`predicate` fields are
    constrained (via `enum`) to the live vocabulary passed in — this is what
    keeps the model from inventing types/predicates outside the shared
    schema, on top of the downstream hard validation."""

    def __init__(self, settings: OllamaSettings, system_prompt: str):
        self.settings = settings
        self.system_prompt = system_prompt

    def extract(
        self,
        text: str,
        entity_types: list[str],
        predicates: list[str],
        retry_note: str | None = None,
        known_entities: list[dict] | None = None,
    ) -> dict:
        user_content = text
        if known_entities:
            # Cross-window continuity: if this window's content refers to
            # one of these — by name, alias, title, or a clear pronoun/role
            # reference — reuse its exact canonical_name rather than
            # inventing a new entity. This is what lets "Dr. Watson" in
            # window 1 and "the narrator" in window 5 resolve to one entity
            # instead of two.
            known_block = json.dumps(known_entities, ensure_ascii=False)
            user_content = (
                f"Entities already established elsewhere in this document:\n{known_block}\n\n"
                f"If anything below refers to one of these, reuse its exact canonical_name.\n\n"
                f"---\n\n{user_content}"
            )
        if retry_note is not None:
            user_content = f"{user_content}\n\n---\nYour previous attempt was invalid: {retry_note}\nTry again."
        response = httpx.post(
            f"{self.settings.base_url}/api/chat",
            json={
                "model": self.settings.chat_model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "format": _extraction_response_schema(entity_types, predicates),
                # num_predict is a runaway-generation safety net, not a token
                # budget — this is local inference, so there's no per-token
                # cost to economize. It must be generous enough that
                # reasoning models (which spend real tokens on <think>
                # before ever reaching `content`) can actually finish: 1024
                # was cutting deepseek-r1:32b off mid-thought with empty
                # content every time. 16384 leaves headroom while still
                # bounding truly pathological loops (the failure mode this
                # guards against, first hit with qwen2.5:7b-instruct).
                "options": {"temperature": 0, "repeat_penalty": 1.3, "num_predict": 16384, "num_ctx": 32768},
            },
            timeout=self.settings.request_timeout_s,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        return json.loads(content)


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

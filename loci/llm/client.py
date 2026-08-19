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
        self, text: str, entity_types: list[str], predicates: list[str], retry_note: str | None = None
    ) -> dict:
        user_content = text if retry_note is None else f"{text}\n\n---\nYour previous attempt was invalid: {retry_note}\nTry again."
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

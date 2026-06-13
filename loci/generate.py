"""Grounded generation: chat model wrapper, streaming, citations, REPL history."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from loci.retrieve import ChunkHit, FactHit

_SYSTEM_PROMPT = """\
You are a knowledge extraction assistant. Your only job is to extract answers from context.

RULES:
1. The context is the authoritative source. Only answer if the context directly supports the answer. Do not use your training data or general knowledge.
2. ALWAYS cite every claim with the exact tag from the context (e.g. [F1], [C3]) immediately after the claim.
3. Only use tags that actually appear in the provided context. Never invent or guess a tag.
4. If the context does not contain the answer to the question, say exactly: "The context does not contain this information." """

_NO_CONTEXT_NOTE = ""  # prompt rule 4 already covers the no-context case

_PREFILL = "Based on the provided context: "

_TAG_RE = re.compile(r"\[([FC]\d+)\]")


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------

def build_messages(
    question: str,
    context_text: str,
    history: list[dict],
) -> list[dict]:
    """Build the messages list for a chat completion call.

    history contains alternating user/assistant dicts from previous turns.
    Fresh context is injected into the current user message each turn.
    """
    sys_content = _SYSTEM_PROMPT
    if not context_text.strip():
        sys_content += _NO_CONTEXT_NOTE
        user_content = f"Question: {question}"
        messages: list[dict] = [{"role": "system", "content": sys_content}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_content})
    else:
        user_content = (
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n"
            f"Answer with inline citations ([C#] or [F#]):"
        )
        messages = [{"role": "system", "content": sys_content}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": _PREFILL})
    return messages


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def stream_response(
    llm: Any,
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
) -> Generator[str, None, None]:
    """Yield token deltas from the chat model (streaming mode).

    When the last message is an assistant prefill, yields it first so callers
    receive the complete response including the forced prefix.
    """
    prefill = ""
    if messages and messages[-1]["role"] == "assistant":
        prefill = messages[-1]["content"]
        if prefill:
            yield prefill
    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        choices = chunk.get("choices", [])
        if choices:
            text = choices[0].get("delta", {}).get("content", "")
            if text:
                yield text


def generate_response(
    llm: Any,
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    """Non-streaming generation. Returns complete response text.

    When the last message is an assistant prefill, prepends it to the output.
    """
    prefill = ""
    if messages and messages[-1]["role"] == "assistant":
        prefill = messages[-1]["content"]
    result = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
    return prefill + result["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

def extract_cited_tags(text: str) -> list[str]:
    """Return unique [F…]/[C…] tags from response text, in order of appearance."""
    seen: set[str] = set()
    result: list[str] = []
    for m in _TAG_RE.finditer(text):
        tag = f"[{m.group(1)}]"
        if tag not in seen:
            seen.add(tag)
            result.append(tag)
    return result


def build_sources_footer(
    cited_tags: list[str],
    fact_hits: list["FactHit"],
    chunk_hits: list["ChunkHit"],
) -> str:
    """Map cited tags to human-readable source lines for the footer."""
    if not cited_tags:
        return ""

    fact_map = {fh.tag: fh for fh in fact_hits}
    chunk_map = {ch.tag: ch for ch in chunk_hits}

    lines = ["Sources:"]
    for tag in cited_tags:
        if tag in fact_map:
            fh = fact_map[tag]
            src = fh.source_info or "unknown source"
            lines.append(f"  {tag} → {src} (fact: {fh.subject_name} — {fh.predicate})")
        elif tag in chunk_map:
            ch = chunk_map[tag]
            src = ch.source_info or "unknown source"
            lines.append(f"  {tag} → {src}")

    return "\n".join(lines) if len(lines) > 1 else ""


def strip_invalid_citations(
    text: str,
    fact_hits: list["FactHit"],
    chunk_hits: list["ChunkHit"],
) -> str:
    """Remove any [F…]/[C…] tags from text that don't appear in the context."""
    valid = {fh.tag for fh in fact_hits} | {ch.tag for ch in chunk_hits}
    def _replace(m: re.Match) -> str:
        tag = f"[{m.group(1)}]"
        return tag if tag in valid else ""
    return _TAG_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------

_REFUSAL_PHRASES = (
    "not in my knowledge base",
    "no context available",
    "none available",
    "no relevant information",
    "context does not contain",
    "context does not include",
    "does not contain this information",
    "not mentioned in",
    "not provided in",
    "cannot find",
    "not found in",
)

def is_refusal(text: str) -> bool:
    """Return True if the response is a knowledge-base refusal."""
    lower = text.strip().lower()
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)


# ---------------------------------------------------------------------------
# Rolling history management
# ---------------------------------------------------------------------------

def trim_history(
    history: list[dict],
    max_turns: int = 3,
    token_budget: int = 1024,
) -> list[dict]:
    """Keep at most max_turns user/assistant pairs; drop oldest if over token budget.

    token_budget is a rough limit on combined chars / 4 (chars-per-token estimate).
    """
    pairs: list[tuple[dict, dict]] = []
    for i in range(0, len(history) - 1, 2):
        if i + 1 < len(history):
            pairs.append((history[i], history[i + 1]))

    pairs = pairs[-max_turns:]

    while pairs:
        total_chars = sum(len(u["content"]) + len(a["content"]) for u, a in pairs)
        if total_chars <= token_budget * 4:
            break
        pairs = pairs[1:]

    return [msg for pair in pairs for msg in pair]

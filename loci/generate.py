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

# When set, prepended to every user message to suppress Qwen3 chain-of-thought.
# Enable via: LOCI_MODELS__NO_THINK=1 (checked at build_messages time).
_NO_THINK_PREFIX = "/no_think\n"

_TAG_RE = re.compile(r"\[([FCP]\d+)\]")


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
    import os
    no_think = os.environ.get("LOCI_MODELS__NO_THINK", "").strip() in ("1", "true", "yes")
    prefix = _NO_THINK_PREFIX if no_think else ""

    sys_content = _SYSTEM_PROMPT
    if not context_text.strip():
        sys_content += _NO_CONTEXT_NOTE
        user_content = f"{prefix}Question: {question}"
        messages: list[dict] = [{"role": "system", "content": sys_content}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_content})
    else:
        user_content = (
            f"{prefix}Context:\n{context_text}\n\n"
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
    prop_hits: "list | None" = None,
) -> str:
    """Remove any [F…]/[C…]/[P…] tags from text that don't appear in the context."""
    valid = {fh.tag for fh in fact_hits} | {ch.tag for ch in chunk_hits}
    if prop_hits:
        for i, _ in enumerate(prop_hits, 1):
            valid.add(f"[P{i}]")
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
    "not stated in the source",  # proposition-path abstention phrase
)

def is_refusal(text: str) -> bool:
    """Return True if the response is a knowledge-base refusal."""
    lower = text.strip().lower()
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)


# ---------------------------------------------------------------------------
# Proposition-path generation (design-v1)
# ---------------------------------------------------------------------------

_PROP_ABSTAIN = "Not stated in the source."

_PROP_WITH_FACTS = (
    "Facts:\n{facts}\n\n"
    "Question: {question}\n"
    "Using ONLY the facts above, answer as briefly as possible and end your answer "
    "with the [P#] tag (example: 'Stamford [P1].'). "
    f"If none of the facts answer the question, reply exactly: {_PROP_ABSTAIN}"
)

_PROP_NO_FACT = (
    "Question: {question}\n"
    f"There is no relevant fact available. Reply exactly: {_PROP_ABSTAIN}"
)

# Token budget (chars) for the combined facts block — keeps prompt lean.
_PROP_FACTS_CHAR_BUDGET = 800


def build_proposition_messages(
    question: str,
    prop_hits: "Any",  # list[PropositionHit] | None  (None treated as empty)
) -> list[dict]:
    """Build the proposition-path prompt.

    Accepts a ranked list of PropositionHit objects.  The top hits are formatted
    as [P1] ... [Pk] lines so the model can cite the source.  Returns an abstain
    prompt when the list is empty or None.
    """
    hits = prop_hits or []
    # Flatten single hit passed by legacy callers (PropositionHit, not a list).
    if hasattr(hits, "statement"):
        hits = [hits]

    if not hits:
        prompt = _PROP_NO_FACT.format(question=question)
        return [{"role": "user", "content": prompt}]

    # Build [P1]...[Pk] lines, respecting the char budget.
    lines: list[str] = []
    total = 0
    for i, h in enumerate(hits, 1):
        line = f"[P{i}] {h.statement}"
        if total + len(line) > _PROP_FACTS_CHAR_BUDGET:
            break
        lines.append(line)
        total += len(line)

    facts_block = "\n".join(lines)
    prompt = _PROP_WITH_FACTS.format(facts=facts_block, question=question)
    return [{"role": "user", "content": prompt}]


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

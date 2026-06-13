"""LLM-assisted fact extraction: catches passives, copulas, and possessives."""
from __future__ import annotations

import json
import re
from typing import Any

_SYSTEM_PROMPT = """\
You are a precise fact extractor for a knowledge graph.
Extract factual statements from the text passage below.

Focus on constructions a syntactic parser misses:
1. Passive voice      — "The letter was signed by Holmes"       → Holmes, sign, letter
2. Copulas (X is Y)   — "Watson is a doctor"                   → Watson, be, doctor
3. Possessives        — "Holmes's pipe" / "Holmes has a pipe"  → Holmes, possess, pipe
4. Location/residence — "they took rooms at No. 221B"          → they, reside, No. 221B, Baker Street
5. Occupation/role    — "he worked as a cab driver"            → he, work_as, cab driver
6. Meaning/equivalence — "'RACHE' means revenge"               → RACHE, mean, revenge
7. Naming/calling     — "called the Baker Street Irregulars"   → group, call, Baker Street Irregulars

Rules:
- "predicate" must be a lowercase lemmatised English verb (or compound like "work_as", "reside_at").
- "subject" and "object" must be noun phrases exactly as they appear in the text.
- Only extract statements clearly supported by the text — never infer.
- Reply with ONLY a valid JSON array, no prose, no markdown fences.
  Each element: {"subject": "...", "predicate": "...", "object": "...", "qualifiers": {}, "negated": false}
- If nothing to extract, reply: []"""

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def build_extraction_messages(chunk_text: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Text:\n{chunk_text}"},
    ]


def parse_llm_facts(response_text: str) -> list[dict]:
    """Parse LLM JSON response into validated raw-fact dicts."""
    text = response_text.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    result = []
    for item in data:
        if not isinstance(item, dict):
            continue
        subj = str(item.get("subject", "")).strip()
        pred = str(item.get("predicate", "")).strip().lower()
        if not subj or not pred:
            continue
        obj_raw = item.get("object", "")
        obj = str(obj_raw).strip() if obj_raw else None
        quals = item.get("qualifiers")
        result.append({
            "subject": subj,
            "predicate": pred,
            "object": obj or None,
            "qualifiers": quals if isinstance(quals, dict) else None,
            "negated": bool(item.get("negated", False)),
        })
    return result


def enhance_chunk(
    conn: Any,
    chunk_id: int,
    chunk_text: str,
    llm: Any,
    *,
    cfg: Any,
    embedder: Any = None,
) -> int:
    """Run LLM extraction on one chunk; mark it extracted. Returns new fact count."""
    from loci.generate import generate_response
    from loci.resolve import resolve_entity
    from loci.store import insert_fact, mark_chunk_extracted

    messages = build_extraction_messages(chunk_text)
    try:
        response = generate_response(llm, messages, max_tokens=512, temperature=0.0)
    except Exception:
        mark_chunk_extracted(conn, chunk_id)
        return 0

    raw_facts = parse_llm_facts(response)
    n_inserted = 0
    for rf in raw_facts:
        subj_id = resolve_entity(
            conn, rf["subject"],
            embedder=embedder,
            entity_sim_threshold=cfg.retrieval.entity_sim_threshold,
        )
        if subj_id is None:
            continue

        obj_id = None
        obj_text = rf["object"]
        if obj_text and obj_text[0].isupper():
            resolved = resolve_entity(
                conn, obj_text,
                embedder=embedder,
                entity_sim_threshold=cfg.retrieval.entity_sim_threshold,
            )
            if resolved is not None:
                obj_id = resolved
                obj_text = None

        fact_id = insert_fact(
            conn,
            chunk_id=chunk_id,
            sentence=chunk_text[:300],
            subject_id=subj_id,
            predicate=rf["predicate"],
            object_id=obj_id,
            object_text=obj_text,
            qualifiers=rf["qualifiers"],
            negated=rf["negated"],
            confidence=0.7,
        )
        if fact_id is not None:
            n_inserted += 1

    mark_chunk_extracted(conn, chunk_id)
    return n_inserted


def run_enhance(
    conn: Any,
    *,
    llm: Any,
    cfg: Any,
    embedder: Any = None,
    limit: int | None = None,
) -> dict:
    """Process all unextracted chunks. Returns {chunks_processed, facts_added}."""
    from loci.store import get_unextracted_chunks

    chunks = get_unextracted_chunks(conn, limit=limit)
    n_facts = 0
    for row in chunks:
        n_facts += enhance_chunk(
            conn, row["id"], row["text"], llm,
            cfg=cfg, embedder=embedder,
        )
    return {"chunks_processed": len(chunks), "facts_added": n_facts}

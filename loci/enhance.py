"""LLM-assisted fact extraction: catches passives, copulas, and possessives."""
from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# P1 relation taxonomy — predicates that match question vocabulary
# ---------------------------------------------------------------------------

_TAXONOMY: frozenset[str] = frozenset([
    # identity / role
    "profession",       # "I'm a consulting detective" → Holmes, profession, consulting detective
    "occupation",       # "he worked as a cab driver" → Hope, occupation, cab driver
    "role",             # "our landlady" → Mrs Hudson, role, landlady
    "identity",         # "I am Sherlock Holmes" → speaker, identity, Sherlock Holmes
    "alias_of",         # X is also known as Y
    "title",            # "Mr Drebber" / "Inspector Gregson"
    # knowledge / meaning
    "means",            # "RACHE means revenge" → RACHE, means, revenge
    # location
    "located_at",       # X is located at Y
    "resides_at",       # X lives/resides at Y
    # relationships
    "relationship_to",  # "her adopted father John Ferrier" → Lucy, relationship_to, John Ferrier
    "affiliation",      # X belongs to / is a member of Y
    "leader_of",        # "Brigham Young, the Mormon leader" → Young, leader_of, Mormon expedition
    # actions with named outcomes
    "introduce",        # "Stamford introduced Watson to Holmes"
    "murder",           # "Hope murdered Drebber"
    "cause_of",         # "his wound caused his financial difficulties"
    "named_after",      # X was named after Y
    # kept from original prompt (broad coverage)
    "work_as",          # alias for occupation
    "reside_at",        # alias for resides_at
    "mean",             # alias for means
    "call",             # X is called Y
    "be",               # X is/was Y (broad copula)
    "possess",          # X has/owns Y
    "sign",             # passive voice: letter was signed by X
    "employ",           # X is employed by Y
    "found",            # X was founded by Y
])

_SYSTEM_PROMPT = """\
You are a precise fact extractor for a knowledge graph.
Extract factual relationships from the text passage below.

Focus on constructions a syntactic parser misses:
1. Copulas / role     — "Watson is a doctor" / "our landlady Mrs Hudson"
2. Passive voice      — "The letter was signed by Holmes" → Holmes, sign, letter
3. Occupation/title   — "he worked as a cab driver" → Hope, occupation, cab driver
4. Meaning/translation — "'RACHE' means revenge" → RACHE, means, revenge
5. Named intro        — "Stamford introduced Watson to Holmes" → Stamford, introduce, Watson
6. Leadership         — "Brigham Young, leader of the train" → Young, leader_of, Mormon wagon train

PREDICATE TAXONOMY — use ONLY these values (exact strings, lowercase):
  profession, occupation, role, identity, alias_of, title,
  means, mean,
  located_at, resides_at, reside_at,
  relationship_to, affiliation, leader_of,
  introduce, murder, cause_of, named_after,
  work_as, call, be, possess, sign, employ, found

COREFERENCE: If the text contains a description like "our landlady", "the leader", or "her adopted father",
AND the passage also names the person (e.g. "Mrs Hudson", "Brigham Young", "John Ferrier"),
use the CANONICAL NAME as subject or object — not the description.

OUTPUT: A JSON array. Each element MUST have exactly these fields:
  {"subject": "...", "predicate": "...", "object": "...", "sentence": "...", "qualifiers": {}, "negated": false}
- "sentence": the exact sentence from the text that most directly supports this fact.
- "predicate": must be from the taxonomy above — reject anything outside it.
- "subject" and "object": proper names when possible (resolve coreferences).
- Never infer beyond what the text says. If nothing fits the taxonomy: []"""

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def build_extraction_messages(
    chunk_text: str,
    known_entities: list[str] | None = None,
) -> list[dict]:
    user_content = ""
    if known_entities:
        user_content = (
            "Known entities in this corpus (prefer these canonical names):\n"
            + ", ".join(known_entities[:25])
            + "\n\n"
        )
    user_content += f"Text:\n{chunk_text}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
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
        # Taxonomy gate: reject predicates outside the allowed set
        if pred not in _TAXONOMY:
            continue
        obj_raw = item.get("object", "")
        obj = str(obj_raw).strip() if obj_raw else None
        quals = item.get("qualifiers")
        sentence = str(item.get("sentence", "")).strip() or None
        result.append({
            "subject": subj,
            "predicate": pred,
            "object": obj or None,
            "qualifiers": quals if isinstance(quals, dict) else None,
            "negated": bool(item.get("negated", False)),
            "sentence": sentence,
        })
    return result


def _get_entity_names(conn: Any, limit: int = 25) -> list[str]:
    """Return the top entity canonical names (by fact count) for coref context."""
    rows = conn.execute(
        """SELECT e.canonical_name, COUNT(f.id) AS n
           FROM entities e LEFT JOIN facts f ON f.subject_id=e.id
           GROUP BY e.id ORDER BY n DESC LIMIT ?""",
        [limit * 2],  # over-fetch to filter noisy names
    ).fetchall()
    names = []
    for r in rows:
        name = r[0] if isinstance(r, (list, tuple)) else r["canonical_name"]
        # Keep only plausible proper nouns: has uppercase, short enough, no line breaks
        if (name and len(name) <= 40 and name != name.lower()
                and "\n" not in name and "," not in name):
            names.append(name)
            if len(names) >= limit:
                break
    return names


def enhance_chunk(
    conn: Any,
    chunk_id: int,
    chunk_text: str,
    llm: Any,
    *,
    cfg: Any,
    embedder: Any = None,
    known_entities: list[str] | None = None,
) -> int:
    """Run LLM extraction on one chunk; mark it extracted. Returns new fact count."""
    from loci.generate import generate_response
    from loci.resolve import resolve_entity
    from loci.store import insert_fact, mark_chunk_extracted

    messages = build_extraction_messages(chunk_text, known_entities=known_entities)
    try:
        response = generate_response(llm, messages, max_tokens=768, temperature=0.0)
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

        # Use LLM-provided sentence if available, else fall back to chunk prefix
        sentence = rf.get("sentence") or chunk_text[:300]

        fact_id = insert_fact(
            conn,
            chunk_id=chunk_id,
            sentence=sentence,
            subject_id=subj_id,
            predicate=rf["predicate"],
            object_id=obj_id,
            object_text=obj_text,
            qualifiers=rf["qualifiers"],
            negated=rf["negated"],
            confidence=0.7,
            source="llm",
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
    force_all: bool = False,
) -> dict:
    """Process chunks and extract facts with the taxonomy-constrained LLM prompt.

    force_all=True resets extracted_v=0 for all chunks before running, so a
    fresh P1 pass runs over the entire corpus even if enhance was run before.
    """
    from loci.store import get_unextracted_chunks

    if force_all:
        conn.execute("UPDATE chunks SET extracted_v=0")
        conn.commit()

    chunks = get_unextracted_chunks(conn, limit=limit)
    known_entities = _get_entity_names(conn)

    n_facts = 0
    for row in chunks:
        cid = row["id"] if hasattr(row, "__getitem__") else row[0]
        ctext = row["text"] if hasattr(row, "__getitem__") else row[1]
        n_facts += enhance_chunk(
            conn, cid, ctext, llm,
            cfg=cfg, embedder=embedder,
            known_entities=known_entities,
        )

    try:
        from loci.store import rebuild_fact_fts, rebuild_fact_vec, _FACT_VEC_VERSION
        n_fts = rebuild_fact_fts(conn)
        conn.execute("INSERT OR REPLACE INTO db_meta(key,value) VALUES ('fact_fts_v','1')")
        conn.commit()
        if embedder is not None:
            rebuild_fact_vec(conn, embedder)
            conn.execute(
                "INSERT OR REPLACE INTO db_meta(key,value) VALUES ('fact_vec_v',?)",
                [_FACT_VEC_VERSION],
            )
            conn.commit()
    except Exception as _err:
        import warnings
        warnings.warn(f"fts/vec rebuild failed after enhance: {_err}")

    return {"chunks_processed": len(chunks), "facts_added": n_facts}

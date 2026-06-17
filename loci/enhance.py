"""LLM-assisted fact extraction: catches passives, copulas, and possessives."""
from __future__ import annotations

import json
import re
from typing import Any

# db_meta keys for idempotency guards
_P2_ENTITY_META_KEY = "p2_entity_v"
_P2_IMPLIED_META_KEY = "p2_implied_v"
_CLOSURE_META_KEY = "closure_v"
_P2_ENTITY_VERSION = "1"
_P2_IMPLIED_VERSION = "1"
_CLOSURE_VERSION = "1"

# Predicates that signal a contradicting occupation/role (used by prune pass)
_ROLE_PREDICATES = frozenset(["title", "profession", "role", "occupation", "work_as"])

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

# ---------------------------------------------------------------------------
# P2 Pass 1 — entity-centric cross-chunk extraction
# ---------------------------------------------------------------------------

_ENTITY_SYSTEM_PROMPT = """\
You are a precise fact extractor for a knowledge graph.
The passages below all mention the entity [{entity}]. Extract every factual relationship about [{entity}].

Focus on:
1. Occupation/profession/role — what do they do or what are they called?
2. Residence — where do they live or where are they found?
3. Relationships — who are they related to, introduced by, affiliated with?
4. Aliases — what other names or descriptions refer to them?
5. Actions with named outcomes — murders, introductions, causes.

COREFERENCE: Descriptions like "our landlady", "the cabman", "the old pioneer" that clearly refer to [{entity}]
should produce a fact with [{entity}] as subject using the canonical name, not the description.

PREDICATE TAXONOMY — use ONLY these values (exact strings, lowercase):
  profession, occupation, role, identity, alias_of, title,
  means, mean,
  located_at, resides_at, reside_at,
  relationship_to, affiliation, leader_of,
  introduce, murder, cause_of, named_after,
  work_as, call, be, possess, sign, employ, found

OUTPUT: A JSON array. Each element MUST have exactly these fields:
  {{"subject": "...", "predicate": "...", "object": "...", "sentence": "...", "qualifiers": {{}}, "negated": false}}
- "sentence": the exact sentence from the text that most directly supports this fact.
- "subject": always use the canonical name [{entity}], not a pronoun or description.
- Never infer beyond what the text says. If nothing fits the taxonomy: []"""

# ---------------------------------------------------------------------------
# P2 Pass 2 — implication / archaic-vocab extraction
# ---------------------------------------------------------------------------

_IMPLIED_SYSTEM_PROMPT = """\
You are a precise fact extractor for a knowledge graph.
Extract facts that are IMPLIED but not stated directly in the text passage below.

Focus on:
1. Occupations described by action — "he drove a cab" → occupation=cab driver; "worked as a constable" → occupation=constable
2. Foreign-word translations — "RACHE is the German word for revenge" → RACHE, means, revenge
3. Archaic or indirect terms decoded — "jarvey" is Victorian slang for cab driver
4. Aliases or disguises — "the old man was in fact Holmes" → alias_of
5. Roles implied by description — "our landlady Mrs Hudson" → Mrs Hudson, role, landlady

PREDICATE TAXONOMY — use ONLY these values (exact strings, lowercase):
  profession, occupation, role, identity, alias_of, title,
  means, mean,
  located_at, resides_at, reside_at,
  relationship_to, affiliation, leader_of,
  introduce, murder, cause_of, named_after,
  work_as, call, be, possess, sign, employ, found

OUTPUT: A JSON array. Each element MUST have exactly these fields:
  {{"subject": "...", "predicate": "...", "object": "...", "sentence": "...", "qualifiers": {{}}, "negated": false}}
- "subject" and "object": use canonical proper names when you can identify them.
- Only output high-confidence implications (0.8+). If nothing fits: []"""

# ---------------------------------------------------------------------------
# Proposition minting pass
# ---------------------------------------------------------------------------

_PROP_MINT_META_KEY = "prop_mint_v"
_PROP_MINT_VERSION = "2"

_PROP_TAXONOMY: frozenset[str] = frozenset([
    "introduce",      # A introduced B to C
    "reside_at",      # X lives/resides at Y
    "work_as",        # X works as / is a Y
    "murder",         # X murdered Y
    "possess",        # X has/owns Y
    "relationship_to", # X is the [role] of Y
    "leader_of",      # X leads/commands Y
    "call",           # X is called Y / referred to as Y
    "be",             # X is/was Y (broad copula)
    "means",          # X means Y (translation)
    "located_at",     # place X is at location Y
    "travel_to",      # X travelled/went to Y
    "find",           # X found/discovered Y
    "use",            # X used Y
    "married_to",     # X married Y
    "killed_by",      # X was killed by Y
    "employed_by",    # X is employed by Y
])

_PROP_SYSTEM_PROMPT = """\
You are a structured proposition extractor for a knowledge graph.
Extract meaningful propositions from the text passage below.

PREDICATE TAXONOMY with EXAMPLES — use ONLY these exact strings (lowercase):

  introduce     "Stamford introduced Watson to Holmes" → agent=Stamford, themes=[Watson, Holmes]
  work_as       "Holmes is a consulting detective" → agent=Holmes, themes=[consulting detective]
  reside_at     "They lived at 221B Baker Street" → agent=Holmes, themes=[], location=221B Baker Street
  murder        "Hope killed Drebber" → agent=Hope, themes=[Drebber]
  killed_by     "Drebber was killed by Hope" → agent=Drebber, themes=[Jefferson Hope]
  married_to    "Lucy was forced to marry Drebber" → agent=Lucy Ferrier, themes=[Drebber]
  relationship_to "John Ferrier was Lucy's adopted father" → agent=John Ferrier, themes=[Lucy Ferrier]
  leader_of     "Brigham Young led the Mormon settlers" → agent=Brigham Young, themes=[Mormon settlers]
  means         "'RACHE' is German for revenge" → agent=RACHE, themes=[revenge]
  call          "Holmes called Lecoq a bungler" → agent=Holmes, themes=[Lecoq, bungler]
  located_at    "The body was found at Lauriston Gardens" → agent=body, location=Lauriston Gardens
  travel_to     "Hope journeyed to Salt Lake City" → agent=Jefferson Hope, themes=[], location=Salt Lake City
  possess       "Holmes had a gold watch" → agent=Holmes, themes=[gold watch]
  find          "Lestrade found the body" → agent=Lestrade, themes=[body]
  use           "Holmes used a magnifying glass" → agent=Holmes, themes=[magnifying glass]
  employed_by   "Gregson works for Scotland Yard" → agent=Tobias Gregson, themes=[Scotland Yard]
  be            "Holmes was a student of chemistry" → agent=Holmes, themes=[student of chemistry]

COREFERENCE: When a description ("our landlady", "the leader", "her adopted father") co-occurs with
a proper name in the same passage, use the CANONICAL NAME as agent/theme.

STATEMENT: Write a self-contained NL sentence. Use entity names, not pronouns.
EVIDENCE: Copy the shortest verbatim span from the text that supports the proposition.

OUTPUT: A JSON array. Each element MUST have exactly these fields:
  {"predicate": "...", "agent": "...", "themes": ["..."], "location": null, "statement": "...", "evidence": "..."}
- "agent": the subject (canonical name — person, place, or word as appropriate for the predicate)
- "themes": list of objects/recipients (canonical names), empty list if none
- "location": location string if relevant, else null
- "predicate": must be from the taxonomy — reject anything outside it
- Skip propositions where agent is a pronoun (I, he, she, it, we, they) or a generic description
- Never infer beyond what the text says. If nothing fits: []"""


def _build_prop_messages(
    chunk_text: str,
    known_entities: list[str] | None = None,
) -> list[dict]:
    user = ""
    if known_entities:
        user = (
            "Known entities in this corpus (prefer these canonical names):\n"
            + ", ".join(known_entities[:25])
            + "\n\n"
        )
    user += f"Text:\n{chunk_text}"
    return [
        {"role": "system", "content": _PROP_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_llm_propositions(response_text: str) -> list[dict]:
    """Parse LLM JSON response into validated proposition dicts."""
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
        pred = str(item.get("predicate", "")).strip().lower()
        if pred not in _PROP_TAXONOMY:
            continue
        agent = str(item.get("agent", "")).strip()
        if not agent:
            continue
        themes_raw = item.get("themes", [])
        if isinstance(themes_raw, str):
            themes_raw = [themes_raw]
        themes = [str(t).strip() for t in themes_raw if str(t).strip()]
        location = str(item.get("location") or "").strip() or None
        statement = str(item.get("statement", "")).strip()
        evidence = str(item.get("evidence") or "").strip() or None
        if not statement:
            continue
        result.append({
            "predicate": pred,
            "agent": agent,
            "themes": themes,
            "location": location,
            "statement": statement,
            "evidence": evidence,
        })
    return result


def mint_propositions_chunk(
    conn: Any,
    chunk_id: int,
    chunk_text: str,
    llm: Any,
    *,
    cfg: Any,
    embedder: Any = None,
    known_entities: list[str] | None = None,
) -> int:
    """LLM-based proposition extraction for one chunk. Returns new proposition count."""
    from loci.generate import generate_response
    from loci.store import (
        ensure_prop_entity, insert_proposition,
        insert_proposition_entity, upsert_vec_proposition,
    )

    messages = _build_prop_messages(chunk_text, known_entities=known_entities)
    try:
        response = generate_response(llm, messages, max_tokens=1024, temperature=0.0)
    except Exception:
        return 0

    raw_props = parse_llm_propositions(response)
    n = 0
    for rp in raw_props[:8]:  # cap per chunk to avoid flooding
        agent_str = rp["agent"]
        themes = rp["themes"]
        location_str = rp.get("location")
        predicate = rp["predicate"]
        statement = rp["statement"]
        evidence = rp.get("evidence")

        agent_eid = ensure_prop_entity(conn, canonical=agent_str, kind="PERSON", aliases=[agent_str])
        theme_eids = [
            ensure_prop_entity(conn, canonical=t, kind="PERSON", aliases=[t])
            for t in themes
        ]
        loc_eid = (
            ensure_prop_entity(conn, canonical=location_str, kind="LOCATION", aliases=[location_str])
            if location_str else None
        )

        roles: dict = {"agent": agent_eid}
        if theme_eids:
            roles["theme"] = theme_eids if len(theme_eids) > 1 else theme_eids[0]
        if loc_eid is not None:
            roles["location"] = loc_eid

        prop_id = insert_proposition(
            conn,
            chunk_id=chunk_id,
            predicate=predicate,
            roles=roles,
            statement=statement,
            confidence=0.8,
            evidence=evidence,
        )
        if prop_id is None:
            continue  # duplicate

        insert_proposition_entity(conn, prop_id=prop_id, prop_entity_id=agent_eid, role="agent")
        for tid in theme_eids:
            insert_proposition_entity(conn, prop_id=prop_id, prop_entity_id=tid, role="theme")
        if loc_eid is not None:
            insert_proposition_entity(conn, prop_id=prop_id, prop_entity_id=loc_eid, role="location")

        if embedder is not None:
            try:
                emb = embedder.encode(statement)
                upsert_vec_proposition(conn, prop_id=prop_id, embedding=emb.tolist())
            except Exception:
                pass

        n += 1
    return n


def run_proposition_mint(
    conn: Any,
    *,
    llm: Any,
    cfg: Any,
    embedder: Any = None,
    limit: int | None = None,
    force: bool = False,
) -> dict:
    """Run LLM-based proposition minting across all chunks.

    Idempotency-guarded via db_meta['prop_mint_v']; use force=True to re-run.
    """
    row = conn.execute(
        "SELECT value FROM db_meta WHERE key=?", [_PROP_MINT_META_KEY]
    ).fetchone()
    if row and not force:
        return {"skipped": True, "propositions_added": 0, "chunks_processed": 0}

    rows = conn.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
    if limit is not None:
        rows = rows[:limit]

    known_entities = _get_entity_names(conn)
    n_props = 0
    for row in rows:
        cid = row[0] if isinstance(row, (list, tuple)) else row["id"]
        ctext = row[1] if isinstance(row, (list, tuple)) else row["text"]
        n_props += mint_propositions_chunk(
            conn, cid, ctext, llm,
            cfg=cfg, embedder=embedder,
            known_entities=known_entities,
        )

    # Clean up junk prop_entities and import rich aliases from the fact system
    from loci.store import sync_prop_entity_aliases_from_facts
    sync_stats = sync_prop_entity_aliases_from_facts(conn)

    conn.execute(
        "INSERT OR REPLACE INTO db_meta(key,value) VALUES (?,?)",
        [_PROP_MINT_META_KEY, _PROP_MINT_VERSION],
    )
    conn.commit()
    return {
        "chunks_processed": len(rows),
        "propositions_added": n_props,
        "prop_entities_removed": sync_stats["prop_entities_removed"],
        "aliases_added": sync_stats["aliases_added"],
        "skipped": False,
    }


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


def build_entity_messages(
    entity_name: str,
    chunks_text: str,
    known_entities: list[str] | None = None,
) -> list[dict]:
    system = _ENTITY_SYSTEM_PROMPT.replace("{entity}", entity_name)
    user = ""
    if known_entities:
        user = (
            "Known entities in this corpus (prefer these canonical names):\n"
            + ", ".join(known_entities[:25])
            + "\n\n"
        )
    user += f"Text passages about {entity_name}:\n{chunks_text}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_implied_messages(
    chunk_text: str,
    known_entities: list[str] | None = None,
) -> list[dict]:
    user = ""
    if known_entities:
        user = (
            "Known entities in this corpus (prefer these canonical names):\n"
            + ", ".join(known_entities[:25])
            + "\n\n"
        )
    user += f"Text:\n{chunk_text}"
    return [
        {"role": "system", "content": _IMPLIED_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _get_proper_entities(conn: Any) -> list[dict]:
    """Return clean proper-noun entities suitable for entity-centric extraction."""
    rows = conn.execute(
        "SELECT id, canonical_name FROM entities ORDER BY canonical_name"
    ).fetchall()
    result = []
    for r in rows:
        eid = r[0] if isinstance(r, (list, tuple)) else r["id"]
        name = r[1] if isinstance(r, (list, tuple)) else r["canonical_name"]
        if not name:
            continue
        # Filter: must have uppercase, reasonable length, no garbage chars
        if (
            len(name) > 2
            and len(name) <= 40
            and name != name.lower()
            and "\n" not in name
            and "|" not in name
            and "(" not in name
            and not name.startswith("_")
        ):
            result.append({"id": eid, "name": name})
    return result


def _insert_extracted_facts(
    conn: Any,
    raw_facts: list[dict],
    chunk_id: int,
    embedder: Any,
    cfg: Any,
    confidence: float,
) -> int:
    """Resolve and insert a list of raw fact dicts; returns count inserted."""
    from loci.generate import generate_response  # noqa: F401 (unused but consistent import pattern)
    from loci.resolve import resolve_entity
    from loci.store import insert_fact

    n = 0
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

        sentence = rf.get("sentence") or ""
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
            confidence=confidence,
            source="llm",
        )
        if fact_id is not None:
            n += 1
    return n


def run_entity_pass(
    conn: Any,
    *,
    llm: Any,
    cfg: Any,
    embedder: Any = None,
) -> dict:
    """P2 Pass 1: entity-centric cross-chunk fact extraction.

    For each proper-noun entity, gathers all chunks mentioning it via FTS,
    concatenates them as a single context window, and asks the LLM to extract
    all facts about that entity. Resolves cross-chunk coreferences naturally.
    """
    from loci.generate import generate_response
    from loci.store import fts_search_chunks

    # Idempotency guard
    done = conn.execute(
        "SELECT value FROM db_meta WHERE key=?", [_P2_ENTITY_META_KEY]
    ).fetchone()
    if done and done[0] == _P2_ENTITY_VERSION:
        return {"entities_processed": 0, "facts_added": 0, "skipped": True}

    entities = _get_proper_entities(conn)
    known_entities = _get_entity_names(conn)

    # Use the first chunk id as a stable anchor for dedup UNIQUE key.
    # We pass chunk_id=0 since entity-centric facts span many chunks.
    # chunk_id=0 is reserved (no chunk has id=0 in SQLite autoincrement).
    n_facts = 0
    n_entities = 0

    for ent in entities:
        eid = ent["id"]
        name = ent["name"]

        # FTS search for entity name across all chunks (up to 8)
        try:
            hits = fts_search_chunks(conn, query=f'"{name}"', k=8)
        except Exception:
            # FTS may reject some names with special chars
            try:
                hits = fts_search_chunks(conn, query=name, k=8)
            except Exception:
                continue

        if not hits:
            continue

        # Fetch chunk texts
        chunk_ids = [h["chunk_id"] for h in hits]
        placeholders = ",".join("?" * len(chunk_ids))
        rows = conn.execute(
            f"SELECT id, text FROM chunks WHERE id IN ({placeholders}) ORDER BY ordinal",
            chunk_ids,
        ).fetchall()

        if not rows:
            continue

        # Use first chunk id as anchor for UNIQUE dedup
        first_chunk_id = rows[0][0] if isinstance(rows[0], (list, tuple)) else rows[0]["id"]

        # Concatenate chunk texts with separator
        sep = "\n\n---\n\n"
        combined = sep.join(
            (r[1] if isinstance(r, (list, tuple)) else r["text"]) for r in rows
        )
        # Cap context to avoid token overflow (~4000 chars ≈ 1000 tokens)
        if len(combined) > 4000:
            combined = combined[:4000]

        messages = build_entity_messages(name, combined, known_entities=known_entities)
        try:
            response = generate_response(llm, messages, max_tokens=512, temperature=0.0)
        except Exception:
            continue

        raw_facts = parse_llm_facts(response)
        n_inserted = _insert_extracted_facts(
            conn, raw_facts, first_chunk_id, embedder, cfg, confidence=0.75
        )
        if n_inserted > 0:
            n_facts += n_inserted
        n_entities += 1

    conn.execute(
        "INSERT OR REPLACE INTO db_meta(key,value) VALUES (?,?)",
        [_P2_ENTITY_META_KEY, _P2_ENTITY_VERSION],
    )
    conn.commit()
    return {"entities_processed": n_entities, "facts_added": n_facts, "skipped": False}


def run_implied_pass(
    conn: Any,
    *,
    llm: Any,
    cfg: Any,
    embedder: Any = None,
) -> dict:
    """P2 Pass 2: implication/archaic-vocab fact extraction.

    Runs a secondary prompt over every chunk targeting implied facts:
    occupations described by action, foreign-word translations, archaic terms,
    and roles implied by description.
    """
    from loci.generate import generate_response

    # Idempotency guard
    done = conn.execute(
        "SELECT value FROM db_meta WHERE key=?", [_P2_IMPLIED_META_KEY]
    ).fetchone()
    if done and done[0] == _P2_IMPLIED_VERSION:
        return {"chunks_processed": 0, "facts_added": 0, "skipped": True}

    rows = conn.execute("SELECT id, text FROM chunks ORDER BY ordinal").fetchall()
    known_entities = _get_entity_names(conn)

    n_facts = 0
    for row in rows:
        cid = row[0] if isinstance(row, (list, tuple)) else row["id"]
        ctext = row[1] if isinstance(row, (list, tuple)) else row["text"]

        messages = build_implied_messages(ctext, known_entities=known_entities)
        try:
            response = generate_response(llm, messages, max_tokens=512, temperature=0.0)
        except Exception:
            continue

        raw_facts = parse_llm_facts(response)
        n_inserted = _insert_extracted_facts(
            conn, raw_facts, cid, embedder, cfg, confidence=0.65
        )
        n_facts += n_inserted

    conn.execute(
        "INSERT OR REPLACE INTO db_meta(key,value) VALUES (?,?)",
        [_P2_IMPLIED_META_KEY, _P2_IMPLIED_VERSION],
    )
    conn.commit()
    return {"chunks_processed": len(rows), "facts_added": n_facts, "skipped": False}


def run_prune_pass(
    conn: Any,
    *,
    dry_run: bool = False,
) -> dict:
    """Remove likely misattributed llm/closure facts.

    Detects entities that have a contradicting role/title/profession fact AND
    also have facts with object_text matching occupation bridging terms (e.g.
    'jarvey'). For those entities the bridging-term facts are almost certainly
    caused by the entity appearing in the same passage as the true subject —
    cross-entity contamination from the P2 entity/implied passes.

    After pruning, also deletes closure facts derived from the removed llm
    facts, then resets closure_v so the closure pass can re-run cleanly.

    dry_run=True: report counts without deleting anything.
    Returns: {pruned_llm, pruned_closure, dry_run}
    """
    # Vocabulary terms that act as "occupation bridges" — if an entity has
    # these as object_text, and the entity also has a DIFFERENT role that
    # contradicts it, the bridge fact is likely misattributed.
    _BRIDGE_TERMS: frozenset[str] = frozenset(["jarvey"])

    # Object texts that clearly signal the entity is NOT a jarvey/cab-driver.
    # Any entity with these in a role/title/profession fact is safe to prune
    # its bridge-term facts from.
    _NONCAB_SIGNALS: frozenset[str] = frozenset([
        "inspector", "detective", "constable", "sergeant", "physician",
        "consulting detective", "scotland yard", "policeman", "officer",
        "doctor",
    ])

    # Step 1: find entities with a contradicting role AND a bridge-term fact
    role_rows = conn.execute(
        """SELECT f.subject_id, LOWER(COALESCE(f.object_text, '')) AS obj
           FROM facts f
           WHERE f.predicate IN ('title','profession','role','occupation','work_as')
             AND f.source IN ('llm','svo','coref')
             AND NOT f.negated"""
    ).fetchall()

    # Map: entity_id → set of role/title objects
    entity_roles: dict[int, set[str]] = {}
    for r in role_rows:
        eid = r[0] if isinstance(r, (list, tuple)) else r["subject_id"]
        obj = r[1] if isinstance(r, (list, tuple)) else r["obj"]
        entity_roles.setdefault(eid, set()).add(obj)

    # Identify entities that have at least one noncab signal in their roles
    noncab_entities: set[int] = {
        eid for eid, roles in entity_roles.items()
        if any(any(sig in role for sig in _NONCAB_SIGNALS) for role in roles)
    }

    # Find bridge-term llm facts for those entities
    bridge_rows = conn.execute(
        """SELECT f.id
           FROM facts f
           WHERE f.source IN ('llm', 'coref')
             AND LOWER(COALESCE(f.object_text,'')) IN ({})
        """.format(",".join(f"'{t}'" for t in _BRIDGE_TERMS))
    ).fetchall()

    bad_llm_ids = [
        (r[0] if isinstance(r, (list, tuple)) else r["id"])
        for r in bridge_rows
    ]

    # Filter to only entities identified as noncab
    bad_llm_ids_filtered: list[int] = []
    for fid in bad_llm_ids:
        row = conn.execute("SELECT subject_id FROM facts WHERE id=?", [fid]).fetchone()
        if row is None:
            continue
        sid = row[0] if isinstance(row, (list, tuple)) else row["subject_id"]
        if sid in noncab_entities:
            bad_llm_ids_filtered.append(fid)

    # Step 2: find closure facts for the same subjects with derived bridge synonyms
    # (e.g. subject|work_as|cab driver where subject is a noncab entity)
    closure_rows = conn.execute(
        "SELECT id, subject_id FROM facts WHERE source='closure'"
    ).fetchall()
    bad_closure_ids = [
        (r[0] if isinstance(r, (list, tuple)) else r["id"])
        for r in closure_rows
        if (r[1] if isinstance(r, (list, tuple)) else r["subject_id"]) in noncab_entities
    ]

    if not dry_run:
        if bad_llm_ids_filtered:
            conn.execute(
                "DELETE FROM facts WHERE id IN ({})".format(
                    ",".join("?" * len(bad_llm_ids_filtered))
                ),
                bad_llm_ids_filtered,
            )
        if bad_closure_ids:
            conn.execute(
                "DELETE FROM facts WHERE id IN ({})".format(
                    ",".join("?" * len(bad_closure_ids))
                ),
                bad_closure_ids,
            )
        # Reset closure_v so closure pass can re-run with clean data
        conn.execute("DELETE FROM db_meta WHERE key=?", [_CLOSURE_META_KEY])
        conn.commit()

    return {
        "pruned_llm": len(bad_llm_ids_filtered),
        "pruned_closure": len(bad_closure_ids),
        "dry_run": dry_run,
    }


def run_closure_pass(
    conn: Any,
    *,
    cfg: Any,
    embedder: Any = None,
) -> dict:
    """Graph closure: for every X--[pred]-->Y + Y--[means]-->Z, mint X--[pred]-->Z.

    Materialises vocabulary bridges at ingest time so FTS can find them with a
    single slot. E.g.: Hope|work_as|jarvey + jarvey|means|cab driver
    → Hope|work_as|cab driver (source='closure').
    No LLM calls — pure graph traversal over existing facts.
    """
    from loci.store import insert_fact, rebuild_fact_fts_llm, _FACT_FTS_LLM_VERSION

    done = conn.execute(
        "SELECT value FROM db_meta WHERE key=?", [_CLOSURE_META_KEY]
    ).fetchone()
    if done and done[0] == _CLOSURE_VERSION:
        return {"facts_added": 0, "chains_found": 0, "skipped": True}

    # Build means map: entity_id → [synonyms] and surface text → [synonyms]
    means_rows = conn.execute(
        """SELECT f.subject_id, e.canonical_name, f.object_text
           FROM facts f
           JOIN entities e ON f.subject_id = e.id
           WHERE f.predicate IN ('means', 'mean')
             AND f.object_text IS NOT NULL
             AND NOT f.negated"""
    ).fetchall()

    means_by_entity_id: dict[int, list[str]] = {}
    means_by_surface: dict[str, list[str]] = {}
    for r in means_rows:
        eid = r[0] if isinstance(r, (list, tuple)) else r["subject_id"]
        name = (r[1] if isinstance(r, (list, tuple)) else r["canonical_name"]).lower()
        syn = r[2] if isinstance(r, (list, tuple)) else r["object_text"]
        means_by_entity_id.setdefault(eid, []).append(syn)
        means_by_surface.setdefault(name, []).append(syn)

    if not means_by_entity_id and not means_by_surface:
        conn.execute(
            "INSERT OR REPLACE INTO db_meta(key,value) VALUES (?,?)",
            [_CLOSURE_META_KEY, _CLOSURE_VERSION],
        )
        conn.commit()
        return {"facts_added": 0, "chains_found": 0, "skipped": False}

    # Scan all non-means facts and look for object → means chain
    fact_rows = conn.execute(
        """SELECT f.subject_id, f.predicate, f.object_id, f.object_text,
                  f.sentence, f.chunk_id, f.confidence,
                  e.canonical_name AS subj
           FROM facts f
           JOIN entities e ON f.subject_id = e.id
           WHERE f.predicate NOT IN ('means', 'mean') AND NOT f.negated"""
    ).fetchall()

    n_facts = 0
    n_chains = 0
    for r in fact_rows:
        if isinstance(r, (list, tuple)):
            subj_id, pred, obj_id, obj_text, sent, chunk_id, conf, subj = r
        else:
            subj_id = r["subject_id"]
            pred = r["predicate"]
            obj_id = r["object_id"]
            obj_text = r["object_text"]
            sent = r["sentence"]
            chunk_id = r["chunk_id"]
            conf = r["confidence"]
            subj = r["subj"]

        synonyms: list[str] = []
        via: str = ""

        if obj_id is not None and obj_id in means_by_entity_id:
            synonyms = means_by_entity_id[obj_id]
            via = str(obj_id)  # entity id; sentence will clarify
        elif obj_text:
            surface = obj_text.lower().strip()
            if surface in means_by_surface:
                synonyms = means_by_surface[surface]
                via = obj_text

        if not synonyms:
            continue

        n_chains += 1
        for syn in synonyms:
            if syn == obj_text:
                continue  # already the same text
            # Pre-check: skip if this closure triple already exists (any chunk)
            if conn.execute(
                "SELECT 1 FROM facts WHERE subject_id=? AND predicate=? AND object_text=?",
                [subj_id, pred, syn],
            ).fetchone():
                continue
            closure_sentence = f"Derived: {subj} {pred} {syn} (via {via or obj_text})"
            fid = insert_fact(
                conn,
                chunk_id=chunk_id,  # inherit source fact's chunk (valid FK)
                sentence=closure_sentence,
                subject_id=subj_id,
                predicate=pred,
                object_text=syn,
                confidence=round((conf or 1.0) * 0.9, 3),
                source="closure",
            )
            if fid is not None:
                n_facts += 1

    # Rebuild llm-only FTS so closure facts are immediately searchable
    rebuild_fact_fts_llm(conn)
    conn.execute(
        "INSERT OR REPLACE INTO db_meta(key,value) VALUES ('fact_fts_llm_v',?)",
        [_FACT_FTS_LLM_VERSION],
    )
    conn.execute(
        "INSERT OR REPLACE INTO db_meta(key,value) VALUES (?,?)",
        [_CLOSURE_META_KEY, _CLOSURE_VERSION],
    )
    conn.commit()
    return {"facts_added": n_facts, "chains_found": n_chains, "skipped": False}


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
        from loci.store import (
            rebuild_fact_fts, rebuild_fact_fts_llm,
            rebuild_fact_vec, _FACT_VEC_VERSION, _FACT_FTS_LLM_VERSION,
        )
        rebuild_fact_fts(conn)
        conn.execute("INSERT OR REPLACE INTO db_meta(key,value) VALUES ('fact_fts_v','1')")
        rebuild_fact_fts_llm(conn)
        conn.execute(
            "INSERT OR REPLACE INTO db_meta(key,value) VALUES ('fact_fts_llm_v',?)",
            [_FACT_FTS_LLM_VERSION],
        )
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

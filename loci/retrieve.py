"""Hybrid retrieval: question parse → fact SQL + vec + FTS → RRF fusion → context."""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loci.config import Config

_WH_WORDS = frozenset(["what", "who", "where", "when", "which", "how", "whom", "whose"])
_AUX_LEMMAS = frozenset(["be", "do", "have", "will", "would", "could", "should",
                          "may", "might", "shall", "can", "must"])

# For proposition FTS we use a narrower stopword set — character names are kept
# so FTS can discriminate between Holmes/Watson/Hope propositions.
# Domain-specific high-frequency terms are added dynamically from db_meta at
# query time via load_corpus_stopwords() — see _get_corpus_stops().
_PROP_FTS_STOPWORDS = frozenset([
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "must", "of", "in", "on",
    "at", "by", "to", "for", "with", "from", "and", "or", "but", "not",
    "no", "it", "its", "this", "that", "their", "there", "they", "them",
    "he", "she", "his", "her", "we", "our", "you", "your", "i", "my",
    "what", "who", "where", "when", "which", "how", "whom", "whose",
])

# Maps question NOUNS → proposition predicates for "What is X's [noun]?" questions
# where spaCy finds no main verb (e.g. "What is Holmes's profession?").
_NOUN_TO_PRED: dict[str, list[str]] = {
    "profession":   ["work_as"],
    "occupation":   ["work_as"],
    "job":          ["work_as"],
    "career":       ["work_as"],
    "role":         ["work_as"],
    "address":      ["reside_at", "located_at"],
    "home":         ["reside_at"],
    "residence":    ["reside_at"],
    "location":     ["located_at", "reside_at"],
    "name":         ["named_after"],
    "alias":        ["call", "named_after"],
    "meaning":      ["means"],
    "leader":       ["leader_of"],
    "killer":       ["murder", "kill", "killed_by"],
    "murderer":     ["murder", "kill", "killed_by"],
    "victim":       ["murder", "killed_by"],
    "relationship": ["relationship_to"],
    "weapon":       ["use"],
}

# Nouns that only augment predicates_to_try when the question has NO main verb.
# When a main verb IS present, these nouns describe an attribute of an entity
# (e.g. "Which X is a murderer?") rather than the question's predicate topic.
_VERB_CONDITIONAL_NOUNS = frozenset(["killer", "murderer", "victim", "weapon"])

# Maps question verb lemmas → proposition predicates that semantically match.
# Enables "What is Holmes's profession?" to find "work_as" propositions, etc.
_VERB_TO_PRED: dict[str, list[str]] = {
    "profession":   ["work_as"],
    "occupation":   ["work_as"],
    "job":          ["work_as"],
    "work":         ["work_as", "employed_by"],
    "employ":       ["employed_by", "work_as"],
    "live":         ["reside_at", "located_at"],
    "reside":       ["reside_at"],
    "stay":         ["reside_at"],
    "locate":       ["located_at"],
    "meet":         ["introduce"],
    "introduce":    ["introduce"],
    "kill":         ["murder", "killed_by"],
    "die":          ["killed_by", "murder"],
    "murder":       ["murder", "kill", "killed_by"],
    "possess":      ["possess"],
    "own":          ["possess"],
    "call":         ["call"],
    "name":         ["call", "named_after"],
    "mean":         ["means"],
    "travel":       ["travel_to"],
    "go":           ["travel_to"],
    "find":         ["find"],
    "marry":        ["married_to"],
    "lead":         ["leader_of"],
    "command":      ["leader_of"],
    "use":          ["use"],
}
_CHARS_PER_TOKEN = 4
_NONWORD = re.compile(r"[^\w\s]")

# Universal stopwords — language-level, not domain-specific.
# Domain-specific high-frequency terms are computed from the corpus during ingest
# and loaded dynamically via _get_corpus_stops(conn) at query time.
_FTS_STOPWORDS = frozenset([
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "must", "of", "in", "on",
    "at", "by", "to", "for", "with", "from", "and", "or", "but", "not",
    "no", "it", "its", "this", "that", "their", "there", "they", "them",
    "he", "she", "his", "her", "we", "our", "you", "your", "i", "my",
    "what", "who", "where", "when", "which", "how", "whom", "whose",
    "name", "called", "used", "ever", "first", "did",
])

# Wh-nouns that must NOT be used as mandatory AND terms in FTS.
# Two categories:
#   1. Abstract question-format words: describe the *type* of answer expected
#      ("What TERM does Holmes use?") rather than content that appears in the source.
#      Making them mandatory AND causes FTS to retrieve chunks about the word's
#      meaning rather than the answer topic.
#   2. Parsing artifacts: very short or ambiguous tokens produced by spaCy splitting
#      hyphenated compounds (e.g. "air" from "air-gun").
_WH_NOUN_BLOCKLIST = frozenset([
    "term", "word", "phrase", "title",
    "work", "book", "text", "story", "novel",
    "condition", "situation", "method", "way", "reason",
    "thing", "fact", "case", "detail", "example",
    "type", "kind", "role", "part", "form",
])

# Module-level cache: conn id → corpus stopword set (avoids repeated db_meta SELECTs)
_corpus_stops_cache: dict[int, frozenset] = {}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QuestionParse:
    wh_type: str | None
    entity_mentions: list[str]   # raw text candidates (resolved later in pipeline)
    verb_lemma: str | None       # main predicate to look up
    raw: str
    wh_noun: str | None = None   # head noun of wh-phrase, e.g. "battle" in "which battle"


@dataclass
class FactHit:
    fact_id: int
    tag: str                     # "[F1]"
    subject_name: str
    predicate: str
    object_text: str | None      # literal object or None
    object_entity_name: str | None
    qualifiers: dict | None
    negated: bool
    sentence: str
    chunk_id: int
    source_info: str | None
    score: float                 # 1.0 exact predicate, 0.8 via synonym
    subject_id: int = 0          # for relevance gate
    object_id: int | None = None # for relevance gate


@dataclass
class ChunkHit:
    chunk_id: int
    tag: str                     # "[C1]"
    text: str
    source_info: str | None
    rrf_score: float


@dataclass
class RetrievalResult:
    parse: QuestionParse
    fact_hits: list[FactHit]
    chunk_hits: list[ChunkHit]
    context_text: str
    explain_text: str | None = None


@dataclass
class PropositionHit:
    prop_id: int
    predicate: str
    statement: str             # the NL sentence given to the model
    roles: dict                # {agent: id, theme: [id, ...]}
    chunk_id: int
    agent_canonical: str | None = None   # filled in by retrieve_propositions


# ---------------------------------------------------------------------------
# Dynamic helpers — corpus-derived data loaded from the DB at query time
# ---------------------------------------------------------------------------

def _get_corpus_stops(conn: sqlite3.Connection) -> frozenset:
    """Return corpus stopwords merged with the universal baseline.

    Result is cached per connection object so repeated calls within a bench
    run don't re-read db_meta on every question.
    """
    cid = id(conn)
    if cid not in _corpus_stops_cache:
        from loci.store import load_corpus_stopwords
        _corpus_stops_cache[cid] = _FTS_STOPWORDS | frozenset(load_corpus_stopwords(conn))
    return _corpus_stops_cache[cid]


def _find_predicates_dynamic(
    conn: sqlite3.Connection,
    embedder: Any,
    term: str,
    k: int = 5,
    max_distance: float = 0.35,
) -> list[str]:
    """Embed term and return predicates from vec_predicates within max_distance.

    Falls back to the static _VERB_TO_PRED / _NOUN_TO_PRED maps when
    vec_predicates is not yet populated (before the first pred-vec pass).
    Returns an empty list on any failure.
    """
    if not term or embedder is None:
        return []
    try:
        from loci.models import embed_batch
        from loci.store import search_pred_vec
        emb = embed_batch(embedder, [term], normalize=True)[0]
        return search_pred_vec(conn, emb, k=k, max_distance=max_distance)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Question parsing
# ---------------------------------------------------------------------------

def parse_question(question: str, nlp: Any = None) -> QuestionParse:
    """Parse a natural-language question.

    Uses spaCy when provided for accurate verb extraction, otherwise falls
    back to simple heuristics. Entity mention extraction is handled downstream
    by scanning the alias table directly (works regardless of capitalisation).
    """
    if nlp is not None:
        return _parse_spacy(question, nlp)
    return _parse_simple(question)


def _parse_spacy(question: str, nlp: Any) -> QuestionParse:
    doc = nlp(question)
    wh_type = next((t.lower_ for t in doc if t.lower_ in _WH_WORDS), None)

    # Main content verb: prefer ROOT VERB, else first non-AUX VERB
    verb_lemma = None
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root and root.pos_ == "VERB" and root.lemma_.lower() not in _AUX_LEMMAS:
        verb_lemma = root.lemma_.lower()
    if verb_lemma is None:
        for t in doc:
            if t.pos_ == "VERB" and t.lemma_.lower() not in _AUX_LEMMAS:
                verb_lemma = t.lemma_.lower()
                break

    # Entity mention candidates: nsubj spans + PROPN sequences
    mentions: list[str] = []
    if root:
        for child in root.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                span = doc[child.left_edge.i : child.right_edge.i + 1].text.strip()
                if span.lower() not in _WH_WORDS:
                    mentions.append(span)

    seq: list[str] = []
    for t in doc:
        if t.pos_ == "PROPN":
            seq.append(t.text)
        else:
            if seq:
                m = " ".join(seq)
                if m not in mentions:
                    mentions.append(m)
                seq = []
    if seq:
        m = " ".join(seq)
        if m not in mentions:
            mentions.append(m)

    # wh-noun: first NOUN/PROPN immediately following the wh-word, e.g. "battle"
    # in "In which battle…". spaCy's dep tree is unreliable here (the wh-word
    # often attaches to the preposition, not the noun), so positional proximity
    # is more robust.
    # Hyphenated compounds (e.g. "air-gun") are split by spaCy into [NOUN, "-", NOUN].
    # The first part of a compound is detected by checking whether the next token is
    # a hyphen, and skipped so the extraction continues to the next NOUN.
    wh_noun = None
    if wh_type:
        for i, t in enumerate(doc):
            if t.lower_ == wh_type:
                for t2 in doc[i + 1:]:
                    if t2.pos_ in ("NOUN", "PROPN"):
                        if t2.i + 1 < len(doc) and doc[t2.i + 1].text == "-":
                            continue  # first part of hyphenated compound — skip
                        wh_noun = t2.lemma_.lower()
                        break
                    if t2.text == "-":
                        continue  # hyphen between compound parts — skip
                    if t2.pos_ not in ("ADP", "DET"):
                        break  # stop at verbs/pronouns — no wh-noun here
                break

    return QuestionParse(wh_type=wh_type, entity_mentions=mentions,
                         verb_lemma=verb_lemma, raw=question, wh_noun=wh_noun)


def _parse_simple(question: str) -> QuestionParse:
    """Pure-Python fallback: heuristic wh-type + verb extraction."""
    words = question.lower().split()
    wh_type = next((w.strip("?") for w in words if w.strip("?") in _WH_WORDS), None)
    # Rough verb: first lowercase alphabetic word after the wh-word that is
    # not a stop-word or auxiliary
    _STOPS = {"the", "a", "an", "did", "do", "does", "was", "were", "is",
               "are", "has", "have", "had", "been", "be", "of", "in", "on",
               "at", "by", "to", "for", "with", "from"}
    verb_lemma = None
    wh_noun = None
    past_wh = wh_type is None
    for i, w in enumerate(words):
        cw = w.strip("?.,!;:")
        if cw in _WH_WORDS:
            past_wh = True
            # word immediately after wh-word is likely the wh-noun (e.g. "battle")
            if i + 1 < len(words):
                nw = words[i + 1].strip("?.,!;:")
                if nw.isalpha() and nw not in _STOPS and nw not in _WH_WORDS:
                    wh_noun = nw
            continue
        if past_wh and cw.isalpha() and cw not in _STOPS:
            verb_lemma = cw
            break
    return QuestionParse(wh_type=wh_type, entity_mentions=[],
                         verb_lemma=verb_lemma, raw=question, wh_noun=wh_noun)


# ---------------------------------------------------------------------------
# Entity scanning (DB-based, works with any capitalisation)
# ---------------------------------------------------------------------------

def find_mentioned_entity_ids(
    conn: sqlite3.Connection, question: str, schema: str = "main"
) -> list[int]:
    """Scan all 1-3 word spans from the question against the alias table.

    Tries longer spans first so 'sherlock holmes' is preferred over 'holmes'
    alone. Works with both lowercase and title-case input.
    """
    from loci.resolve import normalize_mention

    sp = f"{schema}." if schema != "main" else ""
    words = _NONWORD.sub(" ", question.lower()).split()
    n = len(words)
    seen: set[int] = set()
    result: list[int] = []

    for length in range(min(3, n), 0, -1):
        for start in range(n - length + 1):
            span = " ".join(words[start : start + length])
            normalized = normalize_mention(span)
            if not normalized:
                continue
            row = conn.execute(
                f"SELECT entity_id FROM {sp}aliases WHERE alias=?", [normalized]
            ).fetchone()
            if row and row["entity_id"] not in seen:
                seen.add(row["entity_id"])
                result.append(row["entity_id"])

    return result


# ---------------------------------------------------------------------------
# Predicate synonyms
# ---------------------------------------------------------------------------

def get_synonyms(
    conn: sqlite3.Connection, predicate: str, schema: str = "main"
) -> set[str]:
    """Return all synonyms of predicate (bidirectional)."""
    sp = f"{schema}." if schema != "main" else ""
    rows = conn.execute(
        f"SELECT synonym FROM {sp}predicate_synonyms WHERE predicate=? "
        f"UNION "
        f"SELECT predicate FROM {sp}predicate_synonyms WHERE synonym=?",
        [predicate, predicate],
    ).fetchall()
    return {r[0] for r in rows}


# ---------------------------------------------------------------------------
# Fact lookup (SQL)
# ---------------------------------------------------------------------------

def load_fact_hits_by_ids(
    conn: sqlite3.Connection,
    fact_ids: list[int],
    schema: str = "main",
    source_filter: set[str] | None = None,
) -> list[FactHit]:
    """Fetch FactHit objects for a ranked list of fact_ids (FTS results).

    Preserves the ranking order from the input list; scores descend from 0.70
    so SQL-exact hits (1.0/0.8) always sort above these FTS hits.
    When source_filter is set, only facts whose source is in the set are returned.
    """
    if not fact_ids:
        return []
    sp = f"{schema}." if schema != "main" else ""
    id_ph = ",".join("?" * len(fact_ids))
    src_clause = ""
    src_params: list = []
    if source_filter:
        src_ph = ",".join("?" * len(source_filter))
        src_clause = f"AND f.source IN ({src_ph})"
        src_params = sorted(source_filter)
    rows = conn.execute(
        f"""
        SELECT
            f.id, f.subject_id, f.predicate, f.object_text, f.object_id, f.qualifiers,
            f.negated, f.sentence, f.chunk_id,
            e.canonical_name AS subject_name,
            oe.canonical_name AS object_entity_name,
            s.title, s.path
        FROM {sp}facts f
        JOIN {sp}entities e ON f.subject_id = e.id
        LEFT JOIN {sp}entities oe ON f.object_id = oe.id
        JOIN {sp}chunks c ON f.chunk_id = c.id
        LEFT JOIN {sp}sources s ON c.source_id = s.id
        WHERE f.id IN ({id_ph})
          AND (f.object_text IS NOT NULL OR f.object_id IS NOT NULL)
          AND length(e.canonical_name) <= 60
          AND instr(e.canonical_name, ',') = 0
          AND instr(e.canonical_name, char(10)) = 0
          {src_clause}
        """,
        fact_ids + src_params,
    ).fetchall()
    id_to_row = {r["id"]: r for r in rows}
    hits = []
    for rank, fid in enumerate(fact_ids):
        r = id_to_row.get(fid)
        if r is None:
            continue
        # Skip facts whose subject name is entirely lowercase (common noun phrases
        # treated as entities by spaCy, e.g. "address", "young hunter", "door").
        # Real story entities have at least one uppercase letter in their name.
        sn = r["subject_name"] or ""
        if sn and sn == sn.lower():
            continue
        quals = json.loads(r["qualifiers"]) if r["qualifiers"] else None
        hits.append(FactHit(
            fact_id=r["id"],
            tag="",  # renumbered later
            subject_name=sn,
            predicate=r["predicate"],
            object_text=r["object_text"],
            object_entity_name=r["object_entity_name"],
            qualifiers=quals,
            negated=bool(r["negated"]),
            sentence=r["sentence"],
            chunk_id=r["chunk_id"],
            source_info=_source_info(r["title"], r["path"]),
            score=0.7 - 0.01 * rank,
            subject_id=r["subject_id"],
            object_id=r["object_id"],
        ))
    return hits


def fact_lookup(
    conn: sqlite3.Connection,
    entity_ids: list[int],
    predicate: str | None,
    synonyms: set[str],
    schema: str = "main",
    source_filter: set[str] | None = None,
) -> list[FactHit]:
    """Indexed SQL lookup: (subject_id IN ...) AND (predicate IN ...)."""
    if not entity_ids or (not predicate and not synonyms):
        return []

    sp = f"{schema}." if schema != "main" else ""
    all_predicates = ([predicate] if predicate else []) + sorted(synonyms)
    # Score expression: exact match on primary predicate scores 1.0, synonyms 0.8
    _score_pred = predicate or (sorted(synonyms)[0] if synonyms else "")
    id_ph = ",".join("?" * len(entity_ids))
    pred_ph = ",".join("?" * len(all_predicates))
    src_clause = ""
    src_params: list = []
    if source_filter:
        src_ph = ",".join("?" * len(source_filter))
        src_clause = f"AND f.source IN ({src_ph})"
        src_params = sorted(source_filter)

    rows = conn.execute(
        f"""
        SELECT
            f.id, f.subject_id, f.predicate, f.object_text, f.object_id, f.qualifiers,
            f.negated, f.sentence, f.chunk_id,
            e.canonical_name AS subject_name,
            oe.canonical_name AS object_entity_name,
            s.title, s.path,
            CASE WHEN f.predicate=? THEN 1.0 ELSE 0.8 END AS score
        -- ^ score param: primary predicate for exact-match bonus
        FROM {sp}facts f
        JOIN {sp}entities e ON f.subject_id = e.id
        LEFT JOIN {sp}entities oe ON f.object_id = oe.id
        JOIN {sp}chunks c ON f.chunk_id = c.id
        LEFT JOIN {sp}sources s ON c.source_id = s.id
        WHERE f.subject_id IN ({id_ph})
          AND f.predicate IN ({pred_ph})
          AND (f.object_text IS NOT NULL OR f.object_id IS NOT NULL)
          AND length(e.canonical_name) <= 60
          AND instr(e.canonical_name, ',') = 0
          AND instr(e.canonical_name, char(10)) = 0
          {src_clause}
        ORDER BY score DESC, f.id
        """,
        [_score_pred] + entity_ids + all_predicates + src_params,
    ).fetchall()

    hits = []
    for i, r in enumerate(rows, 1):
        quals = json.loads(r["qualifiers"]) if r["qualifiers"] else None
        hits.append(FactHit(
            fact_id=r["id"],
            tag=f"[F{i}]",
            subject_name=r["subject_name"],
            predicate=r["predicate"],
            object_text=r["object_text"],
            object_entity_name=r["object_entity_name"],
            qualifiers=quals,
            negated=bool(r["negated"]),
            sentence=r["sentence"],
            chunk_id=r["chunk_id"],
            source_info=_source_info(r["title"], r["path"]),
            score=r["score"],
            subject_id=r["subject_id"],
            object_id=r["object_id"],
        ))
    return hits


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

def vec_search_question(
    conn: sqlite3.Connection,
    embedder: Any,
    question: str,
    k: int,
    schema: str = "main",
    *,
    embedding: list[float] | None = None,
) -> list[int]:
    """Embed question (or reuse provided embedding) and return top-k chunk_ids."""
    from loci.store import vec_search_chunks
    if embedding is None:
        from loci.models import embed_batch
        vecs = embed_batch(embedder, [question], normalize=True)
        if not vecs:
            return []
        embedding = vecs[0]
    results = vec_search_chunks(conn, embedding=embedding, k=k, schema=schema)
    return [r["chunk_id"] for r in results]


def vec_fact_search_question(
    conn: sqlite3.Connection,
    embedding: list[float],
    k: int,
    schema: str = "main",
) -> list[tuple[int, float]]:
    """Return top-k (fact_id, distance) pairs from vec_facts using a pre-computed embedding."""
    from loci.store import vec_search_facts
    try:
        results = vec_search_facts(conn, embedding=embedding, k=k, schema=schema)
        return [(r["fact_id"], r["distance"]) for r in results]
    except Exception:
        return []


def canonical_names_for_facts(
    conn: sqlite3.Connection,
    fact_ids: list[int],
    schema: str = "main",
) -> list[str]:
    """Return distinct canonical entity names (subject + object) for the given facts.

    Only proper-noun-ish names (has uppercase, no comma, length ≤ 60) are returned.
    Used by expand mode to inject bridge names into the chunk query.
    """
    if not fact_ids:
        return []
    sp = f"{schema}." if schema != "main" else ""
    id_ph = ",".join("?" * len(fact_ids))
    rows = conn.execute(
        f"""SELECT DISTINCT name FROM (
              SELECT e.canonical_name AS name
              FROM {sp}facts f JOIN {sp}entities e ON f.subject_id = e.id
              WHERE f.id IN ({id_ph})
              UNION
              SELECT COALESCE(oe.canonical_name, f.object_text) AS name
              FROM {sp}facts f LEFT JOIN {sp}entities oe ON f.object_id = oe.id
              WHERE f.id IN ({id_ph}) AND (f.object_id IS NOT NULL OR f.object_text IS NOT NULL)
            )
            WHERE name IS NOT NULL
              AND length(name) <= 60
              AND instr(name, ',') = 0""",
        fact_ids + fact_ids,
    ).fetchall()
    names = []
    for r in rows:
        name = r[0]
        if name and name != name.lower():
            names.append(name)
    return names


# ---------------------------------------------------------------------------
# FTS search
# ---------------------------------------------------------------------------

def fts_search_question(
    conn: sqlite3.Connection, question: str, k: int, schema: str = "main",
    wh_noun: str | None = None,
    extra_stopwords: frozenset | None = None,
) -> list[int]:
    """BM25 full-text search: strip stopwords, OR remaining content words.

    extra_stopwords is merged with _FTS_STOPWORDS at call time (used to pass
    corpus-derived high-frequency terms without modifying the module constant).
    If wh_noun is supplied (e.g. "battle" from "which battle"), it is prepended
    as a mandatory AND term so only chunks mentioning that noun rank at all.
    """
    from loci.store import fts_search_chunks
    effective_stops = _FTS_STOPWORDS | (extra_stopwords or frozenset())
    words = _NONWORD.sub(" ", question.lower()).split()
    content = [w for w in words if w and w not in effective_stops and len(w) > 2]
    if not content:
        return []
    or_clause = " OR ".join(content)
    # Mandatory AND prefix: "battle AND (wounded OR returning OR england)"
    # Blocked for abstract question-format words and spaCy parsing artifacts.
    if (wh_noun and wh_noun not in effective_stops
            and wh_noun not in _WH_NOUN_BLOCKLIST and len(wh_noun) > 2):
        query = f"{wh_noun} AND ({or_clause})"
    else:
        query = or_clause
    try:
        results = fts_search_chunks(conn, query=query, k=k, schema=schema)
        return [r["chunk_id"] for r in results]
    except Exception:
        return []


def fact_fts_search_question(
    conn: sqlite3.Connection,
    question: str,
    k: int,
    schema: str = "main",
    *,
    source_filter: set | None = None,
) -> list[int]:
    """Stopword-stripped OR query over fts_facts → ranked fact_ids.

    When source_filter contains 'llm', queries fts_facts_llm (llm+closure only)
    instead of fts_facts (all sources). This eliminates SVO noise from rankings
    when minted injection is active.
    """
    words = _NONWORD.sub(" ", question.lower()).split()
    content = [w for w in words if w and w not in _FTS_STOPWORDS and len(w) > 2]
    if not content:
        return []
    query = " OR ".join(content)

    use_llm_table = source_filter is not None and "llm" in source_filter
    table = "fts_facts_llm" if use_llm_table else "fts_facts"

    try:
        if schema == "main":
            rows = conn.execute(
                f"SELECT rowid AS fact_id, rank FROM {table}"
                f" WHERE text MATCH ? ORDER BY rank LIMIT ?",
                [query, k],
            ).fetchall()
        else:
            sp = f"{schema}."
            rows = conn.execute(
                f"SELECT fts.rowid AS fact_id, fts.rank"
                f" FROM {sp}{table} fts WHERE fts MATCH ? ORDER BY fts.rank LIMIT ?",
                [query, k],
            ).fetchall()
        return [r["fact_id"] for r in rows]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def rrf_fuse(ranked_lists: list[list], k: int = 60, ks: list[int] | None = None) -> list[tuple]:
    """Reciprocal Rank Fusion across multiple ranked ID lists.

    Keys may be any hashable type (int for single-schema, (schema, int) for packs).
    ks: per-list k values (overrides k when provided); smaller k = higher weight for that list.
    """
    scores: dict = {}
    for i, lst in enumerate(ranked_lists):
        ki = ks[i] if ks and i < len(ks) else k
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (ki + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Context bundle
# ---------------------------------------------------------------------------

def build_context(
    fact_hits: list[FactHit],
    chunk_hits: list[ChunkHit],
    token_budget: int,
) -> str:
    """Assemble context ≤ token_budget tokens: chunks first, then facts.

    Chunks come first so reliable passage content always reaches the model.
    Facts follow as supplementary citable evidence (bridging facts, named entities).
    """
    budget_chars = token_budget * _CHARS_PER_TOKEN
    parts: list[str] = []
    used = 0

    for ch in chunk_hits:
        line = _format_chunk(ch)
        if used + len(line) > budget_chars:
            break
        parts.append(line)
        used += len(line)

    for fh in fact_hits:
        line = _format_fact(fh)
        if used + len(line) > budget_chars:
            break
        parts.append(line)
        used += len(line)

    return "\n\n".join(parts)


def _format_fact(f: FactHit) -> str:
    obj_part = ""
    if f.object_entity_name:
        obj_part = f" — {f.object_entity_name}"
    elif f.object_text:
        obj_part = f" — {f.object_text}"
    qual_part = ""
    if f.qualifiers:
        qual_part = " (" + ", ".join(f"{k}: {v}" for k, v in f.qualifiers.items()) + ")"
    neg_part = " [NOT]" if f.negated else ""
    src = f" ({f.source_info})" if f.source_info else ""
    return (
        f"{f.tag} {f.subject_name} — {f.predicate}{neg_part}{obj_part}{qual_part}\n"
        f'     "{f.sentence}"{src}'
    )


def _format_chunk(c: ChunkHit) -> str:
    src = f" ({c.source_info})" if c.source_info else ""
    return f'{c.tag} "{c.text}"{src}'


def _source_info(title: str | None, path: str | None) -> str | None:
    if title:
        return title
    if path:
        return Path(path).name
    return None


def _fact_source_set(fact_sources: str) -> set[str] | None:
    """Map fact_sources config value to a set of allowed source tags, or None for 'all'."""
    if fact_sources == "minted":
        return {"llm"}
    if fact_sources == "minted+coref":
        return {"llm", "coref"}
    return None


# ---------------------------------------------------------------------------
# Chunk loading
# ---------------------------------------------------------------------------

def load_chunk_hits(
    conn: sqlite3.Connection,
    fused: list[tuple],
    token_budget: int,
    offset: int = 1,
) -> list[ChunkHit]:
    """Fetch chunk rows for fused results, return as tagged ChunkHits.

    fused keys may be plain int (main schema) or (schema, int) tuples (packs).
    """
    if not fused:
        return []
    budget_chars = token_budget * _CHARS_PER_TOKEN
    hits: list[ChunkHit] = []
    used = 0
    for i, (key, score) in enumerate(fused, start=offset):
        if isinstance(key, tuple):
            schema, chunk_id = key
            sp = f"{schema}." if schema != "main" else ""
        else:
            chunk_id = key
            sp = ""
        row = conn.execute(
            f"SELECT c.text, src.title, src.path "
            f"FROM {sp}chunks c LEFT JOIN {sp}sources src ON c.source_id = src.id "
            f"WHERE c.id=?",
            [chunk_id],
        ).fetchone()
        if row is None:
            continue
        text = row["text"]
        if used + len(text) > budget_chars:
            break
        hits.append(ChunkHit(
            chunk_id=chunk_id,
            tag=f"[C{i}]",
            text=text,
            source_info=_source_info(row["title"], row["path"]),
            rrf_score=score,
        ))
        used += len(text)
    return hits


# ---------------------------------------------------------------------------
# Explain text
# ---------------------------------------------------------------------------

def build_explain(
    parse: QuestionParse,
    entity_ids: list[int],
    synonyms: set[str],
    fact_hits: list[FactHit],
    vec_ids: list[int],
    fts_ids: list[int],
    fused: list[tuple],
    conn: sqlite3.Connection,
    schemas: list[str] | None = None,
    fact_fts_ids: list[int] | None = None,
    vec_fact_hits: list[tuple[int, float, str]] | None = None,
    expand_terms: list[str] | None = None,
) -> str:
    lines: list[str] = ["=== Question Parse ==="]
    lines.append(f"  wh-type : {parse.wh_type or '(none)'}")
    lines.append(f"  wh-noun : {parse.wh_noun or '(none)'}")
    lines.append(f"  predicate: {parse.verb_lemma or '(none)'}")
    if synonyms:
        lines.append(f"  synonyms : {', '.join(sorted(synonyms))}")
    lines.append(f"  entity_ids: {entity_ids}")
    if schemas and len(schemas) > 1:
        lines.append(f"  schemas  : {schemas}")

    lines.append("\n=== Fact Lookup ===")
    lines.append(f"  hits: {len(fact_hits)}")
    for fh in fact_hits[:5]:
        lines.append(f"  {fh.tag} {fh.subject_name} — {fh.predicate} — {fh.object_text} (score={fh.score})")

    lines.append("\n=== Fact FTS ===")
    fts_fact_hits = [fh for fh in fact_hits if fh.fact_id in set(fact_fts_ids or [])]
    lines.append(f"  new_fact_ids: {(fact_fts_ids or [])[:5]}")
    lines.append(f"  in_context: {len(fts_fact_hits)}")

    lines.append("\n=== Fact Vec ===")
    vf = vec_fact_hits or []
    lines.append(f"  candidates: {len(vf)}")
    for fid, dist, schema in vf[:5]:
        lines.append(f"  fact_id={fid} dist={dist:.4f} schema={schema}")
    if expand_terms:
        lines.append(f"  expand_terms (injected): {expand_terms}")

    lines.append("\n=== Vector Search ===")
    lines.append(f"  top-{len(vec_ids)}: chunk_ids={vec_ids[:5]}")

    lines.append("\n=== FTS Search ===")
    lines.append(f"  top-{len(fts_ids)}: chunk_ids={fts_ids[:5]}")

    lines.append("\n=== RRF Fusion (top-5) ===")
    for key, score in fused[:5]:
        cid = key[1] if isinstance(key, tuple) else key
        lines.append(f"  chunk_id={cid}  rrf={score:.4f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main retrieve function
# ---------------------------------------------------------------------------

def retrieve(
    question: str,
    *,
    conn: sqlite3.Connection,
    cfg: Config,
    embedder: Any = None,
    nlp: Any = None,
    explain: bool = False,
    pack_schemas: list[str] | None = None,
    timings: dict | None = None,
    hyde_embedding: list[float] | None = None,
    source_ids: list[int] | None = None,
) -> RetrievalResult:
    """Full hybrid retrieval pipeline, optionally fanning out across pack schemas.

    If timings dict is provided, per-stage millisecond measurements are written
    into it (parse_ms, fact_ms, vec_ms, fts_ms, fusion_ms).
    """
    schemas = ["main"] + (pack_schemas or [])
    source_filter = _fact_source_set(cfg.retrieval.fact_sources)

    # 1. Parse
    t0 = time.perf_counter()
    parse = parse_question(question, nlp=nlp)
    if timings is not None:
        timings["parse_ms"] = (time.perf_counter() - t0) * 1000

    # 2–4. Per-schema: entity resolution → synonyms → fact lookup
    all_fact_hits: list[FactHit] = []
    main_entity_ids: list[int] = []
    main_synonyms: set[str] = set()

    # Dynamic predicate discovery: embed verb + wh_noun → nearest DB predicates.
    # Falls back to static maps when vec_predicates is not yet populated.
    _dynamic_preds: set[str] = set()
    _terms_to_embed = [t for t in [parse.verb_lemma, parse.wh_noun] if t]
    for term in _terms_to_embed:
        _dynamic_preds.update(_find_predicates_dynamic(conn, embedder, term))
    # Static map fallback: used when vec_predicates is empty (new DB, pre-pred-vec pass)
    _static_preds: set[str] = set(_VERB_TO_PRED.get(parse.verb_lemma or "", []))
    if parse.wh_noun:
        _static_preds.update(_NOUN_TO_PRED.get(parse.wh_noun, []))
    _extra_preds = _dynamic_preds or _static_preds  # prefer dynamic; fall back to static

    t0 = time.perf_counter()
    for schema in schemas:
        s_eids = find_mentioned_entity_ids(conn, question, schema=schema)
        s_syns = get_synonyms(conn, parse.verb_lemma, schema=schema) if parse.verb_lemma else set()
        s_syns |= _extra_preds
        if schema == "main":
            main_entity_ids = s_eids
            main_synonyms = s_syns
        all_fact_hits.extend(
            fact_lookup(conn, s_eids, parse.verb_lemma, s_syns, schema=schema,
                        source_filter=source_filter)
        )
    fact_ms = (time.perf_counter() - t0) * 1000
    if timings is not None:
        timings["fact_ms"] = fact_ms

    # Fact FTS (additive): keyword recall over fact triples + source sentences.
    # When source_filter includes 'llm', routes to fts_facts_llm (no SVO noise)
    # so no over-pull is needed. Otherwise over-pull 4× to compensate for dilution.
    _use_llm_fts = source_filter is not None and "llm" in source_filter
    _fts_fact_k = cfg.retrieval.fact_fts_top_k if _use_llm_fts else (
        cfg.retrieval.fact_fts_top_k * (4 if source_filter else 1)
    )
    all_fact_fts_ids: list[int] = []
    seen_fids = {h.fact_id for h in all_fact_hits}
    for schema in schemas:
        fids = fact_fts_search_question(
            conn, question, _fts_fact_k, schema=schema, source_filter=source_filter
        )
        fids = [fid for fid in fids if fid not in seen_fids]
        if fids:
            all_fact_hits.extend(load_fact_hits_by_ids(conn, fids, schema=schema,
                                                        source_filter=source_filter))
            seen_fids.update(fids)
            all_fact_fts_ids.extend(fids)

    # 5. Vec search (embed question once; reuse for both chunk and fact vec search)
    all_vec_keys: list[tuple] = []
    question_embedding: list[float] | None = None
    expand_terms: list[str] = []
    all_vec_fact_hits: list[tuple[int, float, str]] = []  # (fact_id, distance, schema)
    _vec_raw_scores: dict[tuple, float] = {}   # (schema, chunk_id) → cosine similarity (1-dist)
    t0 = time.perf_counter()
    _rerank_on = cfg.retrieval.rerank_mode != "off"
    _pool_k = cfg.retrieval.rerank_pool if _rerank_on else cfg.retrieval.vec_top_k
    if embedder is not None:
        from loci.models import embed_batch
        from loci.store import vec_search_chunks as _vec_search_chunks
        vecs = embed_batch(embedder, [question], normalize=True)
        if vecs:
            question_embedding = vecs[0]

        # HyDE-lite: average question embedding with hypothetical answer embedding
        # when the caller provides one. This bridges vocabulary gaps in paraphrase Qs.
        _vec_query_embedding = question_embedding
        if hyde_embedding is not None and question_embedding is not None:
            avg = [(a + b) / 2 for a, b in zip(question_embedding, hyde_embedding)]
            norm = sum(x * x for x in avg) ** 0.5
            _vec_query_embedding = [x / norm for x in avg] if norm > 0 else avg

        for schema in schemas:
            if _vec_query_embedding is not None:
                raw = _vec_search_chunks(conn, embedding=_vec_query_embedding,
                                         k=_pool_k, schema=schema)
                for r in raw:
                    key = (schema, r["chunk_id"])
                    all_vec_keys.append(key)
                    _vec_raw_scores[key] = max(0.0, 1.0 - r["distance"])
            else:
                ids = vec_search_question(conn, embedder, question,
                                          _pool_k, schema=schema)
                all_vec_keys.extend((schema, cid) for cid in ids)

        # Vec-over-facts (mode switch)
        if cfg.retrieval.fact_vec_mode != "off" and question_embedding is not None:
            for schema in schemas:
                for fid, dist in vec_fact_search_question(
                    conn, question_embedding, cfg.retrieval.fact_vec_top_k, schema=schema
                ):
                    if fid not in seen_fids:
                        all_vec_fact_hits.append((fid, dist, schema))
                        seen_fids.add(fid)

            # Source filter: drop vec_fact hits from other sources so expand mode
            # only injects entity names from the question's own book.
            if source_ids is not None and all_vec_fact_hits:
                ph = ",".join("?" * len(source_ids))
                _valid_fids = {r[0] for r in conn.execute(
                    f"SELECT f.id FROM facts f JOIN chunks c ON f.chunk_id = c.id"
                    f" WHERE c.source_id IN ({ph})", source_ids,
                )}
                all_vec_fact_hits = [
                    (fid, dist, s) for fid, dist, s in all_vec_fact_hits
                    if fid in _valid_fids
                ]

            if cfg.retrieval.fact_vec_mode == "surface":
                for fid, dist, schema in all_vec_fact_hits:
                    hits = load_fact_hits_by_ids(conn, [fid], schema=schema)
                    for h in hits:
                        h.score = max(0.0, 0.6 - 0.05 * dist)
                    all_fact_hits.extend(hits)

            elif cfg.retrieval.fact_vec_mode == "expand":
                top_fids = [fid for fid, _, _ in all_vec_fact_hits[: cfg.retrieval.fact_expand_names]]
                expand_terms = canonical_names_for_facts(conn, top_fids, schema="main")

    if timings is not None:
        timings["vec_ms"] = (time.perf_counter() - t0) * 1000

    # Relevance gate: when source filter is active, drop facts whose subject/object
    # entity doesn't appear in the question's resolved entity set. This prevents
    # topically-loose minted facts from consuming slots on paraphrase questions.
    if source_filter and main_entity_ids:
        entity_id_set = set(main_entity_ids)
        all_fact_hits = [
            h for h in all_fact_hits
            if h.subject_id in entity_id_set or h.object_id in entity_id_set
        ]
    elif source_filter and not main_entity_ids:
        # No entities resolved → cannot gate by entity; drop all to avoid injecting
        # unrelated minted facts into slots (the bypass flaw from Phase Q bench).
        all_fact_hits = []

    # Re-sort, cap, and renumber tags (after vec-surface may have added hits)
    all_fact_hits.sort(key=lambda h: -h.score)
    all_fact_hits = all_fact_hits[: cfg.retrieval.max_facts_in_context]
    for i, h in enumerate(all_fact_hits, 1):
        h.tag = f"[F{i}]"

    # 6. FTS search (expand mode may inject bridge names into the query)
    all_fts_keys: list[tuple] = []
    _fts_raw_scores: dict[tuple, float] = {}   # (schema, chunk_id) → normalised rank 0-1
    t0 = time.perf_counter()
    fts_question = question
    if expand_terms:
        fts_question = question + " " + " ".join(expand_terms)
    _fts_pool_k = cfg.retrieval.rerank_pool if _rerank_on else cfg.retrieval.fts_top_k
    _corpus_stops = _get_corpus_stops(conn)
    for schema in schemas:
        fts_ids = fts_search_question(conn, fts_question, _fts_pool_k, schema=schema,
                                      wh_noun=parse.wh_noun, extra_stopwords=_corpus_stops)
        for rank, cid in enumerate(fts_ids):
            key = (schema, cid)
            all_fts_keys.append(key)
            _fts_raw_scores[key] = max(0.0, 1.0 - rank / max(len(fts_ids), 1))
    if timings is not None:
        timings["fts_ms"] = (time.perf_counter() - t0) * 1000

    # Source filter: restrict to chunks from specified sources only
    if source_ids is not None:
        ph = ",".join("?" * len(source_ids))
        _valid_cids = {r[0] for r in conn.execute(
            f"SELECT id FROM chunks WHERE source_id IN ({ph})", source_ids
        )}
        all_vec_keys = [(s, cid) for s, cid in all_vec_keys if cid in _valid_cids]
        all_fts_keys = [(s, cid) for s, cid in all_fts_keys if cid in _valid_cids]

    # 7. RRF fusion + optional blend rerank + context build
    t0 = time.perf_counter()
    fused = rrf_fuse([all_vec_keys, all_fts_keys],
                     ks=[cfg.retrieval.vec_rrf_k, cfg.retrieval.rrf_k])

    # Blend rerank: re-sort fused pool by max(vec_cosine, fts_position_score)
    if cfg.retrieval.rerank_mode == "blend" and (_vec_raw_scores or _fts_raw_scores):
        fused = sorted(
            fused,
            key=lambda kv: -(0.5 * _vec_raw_scores.get(kv[0], 0.0)
                             + 0.5 * _fts_raw_scores.get(kv[0], 0.0)),
        )

    chunk_hits = load_chunk_hits(
        conn, fused, cfg.retrieval.context_token_budget, offset=1
    )
    context = build_context(all_fact_hits, chunk_hits, cfg.retrieval.context_token_budget)
    if timings is not None:
        timings["fusion_ms"] = (time.perf_counter() - t0) * 1000

    # 8. Explain
    explain_text = None
    if explain:
        all_vec_ids = [cid for (_, cid) in all_vec_keys]
        all_fts_ids = [cid for (_, cid) in all_fts_keys]
        explain_text = build_explain(
            parse, main_entity_ids, main_synonyms, all_fact_hits,
            all_vec_ids, all_fts_ids, fused, conn, schemas=schemas,
            fact_fts_ids=all_fact_fts_ids,
            vec_fact_hits=all_vec_fact_hits,
            expand_terms=expand_terms,
        )
        explain_text += f"\n\n  fact_lookup_ms: {fact_ms:.1f}"

    return RetrievalResult(
        parse=parse,
        fact_hits=all_fact_hits,
        chunk_hits=chunk_hits,
        context_text=context,
        explain_text=explain_text,
    )


# ---------------------------------------------------------------------------
# Proposition-path retrieval (design-v1)
# ---------------------------------------------------------------------------

def find_prop_entity_ids(
    conn: sqlite3.Connection, question: str
) -> list[int]:
    """Scan 1-3 word spans from the question against prop_entity_aliases.

    Uses the same span-scanning logic as find_mentioned_entity_ids but
    queries prop_entity_aliases (clean, FIX2-compliant) instead of aliases.
    """
    words = _NONWORD.sub(" ", question.lower()).split()
    n = len(words)
    seen: set[int] = set()
    result: list[int] = []
    for length in range(min(3, n), 0, -1):
        for start in range(n - length + 1):
            span = " ".join(words[start : start + length])
            if not span:
                continue
            row = conn.execute(
                "SELECT prop_entity_id FROM prop_entity_aliases WHERE alias=?", [span]
            ).fetchone()
            if row and row[0] not in seen:
                seen.add(row[0])
                result.append(row[0])
    return result


def retrieve_propositions(
    question: str,
    conn: sqlite3.Connection,
    nlp: Any = None,
    embedder: Any = None,
    k: int = 3,
    source_ids: list[int] | None = None,
) -> list["PropositionHit"]:
    """Proposition-path retrieval: returns top-k ranked propositions.

    Gathers candidates from entity-posting, FTS, and (if embedder is given)
    vector-similarity paths, then ranks by semantic relevance.  Predicate match
    is a ranking *bonus*, not a hard filter — kill/murder, work_as/be etc. are
    no longer blocked.  Returns an empty list when no candidates pass the
    minimum-score threshold.
    """
    from loci.store import (
        get_proposition,
        get_prop_agent_canonical,
        fts_search_propositions,
        vec_search_propositions,
    )

    parse = parse_question(question, nlp=nlp)
    predicate = parse.verb_lemma

    # Build the predicate-synonym set (used as a ranking bonus, not a gate).
    predicates_to_try: set[str] = set()
    if not predicate:
        words = _NONWORD.sub(" ", question.lower()).split()
        for w in words:
            cw = w.strip("?.,!;:'\"")
            if cw in _NOUN_TO_PRED:
                predicates_to_try.update(_NOUN_TO_PRED[cw])
    else:
        predicates_to_try = {predicate} | set(_VERB_TO_PRED.get(predicate, []))

    # Also scan question nouns to augment predicates_to_try — handles cases like
    # "term for profession" where the verb "use/describe" hijacks but the noun
    # "profession" maps to the correct "work_as" predicate.
    # Conditional nouns (murderer, killer, etc.) only augment when no verb was
    # found — when a verb IS present they describe an attribute ("is a murderer")
    # rather than the predicate topic, and including them fires wrong props.
    q_words = _NONWORD.sub(" ", question.lower()).split()
    for w in q_words:
        cw = w.strip("?.,!;:'\"")
        if cw in _NOUN_TO_PRED:
            if cw in _VERB_CONDITIONAL_NOUNS and predicate:
                continue  # skip when main verb present
            predicates_to_try.update(_NOUN_TO_PRED[cw])

    # No predicate signal at all → we have no way to select relevant propositions.
    # Fall back to chunk retrieval (caller gets empty list → falls through to chunks).
    if not predicates_to_try:
        return []

    # ── Path 1: entity-posting candidates (NO predicate filter) ──────────────
    prop_entity_ids = find_prop_entity_ids(conn, question)
    entity_prop_ids: set[int] = set()
    if prop_entity_ids:
        id_ph = ",".join("?" * len(prop_entity_ids))
        rows = conn.execute(
            f"SELECT DISTINCT prop_id FROM proposition_entities"
            f" WHERE prop_entity_id IN ({id_ph})",
            prop_entity_ids,
        ).fetchall()
        entity_prop_ids = {r[0] for r in rows}

    # ── Path 2: FTS candidates (NO predicate filter) ──────────────────────────
    fts_scores: dict[int, float] = {}
    words = _NONWORD.sub(" ", question.lower()).split()
    content = [w for w in words if w and w not in _PROP_FTS_STOPWORDS and len(w) > 2]
    for pred in predicates_to_try:
        for pw in pred.replace("_", " ").split():
            if len(pw) > 2 and pw not in _PROP_FTS_STOPWORDS:
                content.append(pw)
    if content:
        fts_query = " OR ".join(set(content))
        fts_hits = fts_search_propositions(conn, query=fts_query, k=30)
        for rank, hit in enumerate(fts_hits):
            fts_scores[hit["prop_id"]] = max(0.0, 1.0 - rank / max(len(fts_hits), 1))

    # ── Path 3: vector-similarity candidates ─────────────────────────────────
    vec_scores: dict[int, float] = {}
    if embedder is not None:
        try:
            from loci.models import embed_batch
            vecs = embed_batch(embedder, [question], normalize=True)
            if vecs:
                for hit in vec_search_propositions(conn, embedding=vecs[0], k=30):
                    # distance is L2 / cosine; smaller = more similar
                    vec_scores[hit["prop_id"]] = max(0.0, 1.0 - hit["distance"] / 2.0)
        except Exception:
            pass

    # ── Union of all candidates ───────────────────────────────────────────────
    all_prop_ids = entity_prop_ids | set(fts_scores) | set(vec_scores)
    if not all_prop_ids:
        return []

    # Source filter: keep only propositions from the specified source(s)
    if source_ids is not None and all_prop_ids:
        ph_s = ",".join("?" * len(source_ids))
        ph_p = ",".join("?" * len(all_prop_ids))
        valid_pids = {r[0] for r in conn.execute(
            f"SELECT p.id FROM propositions p JOIN chunks c ON p.chunk_id = c.id"
            f" WHERE c.source_id IN ({ph_s}) AND p.id IN ({ph_p})",
            list(source_ids) + list(all_prop_ids),
        )}
        all_prop_ids = valid_pids
        if not all_prop_ids:
            return []

    # Batch-load propositions once to avoid repeated DB round-trips.
    props: dict[int, dict] = {}
    for pid in all_prop_ids:
        p = get_proposition(conn, pid)
        if p:
            props[pid] = p

    if not props:
        return []

    # ── Score ─────────────────────────────────────────────────────────────────
    # Primary: vec similarity (if available), else FTS rank.
    # Bonuses: predicate in synonym set (+0.25), entity match (+0.10).
    use_vec = bool(vec_scores)

    def _score(pid: int) -> float:
        p = props.get(pid)
        if p is None:
            return -1.0
        base = vec_scores.get(pid, 0.0) if use_vec else fts_scores.get(pid, 0.0)
        pred_bonus = 0.25 if (predicates_to_try and p["predicate"] in predicates_to_try) else 0.0
        entity_bonus = 0.10 if pid in entity_prop_ids else 0.0
        return base + pred_bonus + entity_bonus

    scored = sorted(
        [(pid, _score(pid)) for pid in props],
        key=lambda x: -x[1],
    )

    # Predicate-alignment gate: require at least one candidate to match a synonym
    # predicate, in both vec and non-vec modes.  Without this gate, vec similarity
    # alone fires the prop path for every question (100/100 in bench), collapsing
    # fact/paraphrase scores.  Vec improves RANKING within aligned candidates; it
    # does not replace predicate alignment as the selectivity signal.
    if predicates_to_try:
        pred_aligned = [
            (pid, s) for pid, s in scored
            if props[pid]["predicate"] in predicates_to_try
        ]
        if not pred_aligned:
            return []
        scored = pred_aligned

    # Minimum-score gate: keeps the negative bucket safe by rejecting very weak
    # matches (random FTS noise with no entity or predicate signal).
    _MIN_SCORE = 0.05

    def _make_hit(p: dict) -> "PropositionHit":
        roles = json.loads(p["roles"]) if isinstance(p["roles"], str) else p["roles"]
        return PropositionHit(
            prop_id=p["id"],
            predicate=p["predicate"],
            statement=p["statement"],
            roles=roles,
            chunk_id=p["chunk_id"],
            agent_canonical=get_prop_agent_canonical(conn, p["id"]),
        )

    result: list[PropositionHit] = []
    for pid, score in scored[:k]:
        if score < _MIN_SCORE:
            break
        result.append(_make_hit(props[pid]))

    return result

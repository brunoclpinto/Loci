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
_CHARS_PER_TOKEN = 4
_NONWORD = re.compile(r"[^\w\s]")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QuestionParse:
    wh_type: str | None
    entity_mentions: list[str]   # raw text candidates (resolved later in pipeline)
    verb_lemma: str | None       # main predicate to look up
    raw: str


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

    return QuestionParse(wh_type=wh_type, entity_mentions=mentions,
                         verb_lemma=verb_lemma, raw=question)


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
    past_wh = wh_type is None
    for w in words:
        cw = w.strip("?.,!;:")
        if cw in _WH_WORDS:
            past_wh = True
            continue
        if past_wh and cw.isalpha() and cw not in _STOPS:
            verb_lemma = cw
            break
    return QuestionParse(wh_type=wh_type, entity_mentions=[],
                         verb_lemma=verb_lemma, raw=question)


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

def fact_lookup(
    conn: sqlite3.Connection,
    entity_ids: list[int],
    predicate: str | None,
    synonyms: set[str],
    schema: str = "main",
) -> list[FactHit]:
    """Indexed SQL lookup: (subject_id IN ...) AND (predicate IN ...)."""
    if not entity_ids or not predicate:
        return []

    sp = f"{schema}." if schema != "main" else ""
    all_predicates = [predicate] + sorted(synonyms)
    id_ph = ",".join("?" * len(entity_ids))
    pred_ph = ",".join("?" * len(all_predicates))

    rows = conn.execute(
        f"""
        SELECT
            f.id, f.predicate, f.object_text, f.object_id, f.qualifiers,
            f.negated, f.sentence, f.chunk_id,
            e.canonical_name AS subject_name,
            oe.canonical_name AS object_entity_name,
            s.title, s.path,
            CASE WHEN f.predicate=? THEN 1.0 ELSE 0.8 END AS score
        FROM {sp}facts f
        JOIN {sp}entities e ON f.subject_id = e.id
        LEFT JOIN {sp}entities oe ON f.object_id = oe.id
        JOIN {sp}chunks c ON f.chunk_id = c.id
        LEFT JOIN {sp}sources s ON c.source_id = s.id
        WHERE f.subject_id IN ({id_ph})
          AND f.predicate IN ({pred_ph})
        ORDER BY score DESC, f.id
        """,
        [predicate] + entity_ids + all_predicates,
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
) -> list[int]:
    """Embed question and return top-k chunk_ids by vector similarity."""
    from loci.models import embed_batch
    from loci.store import vec_search_chunks
    vecs = embed_batch(embedder, [question], normalize=True)
    if not vecs:
        return []
    results = vec_search_chunks(conn, embedding=vecs[0], k=k, schema=schema)
    return [r["chunk_id"] for r in results]


# ---------------------------------------------------------------------------
# FTS search
# ---------------------------------------------------------------------------

def fts_search_question(
    conn: sqlite3.Connection, question: str, k: int, schema: str = "main"
) -> list[int]:
    """BM25 full-text search on the question string."""
    from loci.store import fts_search_chunks
    clean = _NONWORD.sub(" ", question).strip()
    if not clean:
        return []
    try:
        results = fts_search_chunks(conn, query=clean, k=k, schema=schema)
        return [r["chunk_id"] for r in results]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def rrf_fuse(ranked_lists: list[list], k: int = 60) -> list[tuple]:
    """Reciprocal Rank Fusion across multiple ranked ID lists.

    Keys may be any hashable type (int for single-schema, (schema, int) for packs).
    """
    scores: dict = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Context bundle
# ---------------------------------------------------------------------------

def build_context(
    fact_hits: list[FactHit],
    chunk_hits: list[ChunkHit],
    token_budget: int,
) -> str:
    """Assemble context ≤ token_budget tokens: facts first, then chunks."""
    budget_chars = token_budget * _CHARS_PER_TOKEN
    parts: list[str] = []
    used = 0

    for fh in fact_hits:
        line = _format_fact(fh)
        if used + len(line) > budget_chars:
            break
        parts.append(line)
        used += len(line)

    for ch in chunk_hits:
        line = _format_chunk(ch)
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
    preview = c.text[:300].replace("\n", " ")
    src = f" ({c.source_info})" if c.source_info else ""
    return f'{c.tag} "{preview}..."{src}' if len(c.text) > 300 else f'{c.tag} "{c.text}"{src}'


def _source_info(title: str | None, path: str | None) -> str | None:
    if title:
        return title
    if path:
        return Path(path).name
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
) -> str:
    lines: list[str] = ["=== Question Parse ==="]
    lines.append(f"  wh-type : {parse.wh_type or '(none)'}")
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
) -> RetrievalResult:
    """Full hybrid retrieval pipeline, optionally fanning out across pack schemas.

    Steps: question parse → per-schema entity scan + fact SQL →
    per-schema vec + FTS → RRF → context bundle.
    """
    schemas = ["main"] + (pack_schemas or [])

    # 1. Parse
    parse = parse_question(question, nlp=nlp)

    # 2–4. Per-schema: entity resolution → synonyms → fact lookup
    all_fact_hits: list[FactHit] = []
    main_entity_ids: list[int] = []      # for explain (main schema only)
    main_synonyms: set[str] = set()

    t0 = time.perf_counter()
    for schema in schemas:
        s_eids = find_mentioned_entity_ids(conn, question, schema=schema)
        s_syns = get_synonyms(conn, parse.verb_lemma, schema=schema) if parse.verb_lemma else set()
        if schema == "main":
            main_entity_ids = s_eids
            main_synonyms = s_syns
        all_fact_hits.extend(
            fact_lookup(conn, s_eids, parse.verb_lemma, s_syns, schema=schema)
        )
    fact_ms = (time.perf_counter() - t0) * 1000

    # Re-sort by score and renumber tags across all schemas
    all_fact_hits.sort(key=lambda h: -h.score)
    for i, h in enumerate(all_fact_hits, 1):
        h.tag = f"[F{i}]"

    # 5–6. Per-schema vec + FTS → (schema, chunk_id) keys for RRF
    all_vec_keys: list[tuple] = []
    all_fts_keys: list[tuple] = []

    for schema in schemas:
        if embedder is not None:
            ids = vec_search_question(conn, embedder, question,
                                      cfg.retrieval.vec_top_k, schema=schema)
            all_vec_keys.extend((schema, cid) for cid in ids)
        fts_ids = fts_search_question(conn, question, cfg.retrieval.fts_top_k, schema=schema)
        all_fts_keys.extend((schema, cid) for cid in fts_ids)

    # 7. RRF fusion
    fused = rrf_fuse([all_vec_keys, all_fts_keys], k=cfg.retrieval.rrf_k)

    # 8. Load chunk rows
    chunk_hits = load_chunk_hits(
        conn, fused, cfg.retrieval.context_token_budget, offset=len(all_fact_hits) + 1
    )

    # 9. Build context bundle
    context = build_context(all_fact_hits, chunk_hits, cfg.retrieval.context_token_budget)

    # 10. Explain
    explain_text = None
    if explain:
        all_vec_ids = [cid for (_, cid) in all_vec_keys]
        all_fts_ids = [cid for (_, cid) in all_fts_keys]
        explain_text = build_explain(
            parse, main_entity_ids, main_synonyms, all_fact_hits,
            all_vec_ids, all_fts_ids, fused, conn, schemas=schemas,
        )
        explain_text += f"\n\n  fact_lookup_ms: {fact_ms:.1f}"

    return RetrievalResult(
        parse=parse,
        fact_hits=all_fact_hits,
        chunk_hits=chunk_hits,
        context_text=context,
        explain_text=explain_text,
    )

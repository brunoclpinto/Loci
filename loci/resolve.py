"""Entity resolution: normalize → exact alias → fuzzy token-subset → embed → create."""
from __future__ import annotations

import re
import sqlite3
from typing import Any

from loci.store import insert_alias, insert_entity, upsert_vec_entity, vec_search_entities

_TITLES = frozenset(
    ["mr", "mrs", "ms", "dr", "prof", "sir", "lady", "lord", "miss", "saint", "st"]
)
_PUNCT_RE = re.compile(r"[.,;:!?\"'`]")


def normalize_mention(text: str) -> str:
    """Lowercase, strip honorific titles and punctuation, collapse whitespace."""
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    tokens = [t for t in cleaned.split() if t and t not in _TITLES]
    return " ".join(tokens)


def resolve_entity(
    conn: sqlite3.Connection,
    mention: str,
    *,
    embedder: Any = None,
    entity_sim_threshold: float = 0.92,
) -> int | None:
    """Resolve a text mention to an entity_id, creating one if necessary.

    Resolution order:
      1. Normalize
      2. Exact alias hit
      3. Unambiguous fuzzy token-subset match
      4. Embedding cosine similarity ≥ threshold (if embedder supplied)
      5. Create new entity + alias (+ embed its name)

    After resolving, the original lowercased mention is stored as an alias so
    that every surface form ("Mr. Sherlock Holmes", "Holmes", …) is recorded.
    Ambiguous fuzzy matches (2+) go to pending_links; first candidate is
    returned until the user runs `loci entities review`.
    """
    normalized = normalize_mention(mention)
    if not normalized:
        return None

    mention_lower = mention.lower().strip()

    # 2. Exact alias hit on normalized form
    row = conn.execute(
        "SELECT entity_id FROM aliases WHERE alias=?", [normalized]
    ).fetchone()
    if row:
        entity_id = row["entity_id"]
        _store_aliases(conn, entity_id, normalized, mention_lower)
        return entity_id

    # 3. Fuzzy token-subset (unambiguous)
    mention_tokens = set(normalized.split())
    candidates: list[int] = []
    for ent in conn.execute("SELECT id, canonical_name FROM entities"):
        ent_tokens = set(normalize_mention(ent["canonical_name"]).split())
        if mention_tokens and ent_tokens and (
            mention_tokens.issubset(ent_tokens) or ent_tokens.issubset(mention_tokens)
        ):
            candidates.append(ent["id"])

    if len(candidates) == 1:
        entity_id = candidates[0]
        _store_aliases(conn, entity_id, normalized, mention_lower)
        return entity_id

    if len(candidates) > 1:
        _record_pending(conn, normalized, candidates)
        entity_id = candidates[0]
        _store_aliases(conn, entity_id, normalized, mention_lower)
        return entity_id

    # 4. Embedding cosine similarity match
    if embedder is not None:
        from loci.models import cosine_dist_threshold, embed_batch
        vecs = embed_batch(embedder, [mention], normalize=True)
        if vecs:
            dist_limit = cosine_dist_threshold(entity_sim_threshold)
            for hit in vec_search_entities(conn, embedding=vecs[0], k=5):
                if hit["distance"] <= dist_limit:
                    entity_id = hit["entity_id"]
                    _store_aliases(conn, entity_id, normalized, mention_lower)
                    return entity_id

    # 5. Create new entity
    entity_id = insert_entity(conn, canonical_name=mention.strip())
    _store_aliases(conn, entity_id, normalized, mention_lower)

    if embedder is not None:
        from loci.models import embed_batch
        vecs = embed_batch(embedder, [mention.strip()], normalize=True)
        if vecs:
            upsert_vec_entity(conn, entity_id=entity_id, embedding=vecs[0])

    return entity_id


def _store_aliases(
    conn: sqlite3.Connection, entity_id: int, normalized: str, original_lower: str
) -> None:
    """Store both the normalized lookup form and the original lowercased form."""
    insert_alias(conn, entity_id=entity_id, alias=normalized)
    if original_lower != normalized:
        insert_alias(conn, entity_id=entity_id, alias=original_lower)


def _record_pending(
    conn: sqlite3.Connection, mention: str, candidate_ids: list[int]
) -> None:
    import json
    conn.execute(
        "INSERT OR IGNORE INTO pending_links (mention, candidate_entity_ids) VALUES (?,?)",
        [mention, json.dumps(candidate_ids)],
    )
    conn.commit()

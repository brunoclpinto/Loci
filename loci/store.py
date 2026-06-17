"""SQLite store: schema creation, migration, CRUD helpers, and attach logic."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import sqlite_vec


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def open_db(db_path: str | Path, vec_dim: int = 384) -> sqlite3.Connection:
    """Open (or create) a knowledge DB, load sqlite-vec, and apply schema.

    vec_dim is only used when creating vec tables for the first time.
    Existing DBs keep whatever dimension they were created with.
    """
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _load_extensions(conn)
    _configure(conn)
    _migrate(conn, vec_dim=vec_dim)
    return conn


def _load_extensions(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def _configure(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(f"PRAGMA mmap_size={512 * 1024 * 1024}")


def attach_pack(conn: sqlite3.Connection, pack_path: str | Path, schema: str = "pack") -> None:
    """ATTACH an external .locipack.db file under the given schema name."""
    path = Path(pack_path).expanduser()
    conn.execute(f"ATTACH DATABASE ? AS [{schema}]", [str(path)])


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL: list[str] = [
    """CREATE TABLE IF NOT EXISTS sources (
      id          INTEGER PRIMARY KEY,
      path        TEXT,
      title       TEXT,
      author      TEXT,
      meta        JSON,
      sha256      TEXT UNIQUE NOT NULL,
      ingested_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS chunks (
      id        INTEGER PRIMARY KEY,
      source_id INTEGER REFERENCES sources(id),
      ordinal   INTEGER,
      text      TEXT NOT NULL,
      sha256    TEXT UNIQUE NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS entities (
      id             INTEGER PRIMARY KEY,
      canonical_name TEXT NOT NULL,
      kind           TEXT DEFAULT 'unknown',
      created_at     TEXT DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS aliases (
      entity_id INTEGER REFERENCES entities(id),
      alias     TEXT NOT NULL,
      PRIMARY KEY (entity_id, alias)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_aliases_alias ON aliases(alias)",
    """CREATE TABLE IF NOT EXISTS facts (
      id          INTEGER PRIMARY KEY,
      chunk_id    INTEGER REFERENCES chunks(id),
      sentence    TEXT NOT NULL,
      subject_id  INTEGER REFERENCES entities(id),
      predicate   TEXT NOT NULL,
      object_id   INTEGER REFERENCES entities(id),
      object_text TEXT,
      qualifiers  JSON,
      negated     INTEGER DEFAULT 0,
      confidence  REAL    DEFAULT 1.0,
      UNIQUE (chunk_id, subject_id, predicate, object_text, qualifiers)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_facts_subj_pred ON facts(subject_id, predicate)",
    "CREATE INDEX IF NOT EXISTS idx_facts_pred_obj  ON facts(predicate, object_id)",
    """CREATE TABLE IF NOT EXISTS predicate_synonyms (
      predicate TEXT,
      synonym   TEXT,
      PRIMARY KEY (predicate, synonym)
    )""",
    """CREATE TABLE IF NOT EXISTS pending_links (
      id                   INTEGER PRIMARY KEY,
      mention              TEXT NOT NULL UNIQUE,
      candidate_entity_ids JSON NOT NULL,
      chunk_id             INTEGER REFERENCES chunks(id),
      created_at           TEXT DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS db_meta (
      key   TEXT PRIMARY KEY,
      value TEXT NOT NULL
    )""",
    """CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks
       USING fts5(text, content='chunks', content_rowid='id')""",
    # Keep fts_chunks in sync with chunks
    """CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
      INSERT INTO fts_chunks(rowid, text) VALUES (new.id, new.text);
    END""",
    """CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
      INSERT INTO fts_chunks(fts_chunks, rowid, text)
        VALUES ('delete', old.id, old.text);
    END""",
    """CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
      INSERT INTO fts_chunks(fts_chunks, rowid, text)
        VALUES ('delete', old.id, old.text);
      INSERT INTO fts_chunks(rowid, text) VALUES (new.id, new.text);
    END""",
    # Standalone (non-content) FTS5 over fact triples + source sentences.
    # rowid == facts.id so search results map back directly.
    """CREATE VIRTUAL TABLE IF NOT EXISTS fts_facts USING fts5(text)""",
    # LLM-only FTS index (source IN ('llm','closure')) for high-precision minted injection.
    """CREATE VIRTUAL TABLE IF NOT EXISTS fts_facts_llm USING fts5(text)""",
    # -------------------------------------------------------------------------
    # Proposition layer — clean entity + event representation (design-v1)
    # Kept separate from the legacy entities/aliases tables to avoid alias
    # collisions from FIX1 garbage and to enforce spec-compliant alias sets.
    # -------------------------------------------------------------------------
    """CREATE TABLE IF NOT EXISTS prop_entities (
      id        INTEGER PRIMARY KEY,
      canonical TEXT NOT NULL UNIQUE,
      kind      TEXT DEFAULT 'PERSON'
    )""",
    """CREATE TABLE IF NOT EXISTS prop_entity_aliases (
      prop_entity_id INTEGER REFERENCES prop_entities(id),
      alias          TEXT NOT NULL,
      PRIMARY KEY (prop_entity_id, alias)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_prop_entity_aliases ON prop_entity_aliases(alias)",
    """CREATE TABLE IF NOT EXISTS propositions (
      id         INTEGER PRIMARY KEY,
      prop_ref   TEXT,
      chunk_id   INTEGER REFERENCES chunks(id),
      predicate  TEXT NOT NULL,
      roles      JSON NOT NULL,
      polarity   TEXT DEFAULT 'positive',
      salience   TEXT DEFAULT 'medium',
      confidence REAL DEFAULT 1.0,
      statement  TEXT NOT NULL,
      evidence   TEXT,
      char_span  JSON
    )""",
    "CREATE INDEX IF NOT EXISTS idx_propositions_predicate ON propositions(predicate)",
    """CREATE TABLE IF NOT EXISTS proposition_entities (
      prop_id        INTEGER REFERENCES propositions(id),
      prop_entity_id INTEGER REFERENCES prop_entities(id),
      role           TEXT NOT NULL,
      PRIMARY KEY (prop_id, prop_entity_id, role)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_proposition_entities_entity ON proposition_entities(prop_entity_id)",
    # FTS5 content table over proposition statements (same pattern as fts_chunks).
    """CREATE VIRTUAL TABLE IF NOT EXISTS fts_propositions
       USING fts5(statement, content='propositions', content_rowid='id')""",
    """CREATE TRIGGER IF NOT EXISTS propositions_ai AFTER INSERT ON propositions BEGIN
      INSERT INTO fts_propositions(rowid, statement) VALUES (new.id, new.statement);
    END""",
    """CREATE TRIGGER IF NOT EXISTS propositions_ad AFTER DELETE ON propositions BEGIN
      INSERT INTO fts_propositions(fts_propositions, rowid, statement)
        VALUES ('delete', old.id, old.statement);
    END""",
    """CREATE TRIGGER IF NOT EXISTS propositions_au AFTER UPDATE ON propositions BEGIN
      INSERT INTO fts_propositions(fts_propositions, rowid, statement)
        VALUES ('delete', old.id, old.statement);
      INSERT INTO fts_propositions(rowid, statement) VALUES (new.id, new.statement);
    END""",
]


def _migrate(conn: sqlite3.Connection, vec_dim: int = 384) -> None:
    for stmt in _DDL:
        conn.execute(stmt)
    # Vec tables are dimension-specific — only create if absent, record dim in db_meta.
    vec_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
    ).fetchone()
    if not vec_exists:
        conn.execute(
            f"CREATE VIRTUAL TABLE vec_chunks USING vec0("
            f"chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{vec_dim}])"
        )
        conn.execute(
            f"CREATE VIRTUAL TABLE vec_entities USING vec0("
            f"entity_id INTEGER PRIMARY KEY, embedding FLOAT[{vec_dim}])"
        )
        conn.execute(
            "INSERT OR REPLACE INTO db_meta(key,value) VALUES ('vec_dim',?)",
            [str(vec_dim)],
        )
    # vec_propositions: created lazily alongside vec_facts
    vec_props_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_propositions'"
    ).fetchone()
    if not vec_props_exists:
        conn.execute(
            f"CREATE VIRTUAL TABLE vec_propositions USING vec0("
            f"prop_id INTEGER PRIMARY KEY, embedding FLOAT[{vec_dim}])"
        )

    # vec_facts: created lazily (may not exist in older DBs)
    vec_facts_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_facts'"
    ).fetchone()
    if not vec_facts_exists:
        conn.execute(
            f"CREATE VIRTUAL TABLE vec_facts USING vec0("
            f"fact_id INTEGER PRIMARY KEY, embedding FLOAT[{vec_dim}])"
        )
        fact_n = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        if fact_n > 0:
            import warnings
            warnings.warn(
                f"vec_facts created but not populated ({fact_n} facts exist). "
                "Run: loci facts reindex-vec"
            )
    # Schema evolution — add columns that were not in the original DDL
    _ensure_column(conn, "chunks", "extracted_v", "INTEGER DEFAULT 0")
    _ensure_column(conn, "facts", "source", "TEXT")
    conn.commit()

    # Backfill source column from confidence values (one-time migration).
    source_v = conn.execute(
        "SELECT value FROM db_meta WHERE key='fact_source_v'"
    ).fetchone()
    if source_v is None:
        conn.execute("UPDATE facts SET source='svo'   WHERE source IS NULL AND confidence=1.0")
        conn.execute("UPDATE facts SET source='coref' WHERE source IS NULL AND confidence=0.6")
        conn.execute("UPDATE facts SET source='llm'   WHERE source IS NULL AND confidence=0.7")
        conn.execute("INSERT OR REPLACE INTO db_meta(key,value) VALUES ('fact_source_v','1')")
        conn.commit()

    # Backfill/upgrade fact FTS when DB is missing or on an older version.
    fts_v = conn.execute(
        "SELECT value FROM db_meta WHERE key='fact_fts_v'"
    ).fetchone()
    fact_n = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    if (fts_v is None or fts_v[0] != _FACT_FTS_VERSION) and fact_n > 0:
        rebuild_fact_fts(conn)
        conn.execute(
            "INSERT OR REPLACE INTO db_meta(key,value) VALUES ('fact_fts_v',?)",
            [_FACT_FTS_VERSION],
        )
        conn.commit()

    # Build/upgrade the llm-only FTS index.
    fts_llm_v = conn.execute(
        "SELECT value FROM db_meta WHERE key='fact_fts_llm_v'"
    ).fetchone()
    if (fts_llm_v is None or fts_llm_v[0] != _FACT_FTS_LLM_VERSION) and fact_n > 0:
        rebuild_fact_fts_llm(conn)
        conn.execute(
            "INSERT OR REPLACE INTO db_meta(key,value) VALUES ('fact_fts_llm_v',?)",
            [_FACT_FTS_LLM_VERSION],
        )
        conn.commit()


def _ensure_column(conn: sqlite3.Connection, table: str, col: str, col_def: str) -> None:
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if col not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _ser(embedding: list[float]) -> bytes:
    return sqlite_vec.serialize_float32(embedding)


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def insert_source(
    conn: sqlite3.Connection,
    *,
    sha256: str,
    path: str | None = None,
    title: str | None = None,
    author: str | None = None,
    meta: dict | None = None,
) -> int | None:
    """Insert source row; returns new id, or None if sha256 already present."""
    if conn.execute("SELECT id FROM sources WHERE sha256=?", [sha256]).fetchone():
        return None
    cur = conn.execute(
        "INSERT INTO sources (path, title, author, meta, sha256) VALUES (?,?,?,?,?)",
        [path, title, author, json.dumps(meta) if meta else None, sha256],
    )
    conn.commit()
    return cur.lastrowid


def insert_chunk(
    conn: sqlite3.Connection,
    *,
    source_id: int,
    ordinal: int,
    text: str,
    sha256: str,
) -> int | None:
    """Insert chunk row; returns new id, or None if sha256 already present."""
    if conn.execute("SELECT id FROM chunks WHERE sha256=?", [sha256]).fetchone():
        return None
    cur = conn.execute(
        "INSERT INTO chunks (source_id, ordinal, text, sha256) VALUES (?,?,?,?)",
        [source_id, ordinal, text, sha256],
    )
    conn.commit()
    return cur.lastrowid


def insert_entity(
    conn: sqlite3.Connection,
    *,
    canonical_name: str,
    kind: str = "unknown",
) -> int:
    cur = conn.execute(
        "INSERT INTO entities (canonical_name, kind) VALUES (?,?)",
        [canonical_name, kind],
    )
    conn.commit()
    return cur.lastrowid


def insert_alias(conn: sqlite3.Connection, *, entity_id: int, alias: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO aliases (entity_id, alias) VALUES (?,?)",
        [entity_id, alias.lower()],
    )
    conn.commit()


def insert_fact(
    conn: sqlite3.Connection,
    *,
    chunk_id: int,
    sentence: str,
    subject_id: int,
    predicate: str,
    object_id: int | None = None,
    object_text: str | None = None,
    qualifiers: dict | None = None,
    negated: bool = False,
    confidence: float = 1.0,
    source: str = "svo",
) -> int | None:
    """Insert fact; returns new id, or None if it duplicates an existing fact."""
    q_json = json.dumps(qualifiers, sort_keys=True) if qualifiers else None
    existing = conn.execute(
        """SELECT id FROM facts
           WHERE chunk_id=? AND subject_id=? AND predicate=?
             AND object_text IS ? AND qualifiers IS ?""",
        [chunk_id, subject_id, predicate, object_text, q_json],
    ).fetchone()
    if existing:
        return None
    cur = conn.execute(
        """INSERT INTO facts
             (chunk_id, sentence, subject_id, predicate, object_id,
              object_text, qualifiers, negated, confidence, source)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        [chunk_id, sentence, subject_id, predicate, object_id,
         object_text, q_json, int(negated), confidence, source],
    )
    conn.commit()
    return cur.lastrowid


def upsert_vec_chunk(
    conn: sqlite3.Connection, *, chunk_id: int, embedding: list[float]
) -> None:
    # vec0 doesn't support INSERT OR REPLACE; delete-then-insert is the workaround
    conn.execute("DELETE FROM vec_chunks WHERE chunk_id=?", [chunk_id])
    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?,?)",
        [chunk_id, _ser(embedding)],
    )
    conn.commit()


def upsert_vec_entity(
    conn: sqlite3.Connection, *, entity_id: int, embedding: list[float]
) -> None:
    conn.execute("DELETE FROM vec_entities WHERE entity_id=?", [entity_id])
    conn.execute(
        "INSERT INTO vec_entities (entity_id, embedding) VALUES (?,?)",
        [entity_id, _ser(embedding)],
    )
    conn.commit()


def vec_search_chunks(
    conn: sqlite3.Connection, *, embedding: list[float], k: int = 12, schema: str = "main"
) -> list[dict]:
    sp = f"{schema}." if schema != "main" else ""
    rows = conn.execute(
        f"SELECT chunk_id, distance FROM {sp}vec_chunks"
        f" WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        [_ser(embedding), k],
    ).fetchall()
    return [dict(r) for r in rows]


def vec_search_entities(
    conn: sqlite3.Connection, *, embedding: list[float], k: int = 5
) -> list[dict]:
    rows = conn.execute(
        "SELECT entity_id, distance FROM vec_entities"
        " WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        [_ser(embedding), k],
    ).fetchall()
    return [dict(r) for r in rows]


def fts_search_chunks(
    conn: sqlite3.Connection, *, query: str, k: int = 12, schema: str = "main"
) -> list[dict]:
    if schema == "main":
        rows = conn.execute(
            "SELECT rowid AS chunk_id, rank FROM fts_chunks"
            " WHERE text MATCH ? ORDER BY rank LIMIT ?",
            [query, k],
        ).fetchall()
    else:
        sp = f"{schema}."
        rows = conn.execute(
            f"SELECT fts.rowid AS chunk_id, fts.rank"
            f" FROM {sp}fts_chunks fts"
            f" WHERE fts MATCH ? ORDER BY fts.rank LIMIT ?",
            [query, k],
        ).fetchall()
    return [dict(r) for r in rows]


_FACT_FTS_VERSION = "1"
_FACT_FTS_LLM_VERSION = "2"
_FACT_VEC_VERSION = "1"

# Question-mapped predicates only — keeps fts_facts_llm high-precision.
# Excludes generic copulas/verbs (be, call, say, have, take, …) that distract
# the model as [F#] injections without answering questions.
_ANSWER_SHAPED_PREDICATES: frozenset[str] = frozenset([
    "profession", "occupation", "role", "identity", "alias_of", "title",
    "means", "mean",
    "located_at", "resides_at", "reside_at",
    "relationship_to", "affiliation", "leader_of",
    "introduce", "murder", "cause_of", "named_after",
    "work_as",
])


def rebuild_fact_fts(conn: sqlite3.Connection) -> int:
    """(Re)build the fts_facts index from the current facts table.

    Indexed document per fact = subject name + predicate + object + source sentence.
    Returns number of facts indexed.
    """
    conn.execute("DELETE FROM fts_facts")
    rows = conn.execute(
        """
        SELECT f.id AS fid,
               e.canonical_name AS subj,
               f.predicate AS pred,
               COALESCE(oe.canonical_name, f.object_text, '') AS obj,
               f.sentence AS sent
        FROM facts f
        JOIN entities e  ON f.subject_id = e.id
        LEFT JOIN entities oe ON f.object_id = oe.id
        """
    ).fetchall()
    n = 0
    for r in rows:
        pred = (r["pred"] or "").replace("_", " ")
        doc = f"{r['subj']} {pred} {r['obj']} {r['sent']}"
        conn.execute("INSERT INTO fts_facts(rowid, text) VALUES (?, ?)",
                     [r["fid"], doc])
        n += 1
    conn.commit()
    return n


def rebuild_fact_fts_llm(conn: sqlite3.Connection) -> int:
    """(Re)build fts_facts_llm — indexes only llm and closure facts.

    Same document format as rebuild_fact_fts. Used for high-precision retrieval
    when fact_sources=minted so SVO noise doesn't dilute rankings.
    Returns number of facts indexed.
    """
    conn.execute("DELETE FROM fts_facts_llm")
    placeholders = ",".join("?" * len(_ANSWER_SHAPED_PREDICATES))
    rows = conn.execute(
        f"""
        SELECT f.id AS fid,
               e.canonical_name AS subj,
               f.predicate AS pred,
               COALESCE(oe.canonical_name, f.object_text, '') AS obj,
               f.sentence AS sent
        FROM facts f
        JOIN entities e  ON f.subject_id = e.id
        LEFT JOIN entities oe ON f.object_id = oe.id
        WHERE f.source IN ('llm', 'closure')
          AND f.predicate IN ({placeholders})
        """,
        list(_ANSWER_SHAPED_PREDICATES),
    ).fetchall()
    n = 0
    for r in rows:
        pred = (r["pred"] or "").replace("_", " ")
        doc = f"{r['subj']} {pred} {r['obj']} {r['sent']}"
        conn.execute("INSERT INTO fts_facts_llm(rowid, text) VALUES (?, ?)",
                     [r["fid"], doc])
        n += 1
    conn.commit()
    return n


def fts_search_facts(
    conn: sqlite3.Connection, *, query: str, k: int = 10, schema: str = "main"
) -> list[dict]:
    if schema == "main":
        rows = conn.execute(
            "SELECT rowid AS fact_id, rank FROM fts_facts"
            " WHERE text MATCH ? ORDER BY rank LIMIT ?",
            [query, k],
        ).fetchall()
    else:
        sp = f"{schema}."
        rows = conn.execute(
            f"SELECT fts.rowid AS fact_id, fts.rank"
            f" FROM {sp}fts_facts fts WHERE fts MATCH ? ORDER BY fts.rank LIMIT ?",
            [query, k],
        ).fetchall()
    return [dict(r) for r in rows]


def rebuild_fact_vec(conn: sqlite3.Connection, embedder: object) -> int:
    """(Re)build vec_facts by embedding each fact's document string.

    Doc text = same string rebuild_fact_fts builds: '{subj} {pred} {obj} {sentence}'.
    Returns number of facts embedded.
    """
    from loci.models import embed_batch
    rows = conn.execute(
        """SELECT f.id AS fid,
                  e.canonical_name AS subj,
                  f.predicate AS pred,
                  COALESCE(oe.canonical_name, f.object_text, '') AS obj,
                  f.sentence AS sent
           FROM facts f
           JOIN entities e ON f.subject_id = e.id
           LEFT JOIN entities oe ON f.object_id = oe.id"""
    ).fetchall()
    if not rows:
        return 0
    docs = [
        f"{r['subj']} {(r['pred'] or '').replace('_', ' ')} {r['obj']} {r['sent']}"
        for r in rows
    ]
    conn.execute("DELETE FROM vec_facts")
    embs = embed_batch(embedder, docs, normalize=True)
    for r, emb in zip(rows, embs):
        conn.execute(
            "INSERT INTO vec_facts(fact_id, embedding) VALUES (?,?)",
            [r["fid"], _ser(emb)],
        )
    conn.commit()
    return len(rows)


def vec_search_facts(
    conn: sqlite3.Connection, *, embedding: list[float], k: int = 10, schema: str = "main"
) -> list[dict]:
    """Return top-k facts by vector similarity. Returns [{fact_id, distance}]."""
    sp = f"{schema}." if schema != "main" else ""
    try:
        rows = conn.execute(
            f"SELECT fact_id, distance FROM {sp}vec_facts"
            f" WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            [_ser(embedding), k],
        ).fetchall()
    except Exception:
        return []
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Phase 5: stats, entity merge, pending links, maintenance
# ---------------------------------------------------------------------------

def get_stats(conn: sqlite3.Connection) -> dict:
    """Return knowledge-base statistics: counts, db size, most-connected entities."""
    stats: dict = {}
    for table in ("sources", "chunks", "entities", "aliases", "facts"):
        stats[f"{table}_count"] = conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

    page_count = conn.execute("PRAGMA main.page_count").fetchone()[0]
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    stats["db_size_bytes"] = page_count * page_size
    stats["db_size_mb"] = round(stats["db_size_bytes"] / 1024 / 1024, 3)

    rows = conn.execute(
        """
        SELECT e.canonical_name, e.kind, COUNT(f.id) AS fact_count
        FROM entities e
        LEFT JOIN facts f ON f.subject_id = e.id
        GROUP BY e.id
        ORDER BY fact_count DESC
        LIMIT 10
        """
    ).fetchall()
    stats["top_entities"] = [
        {"name": r["canonical_name"], "kind": r["kind"], "facts": r["fact_count"]}
        for r in rows
    ]
    return stats


def merge_entities(
    conn: sqlite3.Connection, keep_id: int, merge_id: int
) -> int:
    """Rewrite all facts and aliases from merge_id onto keep_id; delete merge_id.

    Returns number of facts whose subject_id or object_id was updated.
    """
    conn.execute(
        "UPDATE facts SET subject_id=? WHERE subject_id=?", [keep_id, merge_id]
    )
    conn.execute(
        "UPDATE facts SET object_id=? WHERE object_id=?", [keep_id, merge_id]
    )
    # Move aliases, skipping ones already present on keep_id
    for row in conn.execute(
        "SELECT alias FROM aliases WHERE entity_id=?", [merge_id]
    ).fetchall():
        conn.execute(
            "INSERT OR IGNORE INTO aliases (entity_id, alias) VALUES (?,?)",
            [keep_id, row["alias"]],
        )
    conn.execute("DELETE FROM aliases WHERE entity_id=?", [merge_id])
    conn.execute("DELETE FROM vec_entities WHERE entity_id=?", [merge_id])

    # Clear pending_links that mention either entity
    import json as _json
    for row in conn.execute("SELECT id, candidate_entity_ids FROM pending_links").fetchall():
        ids = _json.loads(row["candidate_entity_ids"])
        if keep_id in ids or merge_id in ids:
            conn.execute("DELETE FROM pending_links WHERE id=?", [row["id"]])

    n_affected = conn.execute(
        "SELECT COUNT(*) FROM facts WHERE subject_id=? OR object_id=?",
        [keep_id, keep_id],
    ).fetchone()[0]
    conn.execute("DELETE FROM entities WHERE id=?", [merge_id])
    conn.commit()
    return n_affected


def get_pending_links(conn: sqlite3.Connection) -> list[dict]:
    """Return all pending entity-resolution links with candidate details."""
    import json as _json
    result = []
    for row in conn.execute(
        "SELECT id, mention, candidate_entity_ids FROM pending_links"
    ).fetchall():
        candidate_ids = _json.loads(row["candidate_entity_ids"])
        candidates = []
        for eid in candidate_ids:
            e = conn.execute(
                "SELECT id, canonical_name, kind FROM entities WHERE id=?", [eid]
            ).fetchone()
            if e:
                candidates.append(
                    {"id": e["id"], "name": e["canonical_name"], "kind": e["kind"]}
                )
        result.append(
            {"id": row["id"], "mention": row["mention"], "candidates": candidates}
        )
    return result


def dismiss_pending_link(conn: sqlite3.Connection, link_id: int) -> None:
    """Dismiss a pending link without merging."""
    conn.execute("DELETE FROM pending_links WHERE id=?", [link_id])
    conn.commit()


def db_vacuum(conn: sqlite3.Connection) -> None:
    """Reclaim unused space (runs VACUUM on main db)."""
    conn.execute("VACUUM")


def db_analyze(conn: sqlite3.Connection) -> None:
    """Update query-planner statistics (runs ANALYZE)."""
    conn.execute("ANALYZE")


def get_unextracted_chunks(
    conn: sqlite3.Connection, limit: int | None = None
) -> list[dict]:
    """Return chunks whose extracted_v = 0 (never processed by LLM extraction)."""
    q = "SELECT id, text FROM chunks WHERE extracted_v = 0 ORDER BY id"
    if limit is not None:
        q += f" LIMIT {limit}"
    return [dict(r) for r in conn.execute(q).fetchall()]


def mark_chunk_extracted(
    conn: sqlite3.Connection, chunk_id: int, version: int = 1
) -> None:
    """Set extracted_v on a chunk to record that LLM extraction has run."""
    conn.execute(
        "UPDATE chunks SET extracted_v=? WHERE id=?", [version, chunk_id]
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Proposition layer CRUD
# ---------------------------------------------------------------------------

def ensure_prop_entity(
    conn: sqlite3.Connection,
    *,
    canonical: str,
    kind: str = "PERSON",
    aliases: list[str] | None = None,
) -> int:
    """Find or create a prop_entity by canonical name; register all aliases.

    Uses its own prop_entity_aliases table so there's no collision with the
    legacy aliases table (which may contain FIX1 garbage multi-name blobs).
    """
    from loci.resolve import normalize_mention
    row = conn.execute(
        "SELECT id FROM prop_entities WHERE canonical=?", [canonical]
    ).fetchone()
    if row:
        eid = row["id"]
    else:
        cur = conn.execute(
            "INSERT INTO prop_entities (canonical, kind) VALUES (?,?)",
            [canonical, kind],
        )
        eid = cur.lastrowid
    if aliases:
        for raw in aliases:
            for a in {raw.lower(), normalize_mention(raw)}:
                if a:
                    conn.execute(
                        "INSERT OR IGNORE INTO prop_entity_aliases"
                        " (prop_entity_id, alias) VALUES (?,?)",
                        [eid, a],
                    )
    conn.commit()
    return eid


def insert_proposition(
    conn: sqlite3.Connection,
    *,
    chunk_id: int,
    predicate: str,
    roles: dict,
    statement: str,
    prop_ref: str | None = None,
    polarity: str = "positive",
    salience: str = "medium",
    confidence: float = 1.0,
    evidence: str | None = None,
    char_span: list | None = None,
) -> int | None:
    """Insert a proposition row; returns new id, or None if already stored."""
    existing = conn.execute(
        "SELECT id FROM propositions WHERE chunk_id=? AND predicate=? AND statement=?",
        [chunk_id, predicate, statement],
    ).fetchone()
    if existing:
        return None
    cur = conn.execute(
        """INSERT INTO propositions
             (prop_ref, chunk_id, predicate, roles, polarity, salience, confidence,
              statement, evidence, char_span)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        [prop_ref, chunk_id, predicate, json.dumps(roles), polarity, salience,
         confidence, statement, evidence,
         json.dumps(char_span) if char_span else None],
    )
    conn.commit()
    return cur.lastrowid


def insert_proposition_entity(
    conn: sqlite3.Connection,
    *,
    prop_id: int,
    prop_entity_id: int,
    role: str,
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO proposition_entities"
        " (prop_id, prop_entity_id, role) VALUES (?,?,?)",
        [prop_id, prop_entity_id, role],
    )
    conn.commit()


def upsert_vec_proposition(
    conn: sqlite3.Connection, *, prop_id: int, embedding: list[float]
) -> None:
    conn.execute("DELETE FROM vec_propositions WHERE prop_id=?", [prop_id])
    conn.execute(
        "INSERT INTO vec_propositions (prop_id, embedding) VALUES (?,?)",
        [prop_id, _ser(embedding)],
    )
    conn.commit()


def sync_prop_entity_aliases_from_facts(conn: sqlite3.Connection) -> dict:
    """Import rich aliases from the facts entity system into prop_entity_aliases.

    For each prop_entity whose canonical name matches a canonical_name in the
    entities table, pull all aliases from the aliases table and register them
    in prop_entity_aliases.  Safe to run multiple times (INSERT OR IGNORE).

    Also removes junk prop_entities: pronouns, short tokens, and multi-word
    phrases that don't look like proper names (no title-case word).
    """
    from loci.resolve import normalize_mention

    _PRONOUNS = frozenset([
        "i", "he", "she", "it", "we", "they", "him", "her", "us", "them",
        "his", "its", "our", "their", "you", "your", "me",
    ])

    # 1. Remove junk prop_entities (pronouns, length < 3, no uppercase letter)
    prop_ents = conn.execute("SELECT id, canonical FROM prop_entities").fetchall()
    removed = 0
    for row in prop_ents:
        eid, canonical = row[0], row[1]
        tokens = canonical.strip().split()
        # Junk: single-token pronoun or lowercase, or length < 3, or no uppercase anywhere
        if (
            (len(tokens) == 1 and tokens[0].lower() in _PRONOUNS)
            or len(canonical.strip()) < 3
            or canonical == canonical.lower()  # all lowercase → likely not a proper noun
        ):
            # Cascade delete: proposition_entities + prop_entity_aliases, then entity
            conn.execute("DELETE FROM proposition_entities WHERE prop_entity_id=?", [eid])
            conn.execute("DELETE FROM prop_entity_aliases WHERE prop_entity_id=?", [eid])
            conn.execute("DELETE FROM prop_entities WHERE id=?", [eid])
            removed += 1

    # 2. For remaining prop_entities, pull fact-system aliases
    prop_ents = conn.execute("SELECT id, canonical FROM prop_entities").fetchall()
    aliases_added = 0
    for row in prop_ents:
        eid, canonical = row[0], row[1]
        fact_rows = conn.execute(
            """SELECT a.alias FROM aliases a
               JOIN entities e ON e.id = a.entity_id
               WHERE LOWER(e.canonical_name) = LOWER(?)""",
            [canonical],
        ).fetchall()
        for fr in fact_rows:
            raw = fr[0]
            for alias in {raw.lower(), normalize_mention(raw)}:
                if alias and len(alias) > 1:
                    cur = conn.execute(
                        "INSERT OR IGNORE INTO prop_entity_aliases"
                        " (prop_entity_id, alias) VALUES (?,?)",
                        [eid, alias],
                    )
                    aliases_added += cur.rowcount

    conn.commit()
    return {"prop_entities_removed": removed, "aliases_added": aliases_added}


def fts_search_propositions(
    conn: sqlite3.Connection, *, query: str, k: int = 10
) -> list[dict]:
    try:
        rows = conn.execute(
            "SELECT rowid AS prop_id, rank FROM fts_propositions"
            " WHERE statement MATCH ? ORDER BY rank LIMIT ?",
            [query, k],
        ).fetchall()
    except Exception:
        return []
    return [dict(r) for r in rows]


def get_props_for_entities_and_predicate(
    conn: sqlite3.Connection,
    *,
    predicate: str,
    prop_entity_ids: list[int],
) -> list[dict]:
    """Find propositions whose predicate matches and that contain ALL given entities."""
    if not prop_entity_ids or not predicate:
        return []
    n = len(prop_entity_ids)
    id_ph = ",".join("?" * n)
    rows = conn.execute(
        f"""SELECT p.id, p.predicate, p.statement, p.roles, p.chunk_id, p.prop_ref
            FROM propositions p
            WHERE p.predicate = ?
              AND p.id IN (
                SELECT prop_id FROM proposition_entities
                WHERE prop_entity_id IN ({id_ph})
                GROUP BY prop_id
                HAVING COUNT(DISTINCT prop_entity_id) >= ?
              )
            ORDER BY
              (SELECT 1 FROM proposition_entities
               WHERE prop_id=p.id AND prop_entity_id IN ({id_ph}) AND role='agent'
               LIMIT 1) DESC,
              p.confidence DESC,
              p.id ASC""",
        [predicate] + prop_entity_ids + [n] + prop_entity_ids,
    ).fetchall()
    return [dict(r) for r in rows]


def get_proposition(conn: sqlite3.Connection, prop_id: int) -> dict | None:
    row = conn.execute(
        """SELECT id, prop_ref, chunk_id, predicate, roles, polarity,
                  salience, confidence, statement, evidence, char_span
           FROM propositions WHERE id=?""",
        [prop_id],
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    if d.get("roles"):
        d["roles"] = json.loads(d["roles"])
    return d


def get_prop_agent_canonical(conn: sqlite3.Connection, prop_id: int) -> str | None:
    """Return the canonical name of the agent entity for a proposition."""
    row = conn.execute(
        """SELECT pe.canonical
           FROM proposition_entities pie
           JOIN prop_entities pe ON pie.prop_entity_id = pe.id
           WHERE pie.prop_id=? AND pie.role='agent'""",
        [prop_id],
    ).fetchone()
    return row["canonical"] if row else None

"""SQLite store: schema creation, migration, CRUD helpers, and attach logic."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import sqlite_vec


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def open_db(db_path: str | Path) -> sqlite3.Connection:
    """Open (or create) a knowledge DB, load sqlite-vec, and apply schema."""
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _load_extensions(conn)
    _configure(conn)
    _migrate(conn)
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
    """CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
      chunk_id  INTEGER PRIMARY KEY,
      embedding FLOAT[384]
    )""",
    """CREATE VIRTUAL TABLE IF NOT EXISTS vec_entities USING vec0(
      entity_id INTEGER PRIMARY KEY,
      embedding FLOAT[384]
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
]


def _migrate(conn: sqlite3.Connection) -> None:
    for stmt in _DDL:
        conn.execute(stmt)
    conn.commit()


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
              object_text, qualifiers, negated, confidence)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        [chunk_id, sentence, subject_id, predicate, object_id,
         object_text, q_json, int(negated), confidence],
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
    conn: sqlite3.Connection, *, embedding: list[float], k: int = 12
) -> list[dict]:
    rows = conn.execute(
        "SELECT chunk_id, distance FROM vec_chunks"
        " WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
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
    conn: sqlite3.Connection, *, query: str, k: int = 12
) -> list[dict]:
    rows = conn.execute(
        "SELECT rowid AS chunk_id, rank FROM fts_chunks"
        " WHERE text MATCH ? ORDER BY rank LIMIT ?",
        [query, k],
    ).fetchall()
    return [dict(r) for r in rows]

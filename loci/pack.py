"""Pack management: export, registry, attach, validate."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

_REGISTRY_FILE = "registry.json"


# ---------------------------------------------------------------------------
# Registry (JSON file in packs_dir)
# ---------------------------------------------------------------------------

def _registry_path(packs_dir: Path) -> Path:
    return packs_dir / _REGISTRY_FILE


def load_registry(packs_dir: Path) -> list[dict]:
    """Return registered packs as [{name, path}] dicts."""
    p = _registry_path(packs_dir)
    if not p.exists():
        return []
    with open(p) as fh:
        return json.load(fh)


def save_registry(packs_dir: Path, registry: list[dict]) -> None:
    packs_dir.mkdir(parents=True, exist_ok=True)
    with open(_registry_path(packs_dir), "w") as fh:
        json.dump(registry, fh, indent=2)


def register_pack(packs_dir: Path, pack_path: Path, name: str) -> dict:
    """Add pack to registry. Raises ValueError if name already exists."""
    registry = load_registry(packs_dir)
    if any(p["name"] == name for p in registry):
        raise ValueError(f"Pack '{name}' already registered.")
    entry = {"name": name, "path": str(pack_path.expanduser().resolve())}
    registry.append(entry)
    save_registry(packs_dir, registry)
    return entry


def unregister_pack(packs_dir: Path, name: str) -> None:
    """Remove pack by name. Raises ValueError if not found."""
    registry = load_registry(packs_dir)
    new_reg = [p for p in registry if p["name"] != name]
    if len(new_reg) == len(registry):
        raise ValueError(f"Pack '{name}' not found in registry.")
    save_registry(packs_dir, new_reg)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_pack(pack_path: Path) -> None:
    """Raise if pack_path is not a valid loci pack."""
    if not pack_path.exists():
        raise FileNotFoundError(f"Pack not found: {pack_path}")
    conn = sqlite3.connect(str(pack_path))
    try:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','shadow')"
        ).fetchall()}
        required = {"sources", "chunks", "entities", "aliases", "facts"}
        missing = required - tables
        if missing:
            raise ValueError(f"Pack missing required tables: {', '.join(sorted(missing))}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_pack(
    main_conn: sqlite3.Connection,
    out_path: Path,
    name: str,
) -> dict[str, int]:
    """Copy all knowledge data from main_conn into a new .locipack.db at out_path.

    Returns row counts per table.
    """
    from loci.store import open_db

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    pack_conn = open_db(out_path)
    counts: dict[str, int] = {}
    try:
        for table in ("sources", "chunks", "entities", "aliases",
                      "facts", "predicate_synonyms", "pending_links"):
            rows = main_conn.execute(f"SELECT * FROM {table}").fetchall()
            if rows:
                desc = main_conn.execute(f"SELECT * FROM {table} LIMIT 0").description
                cols = [d[0] for d in desc]
                ph = ",".join("?" * len(cols))
                pack_conn.executemany(
                    f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) VALUES ({ph})",
                    [tuple(r) for r in rows],
                )
            counts[table] = len(rows)

        # Copy vector embeddings for chunks
        try:
            vec_rows = main_conn.execute(
                "SELECT chunk_id, embedding FROM vec_chunks"
            ).fetchall()
            for r in vec_rows:
                pack_conn.execute(
                    "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?,?)",
                    (r[0], r[1]),
                )
            counts["vec_chunks"] = len(vec_rows)
        except Exception:
            counts["vec_chunks"] = 0

        # Copy entity embeddings
        try:
            ve_rows = main_conn.execute(
                "SELECT entity_id, embedding FROM vec_entities"
            ).fetchall()
            for r in ve_rows:
                pack_conn.execute(
                    "INSERT INTO vec_entities (entity_id, embedding) VALUES (?,?)",
                    (r[0], r[1]),
                )
            counts["vec_entities"] = len(ve_rows)
        except Exception:
            counts["vec_entities"] = 0

        # Rebuild FTS index from chunks content table
        try:
            pack_conn.execute(
                "INSERT INTO fts_chunks(fts_chunks) VALUES('rebuild')"
            )
            counts["fts_chunks"] = counts.get("chunks", 0)
        except Exception:
            counts["fts_chunks"] = 0

        pack_conn.commit()
    finally:
        pack_conn.close()

    return counts


# ---------------------------------------------------------------------------
# Attach helpers
# ---------------------------------------------------------------------------

def attach_registered_packs(conn: sqlite3.Connection, packs_dir: Path) -> list[str]:
    """ATTACH all registered packs to conn. Returns list of schema names used."""
    from loci.store import attach_pack
    registry = load_registry(packs_dir)
    schemas: list[str] = []
    for i, entry in enumerate(registry):
        path = Path(entry["path"])
        if not path.exists():
            continue
        schema = f"pack_{i}"
        attach_pack(conn, path, schema=schema)
        schemas.append(schema)
    return schemas


def pack_schemas(conn: sqlite3.Connection) -> list[str]:
    """Return names of all attached pack schemas (excludes 'main' and 'temp')."""
    rows = conn.execute("PRAGMA database_list").fetchall()
    return [r[1] for r in rows if r[1] not in ("main", "temp")]

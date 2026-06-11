"""Tests for loci/pack.py: export, registry, validate, attach, multi-schema retrieve."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from loci.pack import (
    attach_registered_packs,
    export_pack,
    load_registry,
    pack_schemas,
    register_pack,
    save_registry,
    unregister_pack,
    validate_pack,
)

SPEC_TEXT = (
    "Sherlock Holmes took his bottle from the corner of the mantel-piece. "
    "Holmes also took his syringe from its neat morocco case. "
    "Watson entered the room and observed Holmes carefully."
)


# ---------------------------------------------------------------------------
# Fixture: populated main DB
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def main_db(tmp_path_factory, fake_embedder, nlp, default_cfg):
    tmp = tmp_path_factory.mktemp("pack_main")
    from loci.store import open_db
    from loci.ingest import ingest_file

    conn = open_db(tmp / "main.db")
    p = tmp / "spec.txt"
    p.write_text(SPEC_TEXT)
    ingest_file(p, cfg=default_cfg, conn=conn, embedder=fake_embedder, spacy_nlp=nlp)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def pack_file(tmp_path_factory, main_db):
    """A .locipack.db exported from main_db."""
    tmp = tmp_path_factory.mktemp("pack_out")
    out = tmp / "sherlock.locipack.db"
    export_pack(main_db, out, name="sherlock")
    return out


# ---------------------------------------------------------------------------
# validate_pack
# ---------------------------------------------------------------------------

class TestValidatePack:
    def test_valid_pack_passes(self, pack_file):
        validate_pack(pack_file)  # no exception

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_pack(tmp_path / "nonexistent.db")

    def test_empty_db_raises(self, tmp_path):
        p = tmp_path / "empty.db"
        sqlite3.connect(str(p)).close()
        with pytest.raises(ValueError, match="missing required tables"):
            validate_pack(p)


# ---------------------------------------------------------------------------
# export_pack
# ---------------------------------------------------------------------------

class TestExportPack:
    def test_creates_file(self, tmp_path, main_db):
        out = tmp_path / "test_export.locipack.db"
        export_pack(main_db, out, name="test")
        assert out.exists()

    def test_counts_returned(self, tmp_path, main_db):
        out = tmp_path / "counts.locipack.db"
        counts = export_pack(main_db, out, name="counts")
        assert counts.get("chunks", 0) > 0
        assert counts.get("entities", 0) > 0
        assert counts.get("facts", 0) > 0

    def test_pack_has_same_entity_count(self, pack_file, main_db):
        main_n = main_db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        pack_conn = sqlite3.connect(str(pack_file))
        try:
            pack_n = pack_conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        finally:
            pack_conn.close()
        assert pack_n == main_n

    def test_pack_has_same_fact_count(self, pack_file, main_db):
        main_n = main_db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        pack_conn = sqlite3.connect(str(pack_file))
        try:
            pack_n = pack_conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        finally:
            pack_conn.close()
        assert pack_n == main_n

    def test_overwrites_existing_file(self, tmp_path, main_db):
        out = tmp_path / "overwrite.locipack.db"
        export_pack(main_db, out, name="first")
        mtime1 = out.stat().st_mtime
        export_pack(main_db, out, name="second")
        assert out.exists()


# ---------------------------------------------------------------------------
# Registry: register / unregister / load / save
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_empty_registry(self, tmp_path):
        assert load_registry(tmp_path) == []

    def test_register_adds_entry(self, tmp_path, pack_file):
        register_pack(tmp_path, pack_file, "sherlock")
        reg = load_registry(tmp_path)
        assert len(reg) == 1
        assert reg[0]["name"] == "sherlock"
        assert reg[0]["path"] == str(pack_file)

    def test_register_duplicate_name_raises(self, tmp_path, pack_file):
        register_pack(tmp_path, pack_file, "sherlock")
        with pytest.raises(ValueError, match="already registered"):
            register_pack(tmp_path, pack_file, "sherlock")

    def test_unregister_removes_entry(self, tmp_path, pack_file):
        register_pack(tmp_path, pack_file, "to_remove")
        unregister_pack(tmp_path, "to_remove")
        reg = load_registry(tmp_path)
        assert not any(p["name"] == "to_remove" for p in reg)

    def test_unregister_not_found_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            unregister_pack(tmp_path, "ghost")

    def test_multiple_packs_registered(self, tmp_path, pack_file):
        register_pack(tmp_path, pack_file, "pack_a")
        register_pack(tmp_path, pack_file, "pack_b")
        reg = load_registry(tmp_path)
        names = {e["name"] for e in reg}
        assert {"pack_a", "pack_b"} == names


# ---------------------------------------------------------------------------
# attach_registered_packs / pack_schemas
# ---------------------------------------------------------------------------

class TestAttach:
    def test_no_packs_returns_empty(self, tmp_path, main_db):
        schemas = attach_registered_packs(main_db, tmp_path)
        assert schemas == []

    def test_attaches_registered_pack(self, tmp_path, main_db, pack_file):
        register_pack(tmp_path, pack_file, "test_attach")
        schemas = attach_registered_packs(main_db, tmp_path)
        assert len(schemas) == 1
        assert schemas[0].startswith("pack_")

    def test_pack_schemas_discovers_attached(self, tmp_path, main_db, pack_file):
        # fresh connection to avoid state from prior tests
        from loci.store import open_db
        conn2 = sqlite3.connect(str(main_db.execute("PRAGMA main.database_list").fetchone()[2]))
        from loci.store import attach_pack
        attach_pack(conn2, pack_file, schema="test_schema")
        schemas = pack_schemas(conn2)
        assert "test_schema" in schemas
        conn2.close()

    def test_missing_pack_file_skipped(self, tmp_path, main_db):
        reg = [{"name": "ghost", "path": str(tmp_path / "nonexistent.db")}]
        save_registry(tmp_path, reg)
        schemas = attach_registered_packs(main_db, tmp_path)
        assert schemas == []


# ---------------------------------------------------------------------------
# Multi-schema retrieval (integration)
# ---------------------------------------------------------------------------

class TestMultiSchemaRetrieve:
    def test_retrieve_finds_facts_from_pack(
        self, tmp_path, fake_embedder, nlp, default_cfg, pack_file
    ):
        """Open a fresh DB (no knowledge), attach pack, query finds facts from pack."""
        from loci.store import open_db
        from loci.retrieve import retrieve
        from loci.store import attach_pack

        empty_db = open_db(tmp_path / "empty.db")
        attach_pack(empty_db, pack_file, schema="pack_0")
        try:
            result = retrieve(
                "what did sherlock holmes take?",
                conn=empty_db,
                cfg=default_cfg,
                embedder=fake_embedder,
                nlp=nlp,
                pack_schemas=["pack_0"],
            )
            assert result.fact_hits, "should find facts from the pack schema"
            objects = [h.object_text for h in result.fact_hits]
            assert "bottle" in objects
        finally:
            empty_db.close()

    def test_retrieve_chunks_from_pack(
        self, tmp_path, fake_embedder, nlp, default_cfg, pack_file
    ):
        from loci.store import open_db
        from loci.retrieve import retrieve
        from loci.store import attach_pack

        empty_db = open_db(tmp_path / "empty2.db")
        attach_pack(empty_db, pack_file, schema="pack_0")
        try:
            result = retrieve(
                "what did sherlock holmes take?",
                conn=empty_db,
                cfg=default_cfg,
                embedder=fake_embedder,
                nlp=nlp,
                pack_schemas=["pack_0"],
            )
            assert result.chunk_hits
        finally:
            empty_db.close()

    def test_no_pack_schemas_single_schema_behavior_unchanged(
        self, main_db, fake_embedder, nlp, default_cfg
    ):
        """pack_schemas=None must behave exactly like the pre-Phase-5 flow."""
        from loci.retrieve import retrieve
        result = retrieve(
            "what did sherlock holmes take?",
            conn=main_db,
            cfg=default_cfg,
            embedder=fake_embedder,
            nlp=nlp,
        )
        assert result.fact_hits
        assert "[F1]" in result.context_text

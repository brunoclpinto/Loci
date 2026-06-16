"""Tests for loci/store.py: schema, dedup, FK, vec, FTS, attach."""
import hashlib
import sqlite3

import pytest

from loci.store import (
    attach_pack,
    fts_search_chunks,
    fts_search_facts,
    insert_alias,
    insert_chunk,
    insert_entity,
    insert_fact,
    insert_source,
    open_db,
    rebuild_fact_fts,
    rebuild_fact_fts_llm,
    rebuild_fact_vec,
    upsert_vec_chunk,
    upsert_vec_entity,
    vec_search_chunks,
    vec_search_entities,
    vec_search_facts,
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_regular_tables(self, tmp_db):
        names = {r[0] for r in tmp_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        for t in ("sources", "chunks", "entities", "aliases", "facts", "predicate_synonyms"):
            assert t in names, f"missing table: {t}"

    def test_virtual_tables(self, tmp_db):
        names = {r[0] for r in tmp_db.execute("SELECT name FROM sqlite_master")}
        assert "vec_chunks" in names
        assert "vec_entities" in names
        assert "fts_chunks" in names

    def test_fts_triggers_exist(self, tmp_db):
        names = {r[0] for r in tmp_db.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        )}
        assert "chunks_ai" in names
        assert "chunks_ad" in names
        assert "chunks_au" in names


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

class TestDedup:
    def test_source_dedup_returns_none(self, tmp_db):
        digest = sha("same_content")
        id1 = insert_source(tmp_db, sha256=digest, title="First")
        id2 = insert_source(tmp_db, sha256=digest, title="Duplicate")
        assert id1 is not None
        assert id2 is None

    def test_source_dedup_row_count(self, tmp_db):
        digest = sha("content_x")
        insert_source(tmp_db, sha256=digest)
        insert_source(tmp_db, sha256=digest)
        count = tmp_db.execute("SELECT COUNT(*) FROM sources WHERE sha256=?", [digest]).fetchone()[0]
        assert count == 1

    def test_chunk_dedup_returns_none(self, tmp_db):
        src = insert_source(tmp_db, sha256=sha("src"))
        digest = sha("chunk text")
        id1 = insert_chunk(tmp_db, source_id=src, ordinal=0, text="hello", sha256=digest)
        id2 = insert_chunk(tmp_db, source_id=src, ordinal=1, text="hello", sha256=digest)
        assert id1 is not None
        assert id2 is None

    def test_fact_dedup_returns_none(self, tmp_db):
        src = insert_source(tmp_db, sha256=sha("s"))
        chunk = insert_chunk(tmp_db, source_id=src, ordinal=0, text="x", sha256=sha("x"))
        ent = insert_entity(tmp_db, canonical_name="Holmes")
        id1 = insert_fact(tmp_db, chunk_id=chunk, sentence="s", subject_id=ent, predicate="take",
                          object_text="bottle")
        id2 = insert_fact(tmp_db, chunk_id=chunk, sentence="s", subject_id=ent, predicate="take",
                          object_text="bottle")
        assert id1 is not None
        assert id2 is None


# ---------------------------------------------------------------------------
# Foreign key enforcement
# ---------------------------------------------------------------------------

class TestForeignKeys:
    def test_chunk_requires_valid_source(self, tmp_db):
        with pytest.raises(sqlite3.IntegrityError):
            tmp_db.execute(
                "INSERT INTO chunks (source_id, ordinal, text, sha256) VALUES (9999,0,'x','abc')"
            )

    def test_alias_requires_valid_entity(self, tmp_db):
        with pytest.raises(sqlite3.IntegrityError):
            tmp_db.execute(
                "INSERT INTO aliases (entity_id, alias) VALUES (9999, 'ghost')"
            )

    def test_fact_requires_valid_chunk(self, tmp_db):
        ent = insert_entity(tmp_db, canonical_name="X")
        with pytest.raises(sqlite3.IntegrityError):
            tmp_db.execute(
                "INSERT INTO facts (chunk_id, sentence, subject_id, predicate)"
                " VALUES (9999, 'test', ?, 'do')",
                [ent],
            )


# ---------------------------------------------------------------------------
# Vector round-trip
# ---------------------------------------------------------------------------

class TestVec:
    def test_vec_chunk_round_trip(self, tmp_db, flat_embedding):
        src = insert_source(tmp_db, sha256=sha("vs"))
        cid = insert_chunk(tmp_db, source_id=src, ordinal=0, text="vec test", sha256=sha("vec test"))
        upsert_vec_chunk(tmp_db, chunk_id=cid, embedding=flat_embedding)

        results = vec_search_chunks(tmp_db, embedding=flat_embedding, k=1)
        assert len(results) == 1
        assert results[0]["chunk_id"] == cid

    def test_vec_entity_round_trip(self, tmp_db, flat_embedding):
        eid = insert_entity(tmp_db, canonical_name="Holmes")
        upsert_vec_entity(tmp_db, entity_id=eid, embedding=flat_embedding)

        results = vec_search_entities(tmp_db, embedding=flat_embedding, k=1)
        assert len(results) == 1
        assert results[0]["entity_id"] == eid

    def test_vec_distance_ordering(self, tmp_db):
        src = insert_source(tmp_db, sha256=sha("vd"))
        c1 = insert_chunk(tmp_db, source_id=src, ordinal=0, text="a", sha256=sha("a"))
        c2 = insert_chunk(tmp_db, source_id=src, ordinal=1, text="b", sha256=sha("b"))

        em1 = [1.0] + [0.0] * 383
        em2 = [0.0] + [1.0] + [0.0] * 382

        upsert_vec_chunk(tmp_db, chunk_id=c1, embedding=em1)
        upsert_vec_chunk(tmp_db, chunk_id=c2, embedding=em2)

        results = vec_search_chunks(tmp_db, embedding=em1, k=2)
        # c1 should be the closest match to em1
        assert results[0]["chunk_id"] == c1

    def test_vec_upsert_replaces(self, tmp_db, flat_embedding):
        src = insert_source(tmp_db, sha256=sha("vr"))
        cid = insert_chunk(tmp_db, source_id=src, ordinal=0, text="r", sha256=sha("r"))

        upsert_vec_chunk(tmp_db, chunk_id=cid, embedding=flat_embedding)
        new_em = [0.9] * 384
        upsert_vec_chunk(tmp_db, chunk_id=cid, embedding=new_em)  # replace

        count = tmp_db.execute(
            "SELECT COUNT(*) FROM vec_chunks WHERE chunk_id=?", [cid]
        ).fetchone()[0]
        assert count == 1


# ---------------------------------------------------------------------------
# FTS round-trip
# ---------------------------------------------------------------------------

class TestFTS:
    def test_fts_finds_inserted_chunk(self, tmp_db):
        src = insert_source(tmp_db, sha256=sha("fs"))
        cid = insert_chunk(
            tmp_db, source_id=src, ordinal=0,
            text="sherlock holmes mystery", sha256=sha("sherlock holmes mystery")
        )
        results = fts_search_chunks(tmp_db, query="sherlock", k=5)
        assert any(r["chunk_id"] == cid for r in results)

    def test_fts_no_match_returns_empty(self, tmp_db):
        src = insert_source(tmp_db, sha256=sha("fn"))
        insert_chunk(tmp_db, source_id=src, ordinal=0, text="unrelated text", sha256=sha("unrelated text"))
        results = fts_search_chunks(tmp_db, query="xyznonexistent123", k=5)
        assert results == []

    def test_fts_ranks_better_match_first(self, tmp_db):
        src = insert_source(tmp_db, sha256=sha("fr"))
        c1 = insert_chunk(
            tmp_db, source_id=src, ordinal=0,
            text="watson watson watson", sha256=sha("watson watson watson")
        )
        c2 = insert_chunk(
            tmp_db, source_id=src, ordinal=1,
            text="watson and holmes", sha256=sha("watson and holmes")
        )
        results = fts_search_chunks(tmp_db, query="watson", k=5)
        ids = [r["chunk_id"] for r in results]
        # c1 has higher term frequency for "watson" so should rank first
        assert ids[0] == c1


# ---------------------------------------------------------------------------
# ATTACH — cross-database queries
# ---------------------------------------------------------------------------

class TestAttach:
    def test_attach_union_query(self, tmp_path):
        main_conn = open_db(tmp_path / "main.db")
        src_m = insert_source(main_conn, sha256=sha("main_src"), title="Main")
        insert_chunk(main_conn, source_id=src_m, ordinal=0,
                     text="main chunk text", sha256=sha("main chunk text"))

        pack_conn = open_db(tmp_path / "pack.db")
        src_p = insert_source(pack_conn, sha256=sha("pack_src"), title="Pack")
        insert_chunk(pack_conn, source_id=src_p, ordinal=0,
                     text="pack chunk text", sha256=sha("pack chunk text"))
        pack_conn.close()

        attach_pack(main_conn, tmp_path / "pack.db", schema="pack")

        rows = main_conn.execute(
            "SELECT text FROM chunks UNION ALL SELECT text FROM [pack].chunks"
        ).fetchall()
        texts = {r[0] for r in rows}
        assert "main chunk text" in texts
        assert "pack chunk text" in texts

        main_conn.close()

    def test_attach_counts_both_dbs(self, tmp_path):
        main_conn = open_db(tmp_path / "m.db")
        src_m = insert_source(main_conn, sha256=sha("m"), title="M")
        for i in range(3):
            insert_chunk(main_conn, source_id=src_m, ordinal=i,
                         text=f"main {i}", sha256=sha(f"main {i}"))

        pack_conn = open_db(tmp_path / "p.db")
        src_p = insert_source(pack_conn, sha256=sha("p"), title="P")
        for i in range(2):
            insert_chunk(pack_conn, source_id=src_p, ordinal=i,
                         text=f"pack {i}", sha256=sha(f"pack {i}"))
        pack_conn.close()

        attach_pack(main_conn, tmp_path / "p.db", schema="pack")

        total = main_conn.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT id FROM chunks UNION ALL SELECT id FROM [pack].chunks"
            ")"
        ).fetchone()[0]
        assert total == 5

        main_conn.close()


# ---------------------------------------------------------------------------
# fts_facts: rebuild, search, idempotency, backfill
# ---------------------------------------------------------------------------

def _seed_landlady_fact(conn):
    """Insert entity+chunk+fact for 'Mrs Hudson — role — landlady'."""
    src_id = insert_source(conn, sha256=sha("src_landlady"), title="Test")
    chunk_id = insert_chunk(
        conn, source_id=src_id, ordinal=0,
        text="Mrs. Hudson, our landlady, brought tea.",
        sha256=sha("chunk_landlady"),
    )
    eid = insert_entity(conn, canonical_name="Mrs Hudson", kind="person")
    insert_alias(conn, entity_id=eid, alias="mrs hudson")
    fact_id = insert_fact(
        conn, chunk_id=chunk_id,
        sentence="Mrs. Hudson, our landlady, brought tea.",
        subject_id=eid, predicate="role", object_text="landlady",
    )
    return fact_id


class TestFtsFacts:
    def test_rebuild_count_matches_facts(self, tmp_db):
        _seed_landlady_fact(tmp_db)
        n = rebuild_fact_fts(tmp_db)
        db_n = tmp_db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        assert n == db_n
        assert tmp_db.execute("SELECT COUNT(*) FROM fts_facts").fetchone()[0] == db_n

    def test_search_finds_object_keyword(self, tmp_db):
        fact_id = _seed_landlady_fact(tmp_db)
        rebuild_fact_fts(tmp_db)
        results = fts_search_facts(tmp_db, query="landlady", k=5)
        ids = [r["fact_id"] for r in results]
        assert fact_id in ids

    def test_idempotent_rebuild(self, tmp_db):
        _seed_landlady_fact(tmp_db)
        n1 = rebuild_fact_fts(tmp_db)
        n2 = rebuild_fact_fts(tmp_db)
        assert n1 == n2
        assert tmp_db.execute("SELECT COUNT(*) FROM fts_facts").fetchone()[0] == n1

    def test_backfill_on_reopen(self, tmp_path):
        db_path = tmp_path / "backfill.db"
        conn = open_db(db_path)
        _seed_landlady_fact(conn)
        conn.close()

        conn2 = open_db(db_path)
        count = conn2.execute("SELECT COUNT(*) FROM fts_facts").fetchone()[0]
        meta = conn2.execute(
            "SELECT value FROM db_meta WHERE key='fact_fts_v'"
        ).fetchone()
        conn2.close()
        assert count > 0
        assert meta is not None and meta[0] == "1"


# ---------------------------------------------------------------------------
# vec_facts: Phase B — rebuild, search, idempotency
# ---------------------------------------------------------------------------

class TestVecFacts:
    def test_vec_facts_table_exists(self, tmp_db):
        names = {r[0] for r in tmp_db.execute("SELECT name FROM sqlite_master")}
        assert "vec_facts" in names

    def test_rebuild_fact_vec_returns_count(self, tmp_db, fake_embedder):
        _seed_landlady_fact(tmp_db)
        n = rebuild_fact_vec(tmp_db, fake_embedder)
        db_n = tmp_db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        assert n == db_n

    def test_vec_search_facts_returns_hit(self, tmp_db, fake_embedder):
        _seed_landlady_fact(tmp_db)
        rebuild_fact_vec(tmp_db, fake_embedder)
        from loci.models import embed_batch
        emb = embed_batch(fake_embedder, ["landlady role"], normalize=True)[0]
        results = vec_search_facts(tmp_db, embedding=emb, k=5)
        assert len(results) > 0
        assert "fact_id" in results[0]
        assert "distance" in results[0]

    def test_rebuild_fact_vec_idempotent(self, tmp_db, fake_embedder):
        _seed_landlady_fact(tmp_db)
        n1 = rebuild_fact_vec(tmp_db, fake_embedder)
        n2 = rebuild_fact_vec(tmp_db, fake_embedder)
        assert n1 == n2
        count = tmp_db.execute("SELECT COUNT(*) FROM vec_facts").fetchone()[0]
        assert count == n1

    def test_vec_search_facts_empty_when_not_populated(self, tmp_db, flat_embedding):
        results = vec_search_facts(tmp_db, embedding=flat_embedding, k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Phase Q — source provenance column + backfill
# ---------------------------------------------------------------------------

class TestFactSourceColumn:
    def test_source_column_exists(self, tmp_db):
        cols = {r[1] for r in tmp_db.execute("PRAGMA table_info(facts)").fetchall()}
        assert "source" in cols

    def test_insert_fact_stores_source(self, tmp_db):
        src_id = insert_source(tmp_db, sha256=sha("src_q"), title="Q")
        chunk_id = insert_chunk(tmp_db, source_id=src_id, ordinal=0,
                                text="text", sha256=sha("text_q"))
        eid = insert_entity(tmp_db, canonical_name="Holmes")
        fid = insert_fact(tmp_db, chunk_id=chunk_id, sentence="s",
                          subject_id=eid, predicate="role",
                          object_text="detective", source="llm")
        row = tmp_db.execute("SELECT source FROM facts WHERE id=?", [fid]).fetchone()
        assert row["source"] == "llm"

    def test_backfill_runs_on_existing_db(self, tmp_path):
        """Open a fresh DB that has no fact_source_v meta; backfill must fire."""
        conn = open_db(tmp_path / "backfill.db")
        src_id = insert_source(conn, sha256=sha("bf_src"), title="BF")
        chunk_id = insert_chunk(conn, source_id=src_id, ordinal=0,
                                text="t", sha256=sha("bf_t"))
        eid = insert_entity(conn, canonical_name="Watson")
        # Insert fact with confidence=0.7 (pre-source era: simulates llm fact)
        conn.execute(
            "INSERT INTO facts(chunk_id, sentence, subject_id, predicate, object_text, confidence)"
            " VALUES (?,?,?,?,?,?)",
            [chunk_id, "s", eid, "role", "doctor", 0.7],
        )
        conn.commit()
        # Remove the meta key so backfill triggers on next open
        conn.execute("DELETE FROM db_meta WHERE key='fact_source_v'")
        conn.commit()
        conn.close()

        conn2 = open_db(tmp_path / "backfill.db")
        row = conn2.execute(
            "SELECT source FROM facts WHERE predicate='role'"
        ).fetchone()
        conn2.close()
        assert row["source"] == "llm"

    def test_backfill_svo_confidence(self, tmp_path):
        conn = open_db(tmp_path / "backfill2.db")
        src_id = insert_source(conn, sha256=sha("svo_src"), title="SVO")
        chunk_id = insert_chunk(conn, source_id=src_id, ordinal=0,
                                text="t2", sha256=sha("svo_t"))
        eid = insert_entity(conn, canonical_name="Hope")
        conn.execute(
            "INSERT INTO facts(chunk_id, sentence, subject_id, predicate, object_text, confidence)"
            " VALUES (?,?,?,?,?,?)",
            [chunk_id, "s", eid, "drive", "cab", 1.0],
        )
        conn.commit()
        conn.execute("DELETE FROM db_meta WHERE key='fact_source_v'")
        conn.commit()
        conn.close()

        conn2 = open_db(tmp_path / "backfill2.db")
        row = conn2.execute(
            "SELECT source FROM facts WHERE predicate='drive'"
        ).fetchone()
        conn2.close()
        assert row["source"] == "svo"


# ---------------------------------------------------------------------------
# fts_facts_llm — llm+closure only index
# ---------------------------------------------------------------------------

def _seed_mixed_sources(conn):
    """Insert one svo fact and one llm fact into the same DB."""
    src_id = insert_source(conn, sha256=sha("mix_src"), title="Mix")
    chunk_id = insert_chunk(conn, source_id=src_id, ordinal=0,
                            text="Holmes took a cab.", sha256=sha("mix_chunk"))
    eid = insert_entity(conn, canonical_name="Holmes", kind="person")
    svo_id = insert_fact(conn, chunk_id=chunk_id,
                         sentence="Holmes took a cab.",
                         subject_id=eid, predicate="take",
                         object_text="cab", source="svo")
    llm_id = insert_fact(conn, chunk_id=chunk_id,
                         sentence="Holmes is a consulting detective.",
                         subject_id=eid, predicate="profession",
                         object_text="consulting detective", source="llm")
    return svo_id, llm_id, eid


class TestFtsFactsLLM:
    def test_table_exists(self, tmp_db):
        names = {r[0] for r in tmp_db.execute("SELECT name FROM sqlite_master")}
        assert "fts_facts_llm" in names

    def test_rebuild_excludes_svo(self, tmp_db):
        svo_id, llm_id, _ = _seed_mixed_sources(tmp_db)
        n = rebuild_fact_fts_llm(tmp_db)
        assert n == 1  # only the llm fact
        rows = tmp_db.execute("SELECT rowid FROM fts_facts_llm").fetchall()
        ids = {r[0] for r in rows}
        assert llm_id in ids
        assert svo_id not in ids

    def test_rebuild_includes_closure(self, tmp_db):
        _, _, eid = _seed_mixed_sources(tmp_db)
        src_id = insert_source(tmp_db, sha256=sha("cl_src"), title="Cl")
        chunk_id = insert_chunk(tmp_db, source_id=src_id, ordinal=0,
                                text="Derived.", sha256=sha("cl_chunk"))
        cl_id = insert_fact(tmp_db, chunk_id=chunk_id,
                            sentence="Derived: Holmes profession detective.",
                            subject_id=eid, predicate="profession",
                            object_text="detective", source="closure")
        rebuild_fact_fts_llm(tmp_db)
        rows = tmp_db.execute("SELECT rowid FROM fts_facts_llm").fetchall()
        ids = {r[0] for r in rows}
        assert cl_id in ids

    def test_idempotent_rebuild(self, tmp_db):
        _seed_mixed_sources(tmp_db)
        n1 = rebuild_fact_fts_llm(tmp_db)
        n2 = rebuild_fact_fts_llm(tmp_db)
        assert n1 == n2
        assert tmp_db.execute("SELECT COUNT(*) FROM fts_facts_llm").fetchone()[0] == n1

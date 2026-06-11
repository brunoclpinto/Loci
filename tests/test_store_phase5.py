"""Phase 5 store tests: stats, entity merge, pending links, maintenance."""
from __future__ import annotations

import pytest

from loci.store import (
    db_analyze,
    db_vacuum,
    dismiss_pending_link,
    get_pending_links,
    get_stats,
    insert_alias,
    insert_entity,
    insert_source,
    insert_chunk,
    insert_fact,
    merge_entities,
    open_db,
)

SPEC_TEXT = (
    "Sherlock Holmes took his bottle from the corner of the mantel-piece. "
    "Holmes also took his syringe from its neat morocco case."
)


@pytest.fixture(scope="module")
def populated(tmp_path_factory, fake_embedder, nlp, default_cfg):
    tmp = tmp_path_factory.mktemp("phase5_store")
    from loci.ingest import ingest_file

    conn = open_db(tmp / "test.db")
    p = tmp / "spec.txt"
    p.write_text(SPEC_TEXT)
    ingest_file(p, cfg=default_cfg, conn=conn, embedder=fake_embedder, spacy_nlp=nlp)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_returns_entity_count(self, populated):
        stats = get_stats(populated)
        assert stats["entities_count"] > 0

    def test_returns_facts_count(self, populated):
        stats = get_stats(populated)
        assert stats["facts_count"] > 0

    def test_returns_chunks_count(self, populated):
        stats = get_stats(populated)
        assert stats["chunks_count"] > 0

    def test_db_size_positive(self, populated):
        stats = get_stats(populated)
        assert stats["db_size_bytes"] > 0
        assert stats["db_size_mb"] > 0

    def test_top_entities_non_empty(self, populated):
        stats = get_stats(populated)
        assert len(stats["top_entities"]) > 0

    def test_top_entity_has_required_keys(self, populated):
        stats = get_stats(populated)
        e = stats["top_entities"][0]
        assert "name" in e and "kind" in e and "facts" in e

    def test_top_entity_is_holmes(self, populated):
        stats = get_stats(populated)
        names = [e["name"] for e in stats["top_entities"]]
        # The entity with most facts should be Sherlock Holmes
        assert any("Holmes" in n or "Sherlock" in n for n in names)


# ---------------------------------------------------------------------------
# merge_entities
# ---------------------------------------------------------------------------

class TestMergeEntities:
    @pytest.fixture
    def merge_db(self, tmp_path):
        """Small DB with two entities and facts, for merge testing."""
        conn = open_db(tmp_path / "merge.db")
        src_id = insert_source(conn, path="f.txt", title="T", sha256="abc123")
        cid = insert_chunk(conn, source_id=src_id, ordinal=0,
                           text="Holmes took the bottle.", sha256="ch1")
        eid_a = insert_entity(conn, canonical_name="Sherlock Holmes", kind="person")
        insert_alias(conn, entity_id=eid_a, alias="sherlock holmes")
        eid_b = insert_entity(conn, canonical_name="Holmes", kind="person")
        insert_alias(conn, entity_id=eid_b, alias="holmes")
        insert_fact(conn, chunk_id=cid, sentence="Holmes took the bottle.",
                    subject_id=eid_b, predicate="take", object_text="bottle")
        conn.commit()
        return conn, eid_a, eid_b

    def test_facts_reattach_to_kept_entity(self, merge_db):
        conn, eid_a, eid_b = merge_db
        merge_entities(conn, keep_id=eid_a, merge_id=eid_b)
        facts = conn.execute(
            "SELECT * FROM facts WHERE subject_id=?", [eid_a]
        ).fetchall()
        assert len(facts) > 0

    def test_merged_entity_deleted(self, merge_db):
        conn, eid_a, eid_b = merge_db
        merge_entities(conn, keep_id=eid_a, merge_id=eid_b)
        row = conn.execute("SELECT id FROM entities WHERE id=?", [eid_b]).fetchone()
        assert row is None

    def test_aliases_transferred(self, merge_db):
        conn, eid_a, eid_b = merge_db
        merge_entities(conn, keep_id=eid_a, merge_id=eid_b)
        aliases = {r["alias"] for r in conn.execute(
            "SELECT alias FROM aliases WHERE entity_id=?", [eid_a]
        ).fetchall()}
        assert "holmes" in aliases
        assert "sherlock holmes" in aliases


# ---------------------------------------------------------------------------
# pending_links / dismiss
# ---------------------------------------------------------------------------

class TestPendingLinks:
    @pytest.fixture
    def pending_db(self, tmp_path):
        import json
        conn = open_db(tmp_path / "pending.db")
        eid_a = insert_entity(conn, canonical_name="A", kind="person")
        eid_b = insert_entity(conn, canonical_name="B", kind="person")
        conn.execute(
            "INSERT INTO pending_links (mention, candidate_entity_ids) VALUES (?,?)",
            ["the detective", json.dumps([eid_a, eid_b])],
        )
        conn.commit()
        return conn

    def test_get_pending_links_returns_entries(self, pending_db):
        links = get_pending_links(pending_db)
        assert len(links) == 1

    def test_pending_link_has_mention(self, pending_db):
        link = get_pending_links(pending_db)[0]
        assert link["mention"] == "the detective"

    def test_pending_link_has_candidates(self, pending_db):
        link = get_pending_links(pending_db)[0]
        assert len(link["candidates"]) == 2
        names = {c["name"] for c in link["candidates"]}
        assert names == {"A", "B"}

    def test_dismiss_removes_link(self, pending_db):
        link = get_pending_links(pending_db)[0]
        dismiss_pending_link(pending_db, link["id"])
        assert get_pending_links(pending_db) == []

    def test_empty_when_no_pending(self, tmp_path):
        conn = open_db(tmp_path / "nopending.db")
        assert get_pending_links(conn) == []
        conn.close()


# ---------------------------------------------------------------------------
# db_vacuum / db_analyze
# ---------------------------------------------------------------------------

class TestMaintenance:
    def test_vacuum_runs_without_error(self, populated):
        db_vacuum(populated)  # should not raise

    def test_analyze_runs_without_error(self, populated):
        db_analyze(populated)  # should not raise

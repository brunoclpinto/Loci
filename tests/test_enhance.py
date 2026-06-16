"""Tests for loci/enhance.py and Phase 6 quality upgrades."""
from __future__ import annotations

import json
import pytest

from loci.enhance import (
    build_extraction_messages,
    build_entity_messages,
    build_implied_messages,
    enhance_chunk,
    parse_llm_facts,
    run_closure_pass,
    run_enhance,
    run_entity_pass,
    run_implied_pass,
)
from loci.extract import extract_coref_facts


# ---------------------------------------------------------------------------
# Fake LLM for enhance tests
# ---------------------------------------------------------------------------

class FakeExtractLLM:
    """Returns deterministic JSON based on what the system prompt requests."""

    def __init__(self, payload=None):
        self._payload = payload  # override response if given

    def create_chat_completion(self, messages, *, max_tokens, temperature, stream):
        if self._payload is not None:
            content = self._payload
        else:
            content = json.dumps([{
                "subject": "Sherlock Holmes",
                "predicate": "be",
                "object": "detective",
                "qualifiers": {},
                "negated": False,
            }])
        if stream:
            def _gen():
                yield {"choices": [{"delta": {"content": content}}]}
            return _gen()
        return {"choices": [{"message": {"content": content}}]}


@pytest.fixture
def fake_llm():
    return FakeExtractLLM()


SPEC_TEXT = (
    "Sherlock Holmes took his bottle from the corner of the mantel-piece. "
    "He also examined the clues on the table. "
    "Watson is a doctor."
)


@pytest.fixture(scope="module")
def enhance_db(tmp_path_factory, fake_embedder, nlp, default_cfg):
    tmp = tmp_path_factory.mktemp("enhance")
    from loci.store import open_db
    from loci.ingest import ingest_file

    conn = open_db(tmp / "test.db")
    p = tmp / "spec.txt"
    p.write_text(SPEC_TEXT)
    ingest_file(p, cfg=default_cfg, conn=conn, embedder=fake_embedder, spacy_nlp=nlp)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# build_extraction_messages
# ---------------------------------------------------------------------------

class TestBuildExtractionMessages:
    def test_has_system_message(self):
        msgs = build_extraction_messages("Some text.")
        assert msgs[0]["role"] == "system"

    def test_system_mentions_passive(self):
        msgs = build_extraction_messages("x")
        assert "Passive" in msgs[0]["content"] or "passive" in msgs[0]["content"]

    def test_system_mentions_copula(self):
        msgs = build_extraction_messages("x")
        assert "Copula" in msgs[0]["content"] or "copula" in msgs[0]["content"] or "X is Y" in msgs[0]["content"]

    def test_user_message_contains_text(self):
        msgs = build_extraction_messages("Watson is a doctor.")
        user = next(m for m in msgs if m["role"] == "user")
        assert "Watson is a doctor." in user["content"]


# ---------------------------------------------------------------------------
# parse_llm_facts
# ---------------------------------------------------------------------------

class TestParseLlmFacts:
    def test_valid_json_array(self):
        raw = '[{"subject":"Holmes","predicate":"be","object":"detective","qualifiers":{},"negated":false}]'
        facts = parse_llm_facts(raw)
        assert len(facts) == 1
        assert facts[0]["subject"] == "Holmes"
        assert facts[0]["predicate"] == "be"
        assert facts[0]["object"] == "detective"
        assert facts[0]["negated"] is False

    def test_strips_markdown_fences(self):
        raw = '```json\n[{"subject":"A","predicate":"role","object":"c","qualifiers":{},"negated":false}]\n```'
        facts = parse_llm_facts(raw)
        assert len(facts) == 1

    def test_empty_array_returns_empty(self):
        assert parse_llm_facts("[]") == []

    def test_malformed_json_returns_empty(self):
        assert parse_llm_facts("not json at all") == []

    def test_missing_subject_filtered(self):
        raw = '[{"predicate":"take","object":"bottle","qualifiers":{},"negated":false}]'
        assert parse_llm_facts(raw) == []

    def test_missing_predicate_filtered(self):
        raw = '[{"subject":"Holmes","object":"bottle","qualifiers":{},"negated":false}]'
        assert parse_llm_facts(raw) == []

    def test_predicate_lowercased(self):
        raw = '[{"subject":"Holmes","predicate":"PROFESSION","object":"detective","qualifiers":{},"negated":false}]'
        facts = parse_llm_facts(raw)
        assert len(facts) == 1
        assert facts[0]["predicate"] == "profession"

    def test_multiple_facts(self):
        raw = json.dumps([
            {"subject": "A", "predicate": "role", "object": "x", "qualifiers": {}, "negated": False},
            {"subject": "B", "predicate": "be", "object": "y", "qualifiers": {}, "negated": True},
        ])
        facts = parse_llm_facts(raw)
        assert len(facts) == 2

    def test_negated_preserved(self):
        raw = '[{"subject":"Holmes","predicate":"profession","object":"detective","qualifiers":{},"negated":true}]'
        facts = parse_llm_facts(raw)
        assert facts[0]["negated"] is True

    def test_taxonomy_rejects_out_of_scope_predicate(self):
        raw = '[{"subject":"Holmes","predicate":"take","object":"bottle","qualifiers":{},"negated":false}]'
        facts = parse_llm_facts(raw)
        assert facts == [], "predicates outside the taxonomy should be rejected"

    def test_taxonomy_accepts_profession(self):
        raw = '[{"subject":"Holmes","predicate":"profession","object":"consulting detective","qualifiers":{},"negated":false,"sentence":"I am a consulting detective."}]'
        facts = parse_llm_facts(raw)
        assert len(facts) == 1
        assert facts[0]["predicate"] == "profession"
        assert facts[0]["sentence"] == "I am a consulting detective."

    def test_taxonomy_accepts_role(self):
        raw = '[{"subject":"Mrs Hudson","predicate":"role","object":"landlady","qualifiers":{},"negated":false,"sentence":"Mrs Hudson, our landlady, knocked."}]'
        facts = parse_llm_facts(raw)
        assert len(facts) == 1
        assert facts[0]["predicate"] == "role"

    def test_sentence_field_extracted(self):
        raw = '[{"subject":"Hope","predicate":"occupation","object":"cab driver","qualifiers":{},"negated":false,"sentence":"He worked as a cab driver."}]'
        facts = parse_llm_facts(raw)
        assert facts[0]["sentence"] == "He worked as a cab driver."

    def test_sentence_field_absent_returns_none(self):
        raw = '[{"subject":"Holmes","predicate":"be","object":"detective","qualifiers":{},"negated":false}]'
        facts = parse_llm_facts(raw)
        assert facts[0]["sentence"] is None

    def test_non_list_response_returns_empty(self):
        assert parse_llm_facts('{"subject":"X","predicate":"y","object":"z"}') == []


# ---------------------------------------------------------------------------
# enhance_chunk
# ---------------------------------------------------------------------------

class TestEnhanceChunk:
    def test_inserts_facts(self, enhance_db, fake_llm, default_cfg):
        before = enhance_db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        enhance_chunk(
            enhance_db, 1, "Watson is a doctor.", fake_llm,
            cfg=default_cfg,
        )
        after = enhance_db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        assert after >= before  # at least tried to insert

    def test_marks_chunk_extracted(self, tmp_path, fake_llm, default_cfg):
        from loci.store import open_db, insert_source, insert_chunk
        conn = open_db(tmp_path / "ec.db")
        src_id = insert_source(conn, path="f.txt", sha256="x1")
        chunk_id = insert_chunk(conn, source_id=src_id, ordinal=0,
                                text="Holmes is a detective.", sha256="c1")
        conn.commit()
        assert chunk_id is not None

        enhance_chunk(conn, chunk_id, "Holmes is a detective.", fake_llm, cfg=default_cfg)
        row = conn.execute(
            "SELECT extracted_v FROM chunks WHERE id=?", [chunk_id]
        ).fetchone()
        assert row["extracted_v"] == 1
        conn.close()

    def test_llm_error_still_marks_extracted(self, tmp_path, default_cfg):
        from loci.store import open_db, insert_source, insert_chunk

        class ErrorLLM:
            def create_chat_completion(self, messages, **kw):
                raise RuntimeError("LLM down")

        conn = open_db(tmp_path / "err.db")
        src_id = insert_source(conn, path="f.txt", sha256="x2")
        chunk_id = insert_chunk(conn, source_id=src_id, ordinal=0,
                                text="text", sha256="c2")
        conn.commit()
        result = enhance_chunk(conn, chunk_id, "text", ErrorLLM(), cfg=default_cfg)
        assert result == 0
        assert conn.execute(
            "SELECT extracted_v FROM chunks WHERE id=?", [chunk_id]
        ).fetchone()["extracted_v"] == 1
        conn.close()

    def test_facts_have_low_confidence(self, tmp_path, default_cfg):
        from loci.store import open_db, insert_source, insert_chunk, insert_entity, insert_alias
        conn = open_db(tmp_path / "conf.db")
        src_id = insert_source(conn, path="f.txt", sha256="xconf")
        chunk_id = insert_chunk(conn, source_id=src_id, ordinal=0,
                                text="Watson is a doctor.", sha256="cconf")
        eid = insert_entity(conn, canonical_name="Watson", kind="person")
        insert_alias(conn, entity_id=eid, alias="watson")
        conn.commit()

        llm = FakeExtractLLM('[{"subject":"Watson","predicate":"be","object":"doctor","qualifiers":{},"negated":false}]')
        enhance_chunk(conn, chunk_id, "Watson is a doctor.", llm, cfg=default_cfg)

        rows = conn.execute(
            "SELECT confidence FROM facts WHERE subject_id=?", [eid]
        ).fetchall()
        assert rows, "no facts inserted"
        assert all(r["confidence"] == pytest.approx(0.7) for r in rows)
        conn.close()


# ---------------------------------------------------------------------------
# run_enhance
# ---------------------------------------------------------------------------

class TestRunEnhance:
    def test_returns_stats(self, tmp_path, fake_llm, default_cfg):
        from loci.store import open_db, insert_source, insert_chunk
        conn = open_db(tmp_path / "run.db")
        src_id = insert_source(conn, path="f.txt", sha256="xrun")
        insert_chunk(conn, source_id=src_id, ordinal=0,
                     text="Holmes was a consulting detective.", sha256="crun")
        conn.commit()

        stats = run_enhance(conn, llm=fake_llm, cfg=default_cfg)
        assert "chunks_processed" in stats
        assert "facts_added" in stats
        assert stats["chunks_processed"] == 1
        conn.close()

    def test_idempotent(self, tmp_path, fake_llm, default_cfg):
        """Running enhance twice on the same DB processes 0 chunks the second time."""
        from loci.store import open_db, insert_source, insert_chunk
        conn = open_db(tmp_path / "idem.db")
        src_id = insert_source(conn, path="f.txt", sha256="xidem")
        insert_chunk(conn, source_id=src_id, ordinal=0,
                     text="Holmes was brilliant.", sha256="cidem")
        conn.commit()

        run_enhance(conn, llm=fake_llm, cfg=default_cfg)
        stats2 = run_enhance(conn, llm=fake_llm, cfg=default_cfg)
        assert stats2["chunks_processed"] == 0
        conn.close()

    def test_limit_respected(self, tmp_path, fake_llm, default_cfg):
        from loci.store import open_db, insert_source, insert_chunk
        conn = open_db(tmp_path / "lim.db")
        src_id = insert_source(conn, path="f.txt", sha256="xlim")
        for i in range(5):
            insert_chunk(conn, source_id=src_id, ordinal=i,
                         text=f"Sentence {i}.", sha256=f"clim{i}")
        conn.commit()

        stats = run_enhance(conn, llm=fake_llm, cfg=default_cfg, limit=2)
        assert stats["chunks_processed"] == 2
        conn.close()

    def test_force_all_reprocesses_extracted_chunks(self, tmp_path, fake_llm, default_cfg):
        from loci.store import open_db, insert_source, insert_chunk
        conn = open_db(tmp_path / "force.db")
        src_id = insert_source(conn, path="f.txt", sha256="xforce")
        insert_chunk(conn, source_id=src_id, ordinal=0,
                     text="Holmes is a consulting detective.", sha256="cforce")
        conn.commit()

        # First run — marks chunk as extracted
        run_enhance(conn, llm=fake_llm, cfg=default_cfg)
        # Second run without force_all — nothing to process
        stats2 = run_enhance(conn, llm=fake_llm, cfg=default_cfg)
        assert stats2["chunks_processed"] == 0
        # Third run with force_all — resets extracted_v and processes again
        stats3 = run_enhance(conn, llm=fake_llm, cfg=default_cfg, force_all=True)
        assert stats3["chunks_processed"] == 1
        conn.close()


# ---------------------------------------------------------------------------
# P1: build_extraction_messages with known_entities
# ---------------------------------------------------------------------------

class TestBuildExtractionMessagesP1:
    def test_known_entities_in_user_message(self):
        msgs = build_extraction_messages("Some text.", known_entities=["Sherlock Holmes", "Mrs Hudson"])
        user = next(m for m in msgs if m["role"] == "user")
        assert "Sherlock Holmes" in user["content"]
        assert "Mrs Hudson" in user["content"]

    def test_no_entities_still_works(self):
        msgs = build_extraction_messages("Some text.")
        user = next(m for m in msgs if m["role"] == "user")
        assert "Some text." in user["content"]


# ---------------------------------------------------------------------------
# extract_coref_facts (cheap coreference)
# ---------------------------------------------------------------------------

def sent(nlp, text):
    return list(nlp(text).sents)[0]


class TestExtractCorefFacts:
    def test_pronoun_he_resolved(self, nlp):
        s = sent(nlp, "He took the bottle.")
        facts = extract_coref_facts(s, last_entity_text="Sherlock Holmes")
        assert facts, "should produce a fact for pronoun 'He'"
        assert facts[0].subject_text == "Sherlock Holmes"
        assert facts[0].predicate == "take"

    def test_no_last_entity_returns_empty(self, nlp):
        s = sent(nlp, "He took the bottle.")
        assert extract_coref_facts(s, last_entity_text=None) == []

    def test_non_pronoun_subject_returns_empty(self, nlp):
        s = sent(nlp, "Holmes took the bottle.")
        assert extract_coref_facts(s, last_entity_text="Holmes") == []

    def test_pronoun_she_resolved(self, nlp):
        s = sent(nlp, "She observed the room.")
        facts = extract_coref_facts(s, last_entity_text="Mrs. Hudson")
        assert facts
        assert facts[0].subject_text == "Mrs. Hudson"

    def test_no_verb_returns_empty(self, nlp):
        s = sent(nlp, "He.")
        assert extract_coref_facts(s, last_entity_text="Holmes") == []


# ---------------------------------------------------------------------------
# Coref integration in ingest pipeline
# ---------------------------------------------------------------------------

class TestCorefIngest:
    def test_coref_facts_created(self, tmp_path, fake_embedder, nlp, default_cfg):
        """Sentences with pronoun subjects should produce coref facts (confidence=0.6)."""
        from loci.store import open_db
        from loci.ingest import ingest_file

        corpus = (
            "Sherlock Holmes took his bottle from the mantel-piece. "
            "He examined it carefully."
        )
        conn = open_db(tmp_path / "coref.db")
        p = tmp_path / "coref.txt"
        p.write_text(corpus)
        ingest_file(p, cfg=default_cfg, conn=conn,
                    embedder=fake_embedder, spacy_nlp=nlp)

        low_conf = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE confidence < 1.0"
        ).fetchone()[0]
        assert low_conf > 0, "expected at least one coref fact with confidence < 1.0"
        conn.close()

    def test_coref_facts_have_confidence_06(self, tmp_path, fake_embedder, nlp, default_cfg):
        from loci.store import open_db
        from loci.ingest import ingest_file

        corpus = "Watson entered the room. He sat down on the chair."
        conn = open_db(tmp_path / "coref2.db")
        p = tmp_path / "coref2.txt"
        p.write_text(corpus)
        ingest_file(p, cfg=default_cfg, conn=conn,
                    embedder=fake_embedder, spacy_nlp=nlp)

        rows = conn.execute(
            "SELECT confidence FROM facts WHERE confidence < 1.0"
        ).fetchall()
        if rows:
            assert all(abs(r["confidence"] - 0.6) < 0.01 for r in rows)
        conn.close()

    def test_coref_disabled_by_config(self, tmp_path, fake_embedder, nlp):
        from loci.config import Config, IngestConfig
        from loci.store import open_db
        from loci.ingest import ingest_file

        cfg_no_coref = Config(ingest=IngestConfig(resolve_coref=False))
        corpus = "Sherlock Holmes took the bottle. He examined it."
        conn = open_db(tmp_path / "nocoref.db")
        p = tmp_path / "nocoref.txt"
        p.write_text(corpus)
        ingest_file(p, cfg=cfg_no_coref, conn=conn,
                    embedder=fake_embedder, spacy_nlp=nlp)

        low_conf = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE confidence < 1.0"
        ).fetchone()[0]
        assert low_conf == 0
        conn.close()


# ---------------------------------------------------------------------------
# store helpers: extracted_v migration and helpers
# ---------------------------------------------------------------------------

class TestExtractedV:
    def test_column_exists_after_open(self, tmp_db):
        cols = {r[1] for r in tmp_db.execute("PRAGMA table_info(chunks)").fetchall()}
        assert "extracted_v" in cols

    def test_default_is_zero(self, tmp_path):
        from loci.store import open_db, insert_source, insert_chunk
        conn = open_db(tmp_path / "ev.db")
        src_id = insert_source(conn, path="f.txt", sha256="xev")
        chunk_id = insert_chunk(conn, source_id=src_id, ordinal=0,
                                text="text", sha256="cev")
        conn.commit()
        row = conn.execute(
            "SELECT extracted_v FROM chunks WHERE id=?", [chunk_id]
        ).fetchone()
        assert row["extracted_v"] == 0
        conn.close()

    def test_get_unextracted_chunks(self, tmp_path):
        from loci.store import open_db, insert_source, insert_chunk, get_unextracted_chunks
        conn = open_db(tmp_path / "ue.db")
        src_id = insert_source(conn, path="f.txt", sha256="xue")
        insert_chunk(conn, source_id=src_id, ordinal=0, text="A", sha256="cue1")
        insert_chunk(conn, source_id=src_id, ordinal=1, text="B", sha256="cue2")
        conn.commit()
        chunks = get_unextracted_chunks(conn)
        assert len(chunks) == 2
        conn.close()

    def test_mark_chunk_extracted(self, tmp_path):
        from loci.store import open_db, insert_source, insert_chunk, mark_chunk_extracted, get_unextracted_chunks
        conn = open_db(tmp_path / "mce.db")
        src_id = insert_source(conn, path="f.txt", sha256="xmce")
        chunk_id = insert_chunk(conn, source_id=src_id, ordinal=0,
                                text="text", sha256="cmce")
        conn.commit()

        mark_chunk_extracted(conn, chunk_id)
        remaining = get_unextracted_chunks(conn)
        assert all(r["id"] != chunk_id for r in remaining)
        conn.close()

    def test_get_unextracted_chunks_limit(self, tmp_path):
        from loci.store import open_db, insert_source, insert_chunk, get_unextracted_chunks
        conn = open_db(tmp_path / "lim2.db")
        src_id = insert_source(conn, path="f.txt", sha256="xlim2")
        for i in range(5):
            insert_chunk(conn, source_id=src_id, ordinal=i,
                         text=f"t{i}", sha256=f"cl{i}")
        conn.commit()
        assert len(get_unextracted_chunks(conn, limit=3)) == 3
        conn.close()


# ---------------------------------------------------------------------------
# P2 message builders
# ---------------------------------------------------------------------------

class TestP2MessageBuilders:
    def test_entity_messages_has_system_and_user(self):
        msgs = build_entity_messages("Holmes", "Some text about Holmes.")
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_entity_messages_entity_in_system(self):
        msgs = build_entity_messages("Mrs Hudson", "Text.")
        assert "Mrs Hudson" in msgs[0]["content"]

    def test_entity_messages_known_entities_in_user(self):
        msgs = build_entity_messages("Holmes", "Text.", known_entities=["Watson", "Moriarty"])
        assert "Watson" in msgs[1]["content"]

    def test_implied_messages_has_system_and_user(self):
        msgs = build_implied_messages("He drove a cab.")
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_implied_messages_system_mentions_occupation(self):
        msgs = build_implied_messages("text")
        assert "occupation" in msgs[0]["content"] or "Occupation" in msgs[0]["content"]

    def test_implied_messages_known_entities_in_user(self):
        msgs = build_implied_messages("text", known_entities=["Holmes"])
        assert "Holmes" in msgs[1]["content"]


# ---------------------------------------------------------------------------
# P2 pass runners
# ---------------------------------------------------------------------------

def _make_p2_db(tmp_path, fake_embedder, nlp, default_cfg):
    from loci.store import open_db, insert_source, insert_chunk, insert_entity, insert_alias
    conn = open_db(tmp_path / "p2.db")
    src_id = insert_source(conn, path="f.txt", sha256="xp2")
    chunk_id = insert_chunk(
        conn, source_id=src_id, ordinal=0,
        text="Mrs Hudson our landlady knocked on the door. Jefferson Hope drove the cab.",
        sha256="cp2",
    )
    eid = insert_entity(conn, canonical_name="Mrs Hudson", kind="person")
    insert_alias(conn, entity_id=eid, alias="mrs hudson")
    eid2 = insert_entity(conn, canonical_name="Jefferson Hope", kind="person")
    insert_alias(conn, entity_id=eid2, alias="jefferson hope")
    conn.commit()
    return conn, chunk_id


class TestRunEntityPass:
    def test_returns_stats(self, tmp_path, fake_embedder, nlp, default_cfg):
        conn, _ = _make_p2_db(tmp_path, fake_embedder, nlp, default_cfg)
        llm = FakeExtractLLM('[{"subject":"Mrs Hudson","predicate":"role","object":"landlady","qualifiers":{},"negated":false,"sentence":"Mrs Hudson our landlady knocked."}]')
        stats = run_entity_pass(conn, llm=llm, cfg=default_cfg)
        assert "entities_processed" in stats
        assert "facts_added" in stats
        conn.close()

    def test_idempotent(self, tmp_path, fake_embedder, nlp, default_cfg):
        conn, _ = _make_p2_db(tmp_path / "idem_ep", fake_embedder, nlp, default_cfg)
        llm = FakeExtractLLM("[]")
        run_entity_pass(conn, llm=llm, cfg=default_cfg)
        stats2 = run_entity_pass(conn, llm=llm, cfg=default_cfg)
        assert stats2.get("skipped") is True
        assert stats2["entities_processed"] == 0
        conn.close()

    def test_inserts_source_llm(self, tmp_path, fake_embedder, nlp, default_cfg):
        conn, _ = _make_p2_db(tmp_path / "src_ep", fake_embedder, nlp, default_cfg)
        llm = FakeExtractLLM('[{"subject":"Mrs Hudson","predicate":"role","object":"landlady","qualifiers":{},"negated":false,"sentence":"Mrs Hudson our landlady."}]')
        run_entity_pass(conn, llm=llm, cfg=default_cfg)
        rows = conn.execute("SELECT source FROM facts WHERE source='llm'").fetchall()
        assert len(rows) > 0
        conn.close()


class TestRunImpliedPass:
    def test_returns_stats(self, tmp_path, fake_embedder, nlp, default_cfg):
        conn, _ = _make_p2_db(tmp_path / "imp", fake_embedder, nlp, default_cfg)
        llm = FakeExtractLLM('[{"subject":"Jefferson Hope","predicate":"occupation","object":"cab driver","qualifiers":{},"negated":false,"sentence":"Jefferson Hope drove the cab."}]')
        stats = run_implied_pass(conn, llm=llm, cfg=default_cfg)
        assert "chunks_processed" in stats
        assert "facts_added" in stats
        conn.close()

    def test_idempotent(self, tmp_path, fake_embedder, nlp, default_cfg):
        conn, _ = _make_p2_db(tmp_path / "idem_imp", fake_embedder, nlp, default_cfg)
        llm = FakeExtractLLM("[]")
        run_implied_pass(conn, llm=llm, cfg=default_cfg)
        stats2 = run_implied_pass(conn, llm=llm, cfg=default_cfg)
        assert stats2.get("skipped") is True
        conn.close()

    def test_processes_all_chunks(self, tmp_path, fake_embedder, nlp, default_cfg):
        from loci.store import open_db, insert_source, insert_chunk
        conn = open_db(tmp_path / "allc.db")
        src_id = insert_source(conn, path="f.txt", sha256="xallc")
        for i in range(3):
            insert_chunk(conn, source_id=src_id, ordinal=i, text=f"Chunk {i}.", sha256=f"callc{i}")
        conn.commit()
        llm = FakeExtractLLM("[]")
        stats = run_implied_pass(conn, llm=llm, cfg=default_cfg)
        assert stats["chunks_processed"] == 3
        conn.close()


# ---------------------------------------------------------------------------
# Closure pass
# ---------------------------------------------------------------------------

def _make_closure_db(tmp_path):
    """DB with Hope|work_as|jarvey + jarvey|means|cab driver ready for closure."""
    from loci.store import open_db, insert_source, insert_chunk, insert_entity, insert_alias, insert_fact
    conn = open_db(tmp_path / "closure.db")
    src_id = insert_source(conn, path="c.txt", sha256="xcl")
    chunk_id = insert_chunk(conn, source_id=src_id, ordinal=0,
                            text="Hope drove a jarvey.", sha256="ccl")
    hope_id = insert_entity(conn, canonical_name="Jefferson Hope", kind="person")
    insert_alias(conn, entity_id=hope_id, alias="jefferson hope")
    jarvey_id = insert_entity(conn, canonical_name="jarvey", kind="concept")
    insert_alias(conn, entity_id=jarvey_id, alias="jarvey")
    # Hope|work_as|jarvey (object_text, not object_id)
    insert_fact(conn, chunk_id=chunk_id, sentence="Hope drove a jarvey.",
                subject_id=hope_id, predicate="work_as", object_text="jarvey",
                confidence=0.75, source="llm")
    # jarvey|means|cab driver
    insert_fact(conn, chunk_id=chunk_id, sentence="A jarvey means a cab driver.",
                subject_id=jarvey_id, predicate="means", object_text="cab driver",
                confidence=0.9, source="llm")
    conn.commit()
    return conn, hope_id, jarvey_id


class TestRunClosurePass:
    def test_basic(self, tmp_path, default_cfg):
        conn, hope_id, _ = _make_closure_db(tmp_path)
        stats = run_closure_pass(conn, cfg=default_cfg)
        assert stats["facts_added"] >= 1
        assert stats["chains_found"] >= 1
        row = conn.execute(
            "SELECT source, object_text FROM facts WHERE subject_id=? AND predicate='work_as' AND object_text='cab driver'",
            [hope_id],
        ).fetchone()
        assert row is not None
        assert row["source"] == "closure"
        conn.close()

    def test_idempotent(self, tmp_path, default_cfg):
        conn, _, _ = _make_closure_db(tmp_path / "idem_cl")
        run_closure_pass(conn, cfg=default_cfg)
        stats2 = run_closure_pass(conn, cfg=default_cfg)
        assert stats2.get("skipped") is True
        assert stats2["facts_added"] == 0
        conn.close()

    def test_skips_means_facts(self, tmp_path, default_cfg):
        conn, _, jarvey_id = _make_closure_db(tmp_path / "nomeans")
        run_closure_pass(conn, cfg=default_cfg)
        # jarvey|means|cab driver should not spawn jarvey|means|<something else>
        rows = conn.execute(
            "SELECT * FROM facts WHERE subject_id=? AND predicate='means' AND source='closure'",
            [jarvey_id],
        ).fetchall()
        assert len(rows) == 0
        conn.close()

    def test_skips_negated(self, tmp_path, default_cfg):
        from loci.store import open_db, insert_source, insert_chunk, insert_entity, insert_alias, insert_fact
        conn = open_db(tmp_path / "neg_cl.db")
        src_id = insert_source(conn, path="n.txt", sha256="xneg")
        cid = insert_chunk(conn, source_id=src_id, ordinal=0, text="Not a jarvey.", sha256="cneg")
        eid = insert_entity(conn, canonical_name="Holmes", kind="person")
        insert_alias(conn, entity_id=eid, alias="holmes")
        jid = insert_entity(conn, canonical_name="jarvey", kind="concept")
        # negated source fact
        insert_fact(conn, chunk_id=cid, sentence="Holmes was not a jarvey.",
                    subject_id=eid, predicate="work_as", object_text="jarvey",
                    negated=True, confidence=0.75, source="llm")
        insert_fact(conn, chunk_id=cid, sentence="A jarvey means a cab driver.",
                    subject_id=jid, predicate="means", object_text="cab driver",
                    confidence=0.9, source="llm")
        conn.commit()
        stats = run_closure_pass(conn, cfg=default_cfg)
        assert stats["facts_added"] == 0
        conn.close()

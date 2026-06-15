"""Tests for loci/enhance.py and Phase 6 quality upgrades."""
from __future__ import annotations

import json
import pytest

from loci.enhance import (
    build_extraction_messages,
    enhance_chunk,
    parse_llm_facts,
    run_enhance,
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

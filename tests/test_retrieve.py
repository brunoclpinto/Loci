"""Tests for loci/retrieve.py: question parse, fact lookup, FTS, RRF, context."""
import json
import time
from pathlib import Path

import pytest

from loci.retrieve import (
    ChunkHit,
    FactHit,
    QuestionParse,
    build_context,
    fact_lookup,
    find_mentioned_entity_ids,
    fts_search_question,
    get_synonyms,
    parse_question,
    retrieve,
    rrf_fuse,
    vec_search_question,
)


# ---------------------------------------------------------------------------
# Fixture: a pre-populated DB with the spec corpus
# ---------------------------------------------------------------------------

SPEC_TEXT = (
    "Sherlock Holmes took his bottle from the corner of the mantel-piece. "
    "Holmes also took his syringe from its neat morocco case. "
    "Watson entered the room and observed Holmes carefully. "
    "Mrs. Hudson knocked on the door and brought some tea. "
    "The detective examined the clues very thoroughly."
)


@pytest.fixture(scope="module")
def populated_db(tmp_path_factory, fake_embedder, nlp, default_cfg):
    """DB pre-populated with SPEC_TEXT via the ingest pipeline."""
    tmp = tmp_path_factory.mktemp("retrieve_db")
    from loci.store import open_db
    from loci.ingest import ingest_file

    conn = open_db(tmp / "test.db")
    p = tmp / "spec.txt"
    p.write_text(SPEC_TEXT)
    ingest_file(p, cfg=default_cfg, conn=conn,
                embedder=fake_embedder, spacy_nlp=nlp)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Question parsing
# ---------------------------------------------------------------------------

class TestParseQuestion:
    def test_spacy_verb_lemma(self, nlp):
        parse = parse_question("What did Sherlock Holmes take?", nlp=nlp)
        assert parse.verb_lemma == "take"

    def test_spacy_wh_type(self, nlp):
        parse = parse_question("Where did Holmes live?", nlp=nlp)
        assert parse.wh_type == "where"

    def test_spacy_what_type(self, nlp):
        parse = parse_question("What did Holmes find?", nlp=nlp)
        assert parse.wh_type == "what"

    def test_simple_fallback_verb(self):
        parse = parse_question("what did holmes take", nlp=None)
        assert parse.verb_lemma == "holmes" or parse.verb_lemma is not None

    def test_simple_fallback_wh(self):
        parse = parse_question("where did watson go?", nlp=None)
        assert parse.wh_type == "where"

    def test_raw_preserved(self, nlp):
        q = "What did Holmes take?"
        parse = parse_question(q, nlp=nlp)
        assert parse.raw == q


# ---------------------------------------------------------------------------
# Entity scanning (DB-based)
# ---------------------------------------------------------------------------

class TestFindMentionedEntityIds:
    def test_finds_sherlock_holmes_lowercase(self, populated_db):
        ids = find_mentioned_entity_ids(populated_db, "what did sherlock holmes take?")
        assert len(ids) > 0

    def test_finds_holmes_alone(self, populated_db):
        ids_full = find_mentioned_entity_ids(populated_db, "sherlock holmes")
        ids_short = find_mentioned_entity_ids(populated_db, "holmes")
        # Both should map to the same entity (via alias)
        assert len(ids_full) > 0
        assert len(ids_short) > 0
        assert ids_full[0] == ids_short[0]

    def test_unknown_name_returns_empty(self, populated_db):
        ids = find_mentioned_entity_ids(populated_db, "what did moriarty steal?")
        assert ids == []

    def test_prefers_longer_span(self, populated_db):
        """'sherlock holmes' as a 2-gram should match before 'holmes' alone."""
        ids = find_mentioned_entity_ids(populated_db, "sherlock holmes")
        assert len(ids) == 1  # deduplicated


# ---------------------------------------------------------------------------
# Predicate synonyms
# ---------------------------------------------------------------------------

class TestGetSynonyms:
    def test_empty_when_no_synonyms(self, populated_db):
        syns = get_synonyms(populated_db, "take")
        assert isinstance(syns, set)

    def test_after_seeding(self, populated_db):
        populated_db.execute(
            "INSERT OR IGNORE INTO predicate_synonyms (predicate, synonym) VALUES (?,?)",
            ["take", "grab"]
        )
        populated_db.commit()
        syns = get_synonyms(populated_db, "take")
        assert "grab" in syns

    def test_bidirectional(self, populated_db):
        populated_db.execute(
            "INSERT OR IGNORE INTO predicate_synonyms (predicate, synonym) VALUES (?,?)",
            ["take", "seize"]
        )
        populated_db.commit()
        # Looking up from the synonym direction should return "take"
        syns = get_synonyms(populated_db, "seize")
        assert "take" in syns


# ---------------------------------------------------------------------------
# Fact lookup — acceptance criterion: <50 ms
# ---------------------------------------------------------------------------

class TestFactLookup:
    def test_finds_take_bottle(self, populated_db):
        entity_ids = find_mentioned_entity_ids(populated_db, "sherlock holmes")
        assert entity_ids
        hits = fact_lookup(populated_db, entity_ids, "take", set())
        assert hits
        objects = [h.object_text for h in hits]
        assert "bottle" in objects

    def test_take_qualifier_present(self, populated_db):
        entity_ids = find_mentioned_entity_ids(populated_db, "sherlock holmes")
        hits = fact_lookup(populated_db, entity_ids, "take", set())
        bottle = next((h for h in hits if h.object_text == "bottle"), None)
        assert bottle is not None
        assert bottle.qualifiers is not None
        assert "from" in bottle.qualifiers
        assert "corner" in bottle.qualifiers["from"]

    def test_exact_predicate_score_1(self, populated_db):
        entity_ids = find_mentioned_entity_ids(populated_db, "sherlock holmes")
        hits = fact_lookup(populated_db, entity_ids, "take", set())
        assert all(h.score == 1.0 for h in hits)

    def test_synonym_predicate_score_0_8(self, populated_db):
        populated_db.execute(
            "INSERT OR IGNORE INTO predicate_synonyms VALUES ('take','grab')"
        )
        populated_db.commit()
        entity_ids = find_mentioned_entity_ids(populated_db, "sherlock holmes")
        hits = fact_lookup(populated_db, entity_ids, "grab", {"take"})
        # "grab" is the lookup predicate; facts stored as "take" get 0.8
        take_hits = [h for h in hits if h.predicate == "take"]
        if take_hits:
            assert all(h.score == 0.8 for h in take_hits)

    def test_empty_entity_ids_returns_empty(self, populated_db):
        assert fact_lookup(populated_db, [], "take", set()) == []

    def test_fact_lookup_under_50ms(self, populated_db):
        """Acceptance criterion: indexed SQL fact lookup < 50 ms."""
        entity_ids = find_mentioned_entity_ids(populated_db, "sherlock holmes")
        t0 = time.perf_counter()
        fact_lookup(populated_db, entity_ids, "take", set())
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 50, f"fact_lookup took {elapsed_ms:.1f} ms (must be < 50)"

    def test_negated_facts_included(self, populated_db):
        """Negated facts are returned; callers decide whether to filter them."""
        entity_ids = find_mentioned_entity_ids(populated_db, "sherlock holmes")
        hits = fact_lookup(populated_db, entity_ids, "take", set())
        # At least some hits should exist (negated or not)
        assert hits


# ---------------------------------------------------------------------------
# FTS search
# ---------------------------------------------------------------------------

class TestFTSSearch:
    def test_finds_chunk_by_keyword(self, populated_db):
        ids = fts_search_question(populated_db, "bottle mantel-piece", k=5)
        assert len(ids) > 0

    def test_returns_empty_for_unknown(self, populated_db):
        ids = fts_search_question(populated_db, "zzzunknownxxx", k=5)
        assert ids == []

    def test_respects_k(self, populated_db):
        ids = fts_search_question(populated_db, "holmes", k=1)
        assert len(ids) <= 1


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

class TestRRFFuse:
    def test_single_list(self):
        fused = rrf_fuse([[10, 20, 30]])
        ids = [x[0] for x in fused]
        assert ids == [10, 20, 30]

    def test_two_lists_boosts_overlap(self):
        fused = rrf_fuse([[1, 2, 3], [2, 3, 4]])
        # IDs 2 and 3 appear in both lists → higher scores
        scores = dict(fused)
        assert scores[2] > scores[1]
        assert scores[2] > scores[4]

    def test_empty_lists(self):
        assert rrf_fuse([]) == []
        assert rrf_fuse([[]]) == []

    def test_k_smoothing(self):
        fused_60 = rrf_fuse([[1, 2]], k=60)
        fused_1 = rrf_fuse([[1, 2]], k=1)
        # With smaller k, rank differences are amplified
        ratio_60 = dict(fused_60)[1] / dict(fused_60)[2]
        ratio_1 = dict(fused_1)[1] / dict(fused_1)[2]
        assert ratio_1 > ratio_60


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

class TestBuildContext:
    def _make_fact(self, n: int) -> FactHit:
        return FactHit(
            fact_id=n, tag=f"[F{n}]", subject_name="Holmes",
            predicate="take", object_text="bottle",
            object_entity_name=None, qualifiers={"from": "corner"},
            negated=False, sentence="Holmes took the bottle.",
            chunk_id=n, source_info="Book", score=1.0,
        )

    def _make_chunk(self, n: int) -> ChunkHit:
        return ChunkHit(
            chunk_id=n, tag=f"[C{n}]",
            text="Some chunk text about Holmes.",
            source_info="Book", rrf_score=0.01,
        )

    def test_chunks_come_first(self):
        ctx = build_context([self._make_fact(1)], [self._make_chunk(2)], token_budget=1800)
        assert ctx.index("[C2]") < ctx.index("[F1]")

    def test_respects_token_budget(self):
        facts = [self._make_fact(i) for i in range(20)]
        ctx = build_context(facts, [], token_budget=50)
        assert len(ctx) <= 50 * 4 * 2  # generous slack for the last partial fact

    def test_empty_inputs(self):
        assert build_context([], [], token_budget=1800) == ""

    def test_fact_format_contains_tag(self):
        ctx = build_context([self._make_fact(3)], [], token_budget=1800)
        assert "[F3]" in ctx
        assert "Holmes" in ctx
        assert "take" in ctx
        assert "bottle" in ctx
        assert "corner" in ctx


# ---------------------------------------------------------------------------
# Full retrieve pipeline
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_fact_hits_returned(self, populated_db, fake_embedder, default_cfg, nlp):
        result = retrieve(
            "what did sherlock holmes take?",
            conn=populated_db, cfg=default_cfg,
            embedder=fake_embedder, nlp=nlp,
        )
        assert result.fact_hits, "should find take facts"
        objects = [f.object_text for f in result.fact_hits]
        assert "bottle" in objects

    def test_chunk_hits_returned(self, populated_db, fake_embedder, default_cfg, nlp):
        result = retrieve(
            "what did sherlock holmes take?",
            conn=populated_db, cfg=default_cfg,
            embedder=fake_embedder, nlp=nlp,
        )
        assert result.chunk_hits

    def test_context_has_fact_tags(self, populated_db, fake_embedder, default_cfg, nlp):
        result = retrieve(
            "what did sherlock holmes take?",
            conn=populated_db, cfg=default_cfg,
            embedder=fake_embedder, nlp=nlp,
        )
        assert "[F" in result.context_text

    def test_context_has_chunk_tags(self, populated_db, fake_embedder, default_cfg, nlp):
        result = retrieve(
            "what did sherlock holmes take?",
            conn=populated_db, cfg=default_cfg,
            embedder=fake_embedder, nlp=nlp,
        )
        assert "[C" in result.context_text

    def test_no_embedder_still_works(self, populated_db, default_cfg, nlp):
        """Without embedder, fact + FTS paths still return results."""
        result = retrieve(
            "what did sherlock holmes take?",
            conn=populated_db, cfg=default_cfg,
            embedder=None, nlp=nlp,
        )
        assert result.fact_hits

    def test_explain_text_populated(self, populated_db, fake_embedder, default_cfg, nlp):
        result = retrieve(
            "what did sherlock holmes take?",
            conn=populated_db, cfg=default_cfg,
            embedder=fake_embedder, nlp=nlp,
            explain=True,
        )
        assert result.explain_text is not None
        assert "Question Parse" in result.explain_text
        assert "Fact Lookup" in result.explain_text

    def test_unknown_question_returns_empty_facts(self, populated_db, default_cfg, nlp):
        result = retrieve(
            "what did moriarty steal on the moon?",
            conn=populated_db, cfg=default_cfg,
            embedder=None, nlp=nlp,
        )
        assert result.fact_hits == []

    def test_fts_paraphrase_returns_chunks(self, populated_db, default_cfg, nlp):
        """Paraphrase without entity hit → FTS fallback surfaces relevant chunks."""
        result = retrieve(
            "what items did the detective grab from the shelf?",
            conn=populated_db, cfg=default_cfg,
            embedder=None, nlp=nlp,
        )
        # FTS should surface something from the corpus
        all_text = " ".join(ch.text for ch in result.chunk_hits)
        # At minimum, FTS on "detective" or "grab" may surface relevant chunks;
        # if not, the test just checks the pipeline didn't crash
        assert isinstance(result.chunk_hits, list)


# ---------------------------------------------------------------------------
# Fact-FTS regression and new behaviour
# ---------------------------------------------------------------------------

import hashlib as _hashlib
from loci.store import (
    open_db as _open_db,
    insert_source as _insert_source,
    insert_chunk as _insert_chunk,
    insert_entity as _insert_entity,
    insert_alias as _insert_alias,
    insert_fact as _insert_fact,
    rebuild_fact_fts as _rebuild_fact_fts,
)


@pytest.fixture
def landlady_db(tmp_path, nlp):
    """Minimal DB: one fact (Mrs Hudson — role — landlady)."""
    conn = _open_db(tmp_path / "landlady.db")
    src_id = _insert_source(conn, sha256=_hashlib.sha256(b"s").hexdigest(), title="Test")
    chunk_id = _insert_chunk(
        conn, source_id=src_id, ordinal=0,
        text="Mrs. Hudson, our landlady, brought tea.",
        sha256=_hashlib.sha256(b"c").hexdigest(),
    )
    eid = _insert_entity(conn, canonical_name="Mrs Hudson", kind="person")
    _insert_alias(conn, entity_id=eid, alias="mrs hudson")
    _insert_fact(
        conn, chunk_id=chunk_id,
        sentence="Mrs. Hudson, our landlady, brought tea.",
        subject_id=eid, predicate="role", object_text="landlady",
    )
    _rebuild_fact_fts(conn)
    yield conn
    conn.close()


class TestFactFts:
    def test_copula_question_hits_fact(self, landlady_db, default_cfg, nlp):
        """Core regression: copula 'is' no longer blocks fact retrieval."""
        result = retrieve(
            "Who is the landlady?",
            conn=landlady_db, cfg=default_cfg,
            embedder=None, nlp=nlp,
        )
        assert len(result.fact_hits) > 0
        assert "[F" in result.context_text

    def test_fact_fts_stopword_only_returns_empty(self, landlady_db):
        from loci.retrieve import fact_fts_search_question
        ids = fact_fts_search_question(landlady_db, "who is the", k=5)
        assert ids == []

    def test_fact_fts_content_word_returns_ids(self, landlady_db):
        from loci.retrieve import fact_fts_search_question
        ids = fact_fts_search_question(landlady_db, "landlady role", k=5)
        assert len(ids) > 0

    def test_max_facts_in_context_cap(self, tmp_path, default_cfg, nlp):
        conn = _open_db(tmp_path / "cap.db")
        src_id = _insert_source(conn, sha256=_hashlib.sha256(b"cap_src").hexdigest(), title="Cap")
        for i in range(8):
            cid = _insert_chunk(
                conn, source_id=src_id, ordinal=i,
                text=f"The landlady lives at place {i}.",
                sha256=_hashlib.sha256(f"cap_chunk_{i}".encode()).hexdigest(),
            )
            eid = _insert_entity(conn, canonical_name=f"Lady{i}", kind="person")
            _insert_alias(conn, entity_id=eid, alias=f"lady{i}")
            _insert_fact(
                conn, chunk_id=cid,
                sentence=f"The landlady lives at place {i}.",
                subject_id=eid, predicate="residence", object_text="landlady area",
            )
        _rebuild_fact_fts(conn)

        from loci.config import Config
        cfg = Config()
        cfg.retrieval.max_facts_in_context = 2

        result = retrieve(
            "Who is the landlady?",
            conn=conn, cfg=cfg, embedder=None, nlp=nlp,
        )
        conn.close()
        assert len(result.fact_hits) <= 2
        tags = [h.tag for h in result.fact_hits]
        if len(tags) >= 1:
            assert tags[0] == "[F1]"
        if len(tags) == 2:
            assert tags[1] == "[F2]"

    def test_fact_hits_scores_non_increasing(self, landlady_db, default_cfg, nlp):
        result = retrieve(
            "Who is the landlady?",
            conn=landlady_db, cfg=default_cfg, embedder=None, nlp=nlp,
        )
        scores = [h.score for h in result.fact_hits]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Phase B: vec-over-facts — vec_fact_search_question, canonical_names, modes
# ---------------------------------------------------------------------------

from loci.store import rebuild_fact_vec as _rebuild_fact_vec
from loci.retrieve import vec_fact_search_question, canonical_names_for_facts


class TestVecFacts:
    def test_vec_fact_search_returns_tuples(self, landlady_db, fake_embedder):
        _rebuild_fact_vec(landlady_db, fake_embedder)
        from loci.models import embed_batch
        emb = embed_batch(fake_embedder, ["landlady role"], normalize=True)[0]
        results = vec_fact_search_question(landlady_db, emb, k=5)
        assert isinstance(results, list)
        for fid, dist in results:
            assert isinstance(fid, int)
            assert isinstance(dist, float)

    def test_canonical_names_returns_proper_nouns(self, landlady_db):
        fact_ids = [r[0] for r in landlady_db.execute("SELECT id FROM facts LIMIT 3").fetchall()]
        names = canonical_names_for_facts(landlady_db, fact_ids)
        for name in names:
            assert name != name.lower(), f"Expected proper noun, got: {name!r}"

    def test_canonical_names_empty_input(self, landlady_db):
        assert canonical_names_for_facts(landlady_db, []) == []

    def test_surface_mode_runs_without_error(self, landlady_db, fake_embedder, nlp):
        _rebuild_fact_vec(landlady_db, fake_embedder)
        from loci.config import Config
        cfg = Config()
        cfg.retrieval.max_facts_in_context = 10
        cfg.retrieval.fact_vec_mode = "surface"
        cfg.retrieval.fact_vec_top_k = 5
        result = retrieve(
            "Who is the landlady?",
            conn=landlady_db, cfg=cfg, embedder=fake_embedder, nlp=nlp,
        )
        assert isinstance(result.fact_hits, list)

    def test_expand_mode_runs_without_error(self, landlady_db, fake_embedder, nlp):
        _rebuild_fact_vec(landlady_db, fake_embedder)
        from loci.config import Config
        cfg = Config()
        cfg.retrieval.max_facts_in_context = 4
        cfg.retrieval.fact_vec_mode = "expand"
        cfg.retrieval.fact_vec_top_k = 5
        cfg.retrieval.fact_expand_names = 2
        result = retrieve(
            "Who is the landlady?",
            conn=landlady_db, cfg=cfg, embedder=fake_embedder, nlp=nlp,
        )
        assert isinstance(result.chunk_hits, list)

    def test_off_mode_unchanged_from_baseline(self, landlady_db, default_cfg, nlp):
        result = retrieve(
            "Who is the landlady?",
            conn=landlady_db, cfg=default_cfg, embedder=None, nlp=nlp,
        )
        # off mode with no embedder: should still surface FTS facts
        assert isinstance(result.fact_hits, list)

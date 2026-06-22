"""Tests for the benchmark & evaluation suite (§13)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from loci.bench import (
    JUDGE_RUBRIC,
    MechanicalScore,
    QnAItem,
    QuestionResult,
    _compute_aggregate,
    _split_payload,
    build_judge_prompt,
    compare_runs,
    load_qna,
    parse_judge_response,
    read_run_log,
    render_report,
    score_mechanical,
    write_run_log,
)
from loci.retrieve import ChunkHit, FactHit

SPEC_TEXT = (
    "Sherlock Holmes took his bottle from the corner of the mantel-piece. "
    "Holmes also took his syringe from its neat morocco case. "
    "Watson entered the room and observed Holmes carefully."
)


# ---------------------------------------------------------------------------
# QnAItem
# ---------------------------------------------------------------------------

class TestQnAItem:
    def test_from_dict_full(self):
        d = {
            "id": "q001", "type": "fact",
            "question": "What did Holmes take?",
            "expected_keywords": ["bottle"],
            "expected_facts": [{"subject": "sherlock holmes", "predicate": "take"}],
            "expected_sources": ["spec.txt"],
            "answerable": True,
        }
        item = QnAItem.from_dict(d)
        assert item.id == "q001"
        assert item.type == "fact"
        assert item.expected_keywords == ["bottle"]
        assert item.answerable is True

    def test_from_dict_minimal(self):
        item = QnAItem.from_dict({"id": "q1", "question": "Who?"})
        assert item.type == "fact"
        assert item.expected_keywords == []
        assert item.expected_facts == []
        assert item.answerable is True

    def test_to_dict_round_trip(self):
        d = {"id": "q1", "type": "negative", "question": "X?",
             "expected_keywords": [], "expected_facts": [],
             "expected_sources": [], "answerable": False,
             "expected_answer": None, "book": None}
        assert QnAItem.from_dict(d).to_dict() == d

    def test_expected_answer_loaded(self):
        d = {"id": "q1", "type": "fact", "question": "Who?",
             "expected_keywords": ["Stamford"],
             "expected_facts": [], "expected_sources": [],
             "answerable": True, "expected_answer": "Stamford"}
        item = QnAItem.from_dict(d)
        assert item.expected_answer == "Stamford"

    def test_expected_answer_absent_for_negative(self):
        d = {"id": "q1", "type": "negative", "question": "X?",
             "expected_keywords": [], "expected_facts": [],
             "expected_sources": [], "answerable": False}
        item = QnAItem.from_dict(d)
        assert item.expected_answer is None

    def test_load_qna_from_file(self, tmp_path):
        p = tmp_path / "qna.json"
        p.write_text(json.dumps([
            {"id": "q001", "type": "fact", "question": "Who?",
             "expected_keywords": [], "expected_facts": [],
             "expected_sources": [], "answerable": True},
        ]))
        items = load_qna(p)
        assert len(items) == 1
        assert items[0].id == "q001"

    def test_real_qna_file_loads(self):
        qna_path = Path(__file__).parent.parent / "bench" / "qna.json"
        if not qna_path.exists():
            pytest.skip("bench/qna.json not present")
        items = load_qna(qna_path)
        assert len(items) >= 4
        types = {i.type for i in items}
        assert "fact" in types
        assert "negative" in types


# ---------------------------------------------------------------------------
# Mechanical scoring helpers
# ---------------------------------------------------------------------------

def _make_fact_hit(subject: str, predicate: str) -> FactHit:
    return FactHit(
        fact_id=1, tag="[F1]", subject_name=subject, predicate=predicate,
        object_text="bottle", object_entity_name=None, qualifiers=None,
        negated=False, sentence="Holmes took the bottle.", chunk_id=1,
        source_info="spec.txt", score=1.0,
    )


def _make_chunk_hit(source_info: str) -> ChunkHit:
    return ChunkHit(chunk_id=1, tag="[C1]", text="Some text.",
                    source_info=source_info, rrf_score=0.01)


class TestScoreMechanical:
    def _item(self, **kw) -> QnAItem:
        defaults = {"id": "q1", "type": "fact", "question": "?",
                    "expected_keywords": [], "expected_facts": [],
                    "expected_sources": [], "answerable": True}
        defaults.update(kw)
        return QnAItem.from_dict(defaults)

    def test_keyword_recall_full(self):
        item = self._item(expected_keywords=["bottle", "syringe"])
        s = score_mechanical(item, "Holmes took the bottle and syringe.", [], [])
        assert s.keyword_recall == pytest.approx(1.0)

    def test_keyword_recall_partial(self):
        item = self._item(expected_keywords=["bottle", "syringe"])
        s = score_mechanical(item, "Holmes took the bottle.", [], [])
        assert s.keyword_recall == pytest.approx(0.5)

    def test_keyword_recall_zero(self):
        item = self._item(expected_keywords=["bomb"])
        s = score_mechanical(item, "Holmes took the bottle.", [], [])
        assert s.keyword_recall == pytest.approx(0.0)

    def test_keyword_recall_case_insensitive(self):
        item = self._item(expected_keywords=["Bottle"])
        s = score_mechanical(item, "holmes took the bottle.", [], [])
        assert s.keyword_recall == pytest.approx(1.0)

    def test_no_expected_keywords_is_1(self):
        item = self._item(expected_keywords=[])
        s = score_mechanical(item, "any answer", [], [])
        assert s.keyword_recall == pytest.approx(1.0)

    def test_citation_present_fact_tag(self):
        item = self._item()
        s = score_mechanical(item, "Holmes took the bottle [F1].", [], [])
        assert s.citation_present is True

    def test_citation_present_chunk_tag(self):
        item = self._item()
        s = score_mechanical(item, "See [C3] for context.", [], [])
        assert s.citation_present is True

    def test_citation_absent(self):
        item = self._item()
        s = score_mechanical(item, "Holmes took the bottle.", [], [])
        assert s.citation_present is False

    def test_fact_hit_rate_full_match(self):
        item = self._item(expected_facts=[{"subject": "sherlock holmes", "predicate": "take"}])
        fh = _make_fact_hit("Sherlock Holmes", "take")
        s = score_mechanical(item, "answer", [fh], [])
        assert s.fact_hit_rate == pytest.approx(1.0)

    def test_fact_hit_rate_no_match(self):
        item = self._item(expected_facts=[{"subject": "watson", "predicate": "enter"}])
        fh = _make_fact_hit("Sherlock Holmes", "take")
        s = score_mechanical(item, "answer", [fh], [])
        assert s.fact_hit_rate == pytest.approx(0.0)

    def test_fact_hit_rate_partial(self):
        item = self._item(expected_facts=[
            {"subject": "sherlock holmes", "predicate": "take"},
            {"subject": "watson", "predicate": "enter"},
        ])
        fh = _make_fact_hit("Sherlock Holmes", "take")
        s = score_mechanical(item, "answer", [fh], [])
        assert s.fact_hit_rate == pytest.approx(0.5)

    def test_no_expected_facts_is_1(self):
        item = self._item(expected_facts=[])
        s = score_mechanical(item, "answer", [], [])
        assert s.fact_hit_rate == pytest.approx(1.0)

    def test_retrieval_hit_rate(self):
        item = self._item(expected_sources=["spec.txt"])
        ch = _make_chunk_hit("spec.txt")
        s = score_mechanical(item, "answer", [], [ch])
        assert s.retrieval_hit_rate == pytest.approx(1.0)

    def test_retrieval_hit_rate_miss(self):
        item = self._item(expected_sources=["other.txt"])
        ch = _make_chunk_hit("spec.txt")
        s = score_mechanical(item, "answer", [], [ch])
        assert s.retrieval_hit_rate == pytest.approx(0.0)

    def test_hallucination_true(self):
        item = self._item(answerable=False)
        s = score_mechanical(item, "Holmes ate eggs.", [], [])
        assert s.hallucination is True

    def test_hallucination_false_when_refused(self):
        item = self._item(answerable=False)
        s = score_mechanical(item, "Not in my knowledge base.", [], [])
        assert s.hallucination is False

    def test_hallucination_false_for_answerable(self):
        item = self._item(answerable=True)
        s = score_mechanical(item, "Holmes took the bottle.", [], [])
        assert s.hallucination is False


# ---------------------------------------------------------------------------
# parse_judge_response
# ---------------------------------------------------------------------------

class TestParseJudgeResponse:
    def _ids(self, *ids):
        return set(ids)

    def test_valid_json(self):
        raw = '[{"id":"q001","score":85,"reason":"correct"}]'
        result = parse_judge_response(raw, {"q001"})
        assert len(result) == 1
        assert result[0]["score"] == 85
        assert result[0]["reason"] == "correct"

    def test_strips_markdown_fences(self):
        raw = '```json\n[{"id":"q1","score":70,"reason":"ok"}]\n```'
        result = parse_judge_response(raw, {"q1"})
        assert len(result) == 1

    def test_id_mismatch_returns_empty(self):
        raw = '[{"id":"q001","score":85,"reason":"x"}]'
        assert parse_judge_response(raw, {"q002"}) == []

    def test_malformed_json_returns_empty(self):
        assert parse_judge_response("not json", {"q1"}) == []

    def test_score_clamped_to_100(self):
        raw = '[{"id":"q1","score":150,"reason":"x"}]'
        result = parse_judge_response(raw, {"q1"})
        assert result[0]["score"] == 100

    def test_score_clamped_to_0(self):
        raw = '[{"id":"q1","score":-10,"reason":"x"}]'
        result = parse_judge_response(raw, {"q1"})
        assert result[0]["score"] == 0

    def test_missing_score_filtered(self):
        raw = '[{"id":"q1","reason":"x"}]'
        assert parse_judge_response(raw, {"q1"}) == []

    def test_multiple_items(self):
        raw = json.dumps([
            {"id": "q1", "score": 90, "reason": "a"},
            {"id": "q2", "score": 60, "reason": "b"},
        ])
        result = parse_judge_response(raw, {"q1", "q2"})
        assert len(result) == 2

    def test_non_list_returns_empty(self):
        raw = '{"id":"q1","score":50}'
        assert parse_judge_response(raw, {"q1"}) == []


# ---------------------------------------------------------------------------
# build_judge_prompt
# ---------------------------------------------------------------------------

class TestBuildJudgePrompt:
    def test_contains_rubric(self):
        prompt = build_judge_prompt([])
        assert "100" in prompt
        assert "answerable=false" in prompt

    def test_contains_payload(self):
        payload = [{"id": "q1", "question": "Who?", "system_answer": "Holmes"}]
        prompt = build_judge_prompt(payload)
        assert "Who?" in prompt
        assert "Holmes" in prompt

    def test_expected_answer_in_prompt(self):
        """Judge prompt must contain expected_answer when present — this is the answer key."""
        payload = [{"id": "q1", "question": "Who introduced Watson to Holmes?",
                    "system_answer": "Stamford did it.",
                    "answerable": True, "expected_answer": "Stamford"}]
        prompt = build_judge_prompt(payload)
        assert "Stamford" in prompt

    def test_rubric_mentions_expected_answer(self):
        """Rubric must instruct the judge to use expected_answer when present."""
        prompt = build_judge_prompt([])
        assert "expected_answer" in prompt


# ---------------------------------------------------------------------------
# _split_payload
# ---------------------------------------------------------------------------

class TestSplitPayload:
    def test_single_chunk_when_small(self):
        payload = [{"id": "q1", "question": "short"}]
        chunks = _split_payload(payload, max_chars=10000)
        assert chunks == [payload]

    def test_splits_when_large(self):
        payload = [{"id": f"q{i}", "answer": "x" * 2000} for i in range(5)]
        chunks = _split_payload(payload, max_chars=5000, prefix_len=100)
        assert len(chunks) > 1
        all_items = [item for chunk in chunks for item in chunk]
        assert len(all_items) == 5


# ---------------------------------------------------------------------------
# write_run_log / read_run_log
# ---------------------------------------------------------------------------

def _make_result(q_id: str, q_type: str = "fact", run_index: int = 0,
                 answerable: bool = True, judge_score: int | None = None) -> QuestionResult:
    return QuestionResult(
        q_id=q_id, q_type=q_type, question=f"Q {q_id}?", answerable=answerable,
        answer="Holmes took the bottle [F1].", citations=["[F1]"],
        mechanical=MechanicalScore(
            citation_present=True, citation_valid=True, keyword_recall=1.0,
            fact_hit_rate=1.0, retrieval_hit_rate=1.0, hallucination=False,
        ),
        judge_score=judge_score, judge_reason="good" if judge_score else None,
        timings={"parse_ms": 1.0, "fact_ms": 5.0, "vec_ms": 20.0,
                 "fts_ms": 3.0, "fusion_ms": 2.0, "gen_ms": 500.0,
                 "ttft_ms": 120.0, "tokens": 15},
        peak_rss_mb=2100.0, swap_delta_mb=0.0, run_index=run_index,
    )


class TestRunLog:
    def test_write_creates_file(self, tmp_path):
        r = _make_result("q001")
        path = write_run_log([r], {}, "test", tmp_path)
        assert path.exists()

    def test_round_trip(self, tmp_path):
        results = [_make_result("q001"), _make_result("q002")]
        path = write_run_log(results, {"models": {"chat": "model.gguf"}}, "rt", tmp_path)
        data = read_run_log(path)
        assert data["config"]["label"] == "rt"
        assert len(data["results"]) == 2
        assert data["results"][0]["q_id"] == "q001"

    def test_aggregate_in_log(self, tmp_path):
        results = [_make_result("q001", judge_score=80),
                   _make_result("q002", judge_score=90)]
        path = write_run_log(results, {}, "agg", tmp_path)
        data = read_run_log(path)
        assert "mean_keyword_recall" in data["aggregate"]
        assert "mean_judge_score" in data["aggregate"]
        assert data["aggregate"]["mean_judge_score"] == pytest.approx(85.0)

    def test_label_sanitised_in_filename(self, tmp_path):
        path = write_run_log([], {}, "my run/test", tmp_path)
        assert "/" not in path.name

    def test_hallucination_count_in_aggregate(self, tmp_path):
        good = _make_result("q001", answerable=True)
        bad = _make_result("q002", answerable=False)
        bad.mechanical.hallucination = True
        path = write_run_log([good, bad], {}, "hall", tmp_path)
        data = read_run_log(path)
        assert data["aggregate"]["hallucination_count"] == 1


# ---------------------------------------------------------------------------
# render_report
# ---------------------------------------------------------------------------

class TestRenderReport:
    def test_contains_question_id(self):
        r = _make_result("q001")
        run_data = {
            "config": {"label": "test", "ts": int(time.time())},
            "results": [r.to_dict()],
            "aggregate": {},
        }
        report = render_report(run_data)
        assert "q001" in report

    def test_hallucination_flagged(self):
        r = _make_result("q001", answerable=False)
        r.mechanical.hallucination = True
        run_data = {
            "config": {"label": "test", "ts": 0},
            "results": [r.to_dict()],
            "aggregate": {},
        }
        assert "!" in render_report(run_data)

    def test_shows_aggregate_section(self):
        run_data = {
            "config": {"label": "x", "ts": 0},
            "results": [],
            "aggregate": {"mean_keyword_recall": 0.85, "mean_judge_score": 78.0},
        }
        report = render_report(run_data)
        assert "Aggregate" in report
        assert "0.850" in report


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------

class TestCompareRuns:
    def _run(self, vec_top_k: int, mean_kw: float, mean_judge: float | None = None) -> dict:
        agg: dict = {
            "mean_keyword_recall": mean_kw,
            "mean_fact_hit_rate": 0.8,
            "mean_retrieval_hit_rate": 0.9,
            "citation_present_rate": 1.0,
            "hallucination_count": 0,
            "mean_gen_ms": 400.0,
            "mean_fact_ms": 5.0,
            "mean_peak_rss_mb": 2000.0,
        }
        if mean_judge is not None:
            agg["mean_judge_score"] = mean_judge
        return {
            "config": {
                "config": {"retrieval": {"vec_top_k": str(vec_top_k)}},
                "label": "run",
            },
            "aggregate": agg,
        }

    def test_shows_changed_config(self):
        out = compare_runs(self._run(12, 0.7), self._run(24, 0.8))
        assert "vec_top_k" in out
        assert "12" in out
        assert "24" in out

    def test_shows_positive_delta(self):
        out = compare_runs(self._run(12, 0.7), self._run(12, 0.9))
        assert "+0.200" in out or "better" in out.lower()

    def test_shows_regression(self):
        out = compare_runs(self._run(12, 0.9), self._run(12, 0.7))
        assert "REGRESS" in out

    def test_judge_score_delta(self):
        out = compare_runs(self._run(12, 0.8, 70.0), self._run(12, 0.8, 85.0))
        assert "mean_judge_score" in out


# ---------------------------------------------------------------------------
# retrieve() timings parameter
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def timings_db(tmp_path_factory, fake_embedder, nlp, default_cfg):
    tmp = tmp_path_factory.mktemp("timings")
    from loci.store import open_db
    from loci.ingest import ingest_file
    conn = open_db(tmp / "t.db")
    p = tmp / "spec.txt"
    p.write_text(SPEC_TEXT)
    ingest_file(p, cfg=default_cfg, conn=conn, embedder=fake_embedder, spacy_nlp=nlp)
    yield conn
    conn.close()


class TestRetrieveTimings:
    def test_timings_populated(self, timings_db, fake_embedder, nlp, default_cfg):
        from loci.retrieve import retrieve
        timings = {}
        retrieve(
            "what did sherlock holmes take?",
            conn=timings_db, cfg=default_cfg,
            embedder=fake_embedder, nlp=nlp,
            timings=timings,
        )
        assert "parse_ms" in timings
        assert "fact_ms" in timings
        assert "vec_ms" in timings
        assert "fts_ms" in timings
        assert "fusion_ms" in timings

    def test_timings_are_non_negative(self, timings_db, fake_embedder, nlp, default_cfg):
        from loci.retrieve import retrieve
        timings = {}
        retrieve(
            "what did sherlock holmes take?",
            conn=timings_db, cfg=default_cfg,
            embedder=fake_embedder, nlp=nlp,
            timings=timings,
        )
        for k, v in timings.items():
            assert v >= 0, f"{k} = {v}"

    def test_no_timings_when_none(self, timings_db, fake_embedder, nlp, default_cfg):
        from loci.retrieve import retrieve
        result = retrieve(
            "what did sherlock holmes take?",
            conn=timings_db, cfg=default_cfg,
            embedder=fake_embedder, nlp=nlp,
        )
        assert result.fact_hits  # still works


# ---------------------------------------------------------------------------
# measure() exposes peak RSS
# ---------------------------------------------------------------------------

class TestMeasurePeakRss:
    def test_peak_rss_accessible_after_context(self):
        from loci.bench import measure
        with measure("test_rss", silent=True) as c:
            _ = list(range(10_000))
        assert "_peak_rss_mb" in c
        assert c["_peak_rss_mb"] > 0

    def test_swap_delta_accessible(self):
        from loci.bench import measure
        with measure("test_swap", silent=True) as c:
            pass
        assert "_swap_delta_mb" in c


# ---------------------------------------------------------------------------
# ingest sentences_skipped
# ---------------------------------------------------------------------------

class TestIngestSentencesSkipped:
    def test_returns_sentences_counters(self, tmp_path, fake_embedder, nlp, default_cfg):
        from loci.store import open_db
        from loci.ingest import ingest_file
        conn = open_db(tmp_path / "sk.db")
        p = tmp_path / "sk.txt"
        # Include a pronoun sentence that will be skipped by standard SVO
        p.write_text(
            "Sherlock Holmes took his bottle. He examined it carefully. "
            "Watson entered the room."
        )
        stats = ingest_file(p, cfg=default_cfg, conn=conn,
                            embedder=fake_embedder, spacy_nlp=nlp)
        conn.close()
        assert "sentences_total" in stats
        assert "sentences_skipped" in stats
        assert stats["sentences_total"] >= 1

    def test_skipped_less_than_total(self, tmp_path, fake_embedder, nlp, default_cfg):
        from loci.store import open_db
        from loci.ingest import ingest_file
        conn = open_db(tmp_path / "sk2.db")
        p = tmp_path / "sk2.txt"
        p.write_text("Sherlock Holmes took the bottle. Watson entered the room.")
        stats = ingest_file(p, cfg=default_cfg, conn=conn,
                            embedder=fake_embedder, spacy_nlp=nlp)
        conn.close()
        assert stats["sentences_skipped"] <= stats["sentences_total"]

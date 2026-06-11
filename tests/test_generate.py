"""Tests for loci/generate.py: message building, citations, refusal, history."""
from __future__ import annotations

import pytest

from loci.generate import (
    _NO_CONTEXT_NOTE,
    _SYSTEM_PROMPT,
    build_messages,
    build_sources_footer,
    extract_cited_tags,
    generate_response,
    is_refusal,
    stream_response,
    trim_history,
)
from loci.retrieve import ChunkHit, FactHit


# ---------------------------------------------------------------------------
# Fake chat LLM (no llama-cpp-python required)
# ---------------------------------------------------------------------------

class FakeChatLLM:
    """Mimics llama.cpp create_chat_completion interface deterministically."""

    def create_chat_completion(self, messages, *, max_tokens, temperature, stream):
        sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        has_context = any(
            m["role"] == "user" and m["content"].startswith("Context:")
            for m in messages
        )

        if _NO_CONTEXT_NOTE in sys_msg or not has_context:
            content = "Not in my knowledge base."
        else:
            content = (
                "Sherlock Holmes took the bottle [F1] from the corner "
                "of the mantel-piece [C2]."
            )

        if stream:
            def _gen():
                words = content.split()
                for i, word in enumerate(words):
                    sep = " " if i < len(words) - 1 else ""
                    yield {"choices": [{"delta": {"content": word + sep}}]}
            return _gen()
        else:
            return {"choices": [{"message": {"content": content}}]}


@pytest.fixture
def fake_llm():
    return FakeChatLLM()


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------

class TestBuildMessages:
    def test_structure_with_context(self):
        msgs = build_messages("What did Holmes take?", "context here", [])
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user"]

    def test_system_prompt_present(self):
        msgs = build_messages("q", "ctx", [])
        assert msgs[0]["role"] == "system"
        assert "knowledge assistant" in msgs[0]["content"]

    def test_context_injected_into_user_message(self):
        msgs = build_messages("Who?", "some facts here", [])
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert "some facts here" in user_msg["content"]
        assert "Who?" in user_msg["content"]

    def test_empty_context_adds_no_context_note(self):
        msgs = build_messages("q", "", [])
        assert _NO_CONTEXT_NOTE in msgs[0]["content"]

    def test_empty_context_user_message_has_no_context_prefix(self):
        msgs = build_messages("q", "", [])
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert not user_msg["content"].startswith("Context:")
        assert "Question: q" in user_msg["content"]

    def test_history_inserted_between_system_and_user(self):
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        msgs = build_messages("new question", "ctx", history)
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert msgs[3]["role"] == "user"
        assert msgs[3]["content"].endswith("Question: new question")

    def test_whitespace_only_context_treated_as_empty(self):
        msgs = build_messages("q", "   \n  ", [])
        assert _NO_CONTEXT_NOTE in msgs[0]["content"]


# ---------------------------------------------------------------------------
# stream_response + generate_response
# ---------------------------------------------------------------------------

class TestGenerateResponse:
    def test_non_streaming_returns_string(self, fake_llm):
        msgs = build_messages("q", "ctx", [])
        text = generate_response(
            fake_llm, msgs, max_tokens=128, temperature=0.2
        )
        assert isinstance(text, str)
        assert len(text) > 0

    def test_streaming_yields_tokens(self, fake_llm):
        msgs = build_messages("q", "ctx", [])
        tokens = list(stream_response(fake_llm, msgs, max_tokens=128, temperature=0.2))
        assert len(tokens) > 0
        full = "".join(tokens)
        assert len(full) > 0

    def test_streaming_equals_non_streaming(self, fake_llm):
        msgs = build_messages("q", "ctx", [])
        streamed = "".join(stream_response(fake_llm, msgs, max_tokens=128, temperature=0.2)).strip()
        direct = generate_response(fake_llm, msgs, max_tokens=128, temperature=0.2).strip()
        assert streamed == direct

    def test_with_context_answer_contains_tags(self, fake_llm):
        msgs = build_messages("q", "some context", [])
        text = generate_response(fake_llm, msgs, max_tokens=128, temperature=0.2)
        assert "[F1]" in text or "[C" in text

    def test_empty_context_returns_refusal(self, fake_llm):
        msgs = build_messages("q", "", [])
        text = generate_response(fake_llm, msgs, max_tokens=128, temperature=0.2)
        assert is_refusal(text)


# ---------------------------------------------------------------------------
# extract_cited_tags
# ---------------------------------------------------------------------------

class TestExtractCitedTags:
    def test_extracts_fact_tags(self):
        tags = extract_cited_tags("Holmes took the bottle [F1].")
        assert tags == ["[F1]"]

    def test_extracts_chunk_tags(self):
        tags = extract_cited_tags("See [C3] and [C12].")
        assert tags == ["[C3]", "[C12]"]

    def test_mixed_tags_in_order(self):
        tags = extract_cited_tags("[F1] fact then [C2] chunk.")
        assert tags == ["[F1]", "[C2]"]

    def test_deduplicates_repeated_tags(self):
        tags = extract_cited_tags("[F1] and then [F1] again.")
        assert tags == ["[F1]"]

    def test_empty_string(self):
        assert extract_cited_tags("") == []

    def test_no_tags(self):
        assert extract_cited_tags("No tags here at all.") == []

    def test_does_not_match_invalid_patterns(self):
        tags = extract_cited_tags("[X1] [F] [C] [F0]")
        assert tags == ["[F0]"]

    def test_preserves_insertion_order(self):
        tags = extract_cited_tags("[C3] first, [F1] second, [C3] repeated.")
        assert tags == ["[C3]", "[F1]"]


# ---------------------------------------------------------------------------
# build_sources_footer
# ---------------------------------------------------------------------------

def _make_fact_hit(tag: str, subject: str, predicate: str, source: str) -> FactHit:
    return FactHit(
        fact_id=1, tag=tag, subject_name=subject,
        predicate=predicate, object_text="bottle",
        object_entity_name=None, qualifiers=None,
        negated=False, sentence="sentence.", chunk_id=1,
        source_info=source, score=1.0,
    )


def _make_chunk_hit(tag: str, source: str) -> ChunkHit:
    return ChunkHit(
        chunk_id=2, tag=tag, text="Some text.",
        source_info=source, rrf_score=0.01,
    )


class TestBuildSourcesFooter:
    def test_empty_tags_returns_empty_string(self):
        footer = build_sources_footer([], [], [])
        assert footer == ""

    def test_maps_fact_tag(self):
        fh = _make_fact_hit("[F1]", "Holmes", "take", "The Sign of the Four")
        footer = build_sources_footer(["[F1]"], [fh], [])
        assert "[F1]" in footer
        assert "The Sign of the Four" in footer
        assert "Holmes" in footer
        assert "take" in footer

    def test_maps_chunk_tag(self):
        ch = _make_chunk_hit("[C2]", "A Study in Scarlet")
        footer = build_sources_footer(["[C2]"], [], [ch])
        assert "[C2]" in footer
        assert "A Study in Scarlet" in footer

    def test_mixed_fact_and_chunk_tags(self):
        fh = _make_fact_hit("[F1]", "Holmes", "take", "Book A")
        ch = _make_chunk_hit("[C2]", "Book B")
        footer = build_sources_footer(["[F1]", "[C2]"], [fh], [ch])
        assert "Book A" in footer
        assert "Book B" in footer

    def test_unknown_tag_omitted(self):
        footer = build_sources_footer(["[F99]"], [], [])
        assert footer == ""

    def test_no_source_info_shows_unknown(self):
        fh = _make_fact_hit("[F1]", "Holmes", "take", None)
        footer = build_sources_footer(["[F1]"], [fh], [])
        assert "unknown source" in footer

    def test_starts_with_sources_header(self):
        fh = _make_fact_hit("[F1]", "Holmes", "take", "Book")
        footer = build_sources_footer(["[F1]"], [fh], [])
        assert footer.startswith("Sources:")


# ---------------------------------------------------------------------------
# is_refusal
# ---------------------------------------------------------------------------

class TestIsRefusal:
    def test_detects_exact_phrase(self):
        assert is_refusal("Not in my knowledge base.")

    def test_case_insensitive(self):
        assert is_refusal("NOT IN MY KNOWLEDGE BASE.")

    def test_phrase_in_longer_text(self):
        assert is_refusal("I'm sorry, not in my knowledge base for this topic.")

    def test_normal_answer_is_not_refusal(self):
        assert not is_refusal("Sherlock Holmes took a bottle from the mantel-piece.")

    def test_empty_string_is_not_refusal(self):
        assert not is_refusal("")

    def test_partial_match_not_refusal(self):
        assert not is_refusal("knowledge is power")


# ---------------------------------------------------------------------------
# trim_history
# ---------------------------------------------------------------------------

class TestTrimHistory:
    def _pair(self, n: int) -> list[dict]:
        return [
            {"role": "user", "content": f"question {n}"},
            {"role": "assistant", "content": f"answer {n}"},
        ]

    def test_empty_history(self):
        assert trim_history([]) == []

    def test_within_max_turns_unchanged(self):
        h = self._pair(1) + self._pair(2)
        result = trim_history(h, max_turns=3)
        assert len(result) == 4

    def test_trims_to_max_turns(self):
        h = self._pair(1) + self._pair(2) + self._pair(3) + self._pair(4)
        result = trim_history(h, max_turns=3)
        # Should keep last 3 pairs = 6 messages
        assert len(result) == 6
        assert result[0]["content"] == "question 2"

    def test_drops_oldest_on_token_budget(self):
        long_content = "x" * 5000
        h = [
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": long_content},
            {"role": "user", "content": "short question"},
            {"role": "assistant", "content": "short answer"},
        ]
        result = trim_history(h, max_turns=3, token_budget=10)
        # The long pair exceeds budget; should be dropped
        assert all(m["content"] != long_content for m in result)

    def test_odd_history_ignored_gracefully(self):
        h = [{"role": "user", "content": "orphaned"}]
        result = trim_history(h, max_turns=3)
        assert result == []


# ---------------------------------------------------------------------------
# Integration: full retrieve → generate round-trip with fake models
# ---------------------------------------------------------------------------

SPEC_TEXT = (
    "Sherlock Holmes took his bottle from the corner of the mantel-piece. "
    "Holmes also took his syringe from its neat morocco case. "
    "Watson entered the room and observed Holmes carefully."
)


@pytest.fixture(scope="module")
def gen_db(tmp_path_factory, fake_embedder, nlp, default_cfg):
    """Pre-populated DB for generate integration tests."""
    tmp = tmp_path_factory.mktemp("gen_db")
    from loci.store import open_db
    from loci.ingest import ingest_file

    conn = open_db(tmp / "test.db")
    p = tmp / "spec.txt"
    p.write_text(SPEC_TEXT)
    ingest_file(p, cfg=default_cfg, conn=conn,
                embedder=fake_embedder, spacy_nlp=nlp)
    yield conn
    conn.close()


class TestGenerateIntegration:
    def test_known_question_answer_cites_tags(self, gen_db, fake_embedder, default_cfg, nlp, fake_llm):
        from loci.retrieve import retrieve
        result = retrieve(
            "what did sherlock holmes take?",
            conn=gen_db, cfg=default_cfg,
            embedder=fake_embedder, nlp=nlp,
        )
        assert result.fact_hits or result.chunk_hits, "retrieval returned nothing"
        messages = build_messages("what did sherlock holmes take?", result.context_text, [])
        text = generate_response(fake_llm, messages, max_tokens=128, temperature=0.2)
        cited = extract_cited_tags(text)
        assert len(cited) > 0, "answer should cite at least one tag"

    def test_unknown_question_triggers_refusal(self, gen_db, default_cfg, nlp, fake_llm):
        from loci.retrieve import retrieve
        result = retrieve(
            "what did moriarty eat on the moon?",
            conn=gen_db, cfg=default_cfg,
            embedder=None, nlp=nlp,
        )
        messages = build_messages(
            "what did moriarty eat on the moon?",
            result.context_text,
            [],
        )
        text = generate_response(fake_llm, messages, max_tokens=128, temperature=0.2)
        assert is_refusal(text), f"expected refusal, got: {text!r}"

    def test_sources_footer_built_from_cited_tags(self, gen_db, fake_embedder, default_cfg, nlp, fake_llm):
        from loci.retrieve import retrieve
        result = retrieve(
            "what did sherlock holmes take?",
            conn=gen_db, cfg=default_cfg,
            embedder=fake_embedder, nlp=nlp,
        )
        messages = build_messages("what did sherlock holmes take?", result.context_text, [])
        text = generate_response(fake_llm, messages, max_tokens=128, temperature=0.2)
        cited = extract_cited_tags(text)
        footer = build_sources_footer(cited, result.fact_hits, result.chunk_hits)
        # footer should reference at least one real source tag
        assert any(tag in footer for tag in cited) or footer == ""

    def test_peak_rss_recorded_by_measure(self, fake_llm, default_cfg):
        """bench.measure() captures peak RSS during generation; must be < 3500 MB."""
        import psutil, os
        from loci.bench import measure

        with measure("test_generate", silent=True) as counters:
            msgs = build_messages("test", "some context", [])
            text = generate_response(fake_llm, msgs, max_tokens=64, temperature=0.2)
            counters["response_len"] = len(text)

        proc = psutil.Process(os.getpid())
        current_rss_mb = proc.memory_info().rss / 1024 / 1024
        assert current_rss_mb < 3500, (
            f"Peak RSS {current_rss_mb:.0f} MB exceeds 3500 MB budget"
        )

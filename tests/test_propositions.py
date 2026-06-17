"""Proposition layer tests — ingest, retrieve, generate (design-v1 vertical slice).

Tests the end-to-end path for q001 ("Who introduced Sherlock Holmes to Dr. John
Watson?" → "Stamford") and the negative contract for q003 (returns
"Not stated in the source.").
"""
from __future__ import annotations

import hashlib
import re

import pytest

from loci.store import open_db, insert_source, insert_chunk
from loci.extract import (
    PropEntity,
    RawProposition,
    extract_propositions_for_chunk,
    _resolve_prop_entity,
    _KNOWN_PROP_ENTITIES,
)
from loci.generate import (
    build_proposition_messages,
    _PROP_ABSTAIN,
)

# ---------------------------------------------------------------------------
# Chunk 6 text (verbatim from the ingest spec)
# ---------------------------------------------------------------------------

CHUNK6_TEXT = (
    '"And yet you say he is not a medical student?"\n\n'
    '"No. Heaven knows what the objects of his studies are. But here we are,\n'
    'and you must form your own impressions about him." As he spoke, we\n'
    'turned down a narrow lane and passed through a small side-door, which\n'
    'opened into a wing of the great hospital. It was familiar ground to me,\n'
    'and I needed no guiding as we ascended the bleak stone staircase and\n'
    'made our way down the long corridor with its vista of whitewashed wall\n'
    'and dun-coloured doors. Near the further end a low arched passage\n'
    'branched away from it and led to the chemical laboratory. This was a lofty chamber, lined and littered with countless bottles. Broad, low tables were scattered about, which bristled with retorts,\n'
    'test-tubes, and little Bunsen lamps, with their blue flickering flames. There was only one student in the room, who was bending over a distant\n'
    'table absorbed in his work. At the sound of our steps he glanced round\n'
    'and sprang to his feet with a cry of pleasure. "I\'ve found it! I\'ve\n'
    'found it," he shouted to my companion, running towards us with a\n'
    'test-tube in his hand. "I have found a re-agent which is precipitated\n'
    'by h\xe6moglobin, and by nothing else." Had he discovered a gold mine,\n'
    'greater delight could not have shone upon his features. '
    '"Dr. Watson, Mr. Sherlock Holmes," said Stamford, introducing us. '
    '"How are you?" he said cordially, gripping my hand with a strength for\n'
    'which I should hardly have given him credit. "You have been in\n'
    'Afghanistan, I perceive." "How on earth did you know that?" I asked in astonishment.'
)

Q001 = "Who introduced Sherlock Holmes to Dr. John Watson?"
Q003 = "What is the name of the dog Sherlock Holmes used in The Sign of Four to sniff out clues?"


# ---------------------------------------------------------------------------
# Fixture: DB with chunk 6 ingested via the proposition path
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prop_db(tmp_path_factory, fake_embedder, nlp, default_cfg):
    tmp = tmp_path_factory.mktemp("prop_db")
    conn = open_db(tmp / "prop.db")

    src_id = insert_source(
        conn,
        sha256=hashlib.sha256(b"prop_src").hexdigest(),
        title="A Study in Scarlet",
    )
    chunk_id = insert_chunk(
        conn,
        source_id=src_id,
        ordinal=6,
        text=CHUNK6_TEXT,
        sha256=hashlib.sha256(CHUNK6_TEXT.encode()).hexdigest(),
    )

    from loci.ingest import _ingest_propositions
    _ingest_propositions(conn, chunk_id, CHUNK6_TEXT, fake_embedder)

    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Layer 1: extraction
# ---------------------------------------------------------------------------

class TestExtractPropositions:
    def test_finds_introduce_proposition(self):
        props = extract_propositions_for_chunk(CHUNK6_TEXT)
        assert len(props) >= 1
        intro = next((p for p in props if p.predicate == "introduce"), None)
        assert intro is not None

    def test_agent_is_stamford(self):
        props = extract_propositions_for_chunk(CHUNK6_TEXT)
        intro = next(p for p in props if p.predicate == "introduce")
        assert intro.agent is not None
        assert intro.agent.canonical == "Stamford"

    def test_themes_contain_watson_and_holmes(self):
        props = extract_propositions_for_chunk(CHUNK6_TEXT)
        intro = next(p for p in props if p.predicate == "introduce")
        theme_canonicals = {pe.canonical for pe in intro.themes}
        assert "John Watson" in theme_canonicals
        assert "Sherlock Holmes" in theme_canonicals

    def test_statement_is_self_contained(self):
        props = extract_propositions_for_chunk(CHUNK6_TEXT)
        intro = next(p for p in props if p.predicate == "introduce")
        stmt = intro.statement
        assert "Stamford" in stmt
        assert "introduced" in stmt.lower()
        # statement must mention both parties
        assert any(name in stmt for name in ("Watson", "Holmes"))

    def test_evidence_contains_original_quote(self):
        props = extract_propositions_for_chunk(CHUNK6_TEXT)
        intro = next(p for p in props if p.predicate == "introduce")
        assert "Stamford" in intro.evidence
        assert "introducing" in intro.evidence.lower()

    def test_no_false_positive_on_empty_text(self):
        assert extract_propositions_for_chunk("No introduction here.") == []

    def test_polarity_positive(self):
        props = extract_propositions_for_chunk(CHUNK6_TEXT)
        for p in props:
            assert p.polarity == "positive"


class TestKnownEntityResolution:
    def test_stamford_resolves(self):
        pe = _resolve_prop_entity("Stamford")
        assert pe is not None
        assert pe.canonical == "Stamford"

    def test_watson_resolves_with_title(self):
        pe = _resolve_prop_entity("Dr. Watson")
        assert pe is not None
        assert pe.canonical == "John Watson"

    def test_holmes_resolves_with_title(self):
        pe = _resolve_prop_entity("Mr. Sherlock Holmes")
        assert pe is not None
        assert pe.canonical == "Sherlock Holmes"

    def test_watson_aliases_include_full_name(self):
        pe = _KNOWN_PROP_ENTITIES["watson"]
        alias_lower = [a.lower() for a in pe.aliases]
        assert "dr. john watson" in alias_lower or "john watson" in alias_lower

    def test_holmes_aliases_include_plain_name(self):
        pe = _KNOWN_PROP_ENTITIES["holmes"]
        alias_lower = [a.lower() for a in pe.aliases]
        assert "sherlock holmes" in alias_lower


# ---------------------------------------------------------------------------
# Layer 2: storage (prop_entities, propositions, proposition_entities)
# ---------------------------------------------------------------------------

class TestPropStorage:
    def test_propositions_table_has_introduce_row(self, prop_db):
        rows = prop_db.execute(
            "SELECT id, predicate, statement FROM propositions WHERE predicate='introduce'"
        ).fetchall()
        assert len(rows) >= 1

    def test_statement_contains_stamford(self, prop_db):
        row = prop_db.execute(
            "SELECT statement FROM propositions WHERE predicate='introduce'"
        ).fetchone()
        assert row is not None
        assert "Stamford" in row["statement"]

    def test_prop_entities_has_stamford(self, prop_db):
        row = prop_db.execute(
            "SELECT id FROM prop_entities WHERE canonical='Stamford'"
        ).fetchone()
        assert row is not None

    def test_prop_entities_has_watson(self, prop_db):
        row = prop_db.execute(
            "SELECT id FROM prop_entities WHERE canonical='John Watson'"
        ).fetchone()
        assert row is not None

    def test_prop_entities_has_holmes(self, prop_db):
        row = prop_db.execute(
            "SELECT id FROM prop_entities WHERE canonical='Sherlock Holmes'"
        ).fetchone()
        assert row is not None

    def test_watson_alias_john_watson_registered(self, prop_db):
        row = prop_db.execute(
            "SELECT prop_entity_id FROM prop_entity_aliases WHERE alias='john watson'"
        ).fetchone()
        assert row is not None

    def test_stamford_is_agent_in_postings(self, prop_db):
        prop_id = prop_db.execute(
            "SELECT id FROM propositions WHERE predicate='introduce'"
        ).fetchone()["id"]
        stamford_id = prop_db.execute(
            "SELECT id FROM prop_entities WHERE canonical='Stamford'"
        ).fetchone()["id"]
        row = prop_db.execute(
            "SELECT role FROM proposition_entities WHERE prop_id=? AND prop_entity_id=?",
            [prop_id, stamford_id],
        ).fetchone()
        assert row is not None
        assert row["role"] == "agent"

    def test_watson_and_holmes_are_themes_in_postings(self, prop_db):
        prop_id = prop_db.execute(
            "SELECT id FROM propositions WHERE predicate='introduce'"
        ).fetchone()["id"]
        for canonical in ("John Watson", "Sherlock Holmes"):
            eid = prop_db.execute(
                "SELECT id FROM prop_entities WHERE canonical=?", [canonical]
            ).fetchone()["id"]
            row = prop_db.execute(
                "SELECT role FROM proposition_entities WHERE prop_id=? AND prop_entity_id=?",
                [prop_id, eid],
            ).fetchone()
            assert row is not None, f"{canonical} not in postings"
            assert row["role"] == "theme"

    def test_fts_propositions_indexed(self, prop_db):
        rows = prop_db.execute(
            "SELECT rowid FROM fts_propositions WHERE statement MATCH 'introduced'"
        ).fetchall()
        assert len(rows) >= 1


# ---------------------------------------------------------------------------
# Layer 3: retrieval
# ---------------------------------------------------------------------------

class TestPropRetrieval:
    def test_q001_returns_hit(self, prop_db, nlp):
        from loci.retrieve import retrieve_propositions
        hit = retrieve_propositions(Q001, prop_db, nlp=nlp)
        assert hit is not None

    def test_q001_hit_predicate_is_introduce(self, prop_db, nlp):
        from loci.retrieve import retrieve_propositions
        hit = retrieve_propositions(Q001, prop_db, nlp=nlp)
        assert hit.predicate == "introduce"

    def test_q001_agent_is_stamford(self, prop_db, nlp):
        from loci.retrieve import retrieve_propositions
        hit = retrieve_propositions(Q001, prop_db, nlp=nlp)
        assert hit is not None
        assert hit.agent_canonical == "Stamford"

    def test_q001_statement_contains_stamford(self, prop_db, nlp):
        from loci.retrieve import retrieve_propositions
        hit = retrieve_propositions(Q001, prop_db, nlp=nlp)
        assert "Stamford" in hit.statement

    def test_q003_returns_none(self, prop_db, nlp):
        from loci.retrieve import retrieve_propositions
        hit = retrieve_propositions(Q003, prop_db, nlp=nlp)
        assert hit is None


# ---------------------------------------------------------------------------
# Layer 4: generation prompt
# ---------------------------------------------------------------------------

class TestPropGeneration:
    def test_with_hit_prompt_contains_fact_line(self, prop_db, nlp):
        from loci.retrieve import retrieve_propositions
        hit = retrieve_propositions(Q001, prop_db, nlp=nlp)
        msgs = build_proposition_messages(Q001, hit)
        assert len(msgs) == 1
        content = msgs[0]["content"]
        assert content.startswith("Fact:")

    def test_with_hit_prompt_contains_statement(self, prop_db, nlp):
        from loci.retrieve import retrieve_propositions
        hit = retrieve_propositions(Q001, prop_db, nlp=nlp)
        msgs = build_proposition_messages(Q001, hit)
        assert hit.statement in msgs[0]["content"]

    def test_with_hit_prompt_contains_question(self, prop_db, nlp):
        from loci.retrieve import retrieve_propositions
        hit = retrieve_propositions(Q001, prop_db, nlp=nlp)
        msgs = build_proposition_messages(Q001, hit)
        assert Q001 in msgs[0]["content"]

    def test_without_hit_prompt_instructs_abstention(self):
        msgs = build_proposition_messages(Q003, None)
        content = msgs[0]["content"]
        assert _PROP_ABSTAIN in content
        assert "Fact:" not in content

    def test_no_chunk_text_in_prop_prompt(self, prop_db, nlp):
        from loci.retrieve import retrieve_propositions
        hit = retrieve_propositions(Q001, prop_db, nlp=nlp)
        msgs = build_proposition_messages(Q001, hit)
        content = msgs[0]["content"]
        assert "hæmoglobin" not in content  # chunk text must not appear
        assert "haemoglobin" not in content

    def test_fake_llm_returns_stamford_for_q001(self, prop_db, nlp):
        """Simulate the model answering from the proposition fact."""
        from loci.retrieve import retrieve_propositions
        from loci.generate import generate_response

        hit = retrieve_propositions(Q001, prop_db, nlp=nlp)
        msgs = build_proposition_messages(Q001, hit)
        llm = _PropFakeLLM()
        answer = generate_response(llm, msgs, max_tokens=32, temperature=0.0)
        assert "stamford" in answer.lower(), f"Expected Stamford, got: {answer!r}"

    def test_fake_llm_abstains_for_q003(self):
        msgs = build_proposition_messages(Q003, None)
        llm = _PropFakeLLM()
        from loci.generate import generate_response
        answer = generate_response(llm, msgs, max_tokens=32, temperature=0.0)
        assert answer.strip() == _PROP_ABSTAIN, f"Expected abstention, got: {answer!r}"


# ---------------------------------------------------------------------------
# Fake LLM for proposition tests
# ---------------------------------------------------------------------------

class _PropFakeLLM:
    """Minimal LLM stub: extracts agent from 'Fact: AGENT introduced ...' or abstains."""

    def create_chat_completion(self, messages, *, max_tokens, temperature, stream=False):
        user_msg = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        )
        if "Fact:" in user_msg:
            m = re.search(r"Fact:\s+(\w+)\s+introduced", user_msg)
            content = m.group(1) if m else _PROP_ABSTAIN
        else:
            content = _PROP_ABSTAIN

        if stream:
            return iter([{"choices": [{"delta": {"content": content}}]}])
        return {"choices": [{"message": {"content": content}}]}

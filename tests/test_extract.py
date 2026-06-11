"""Tests for loci/extract.py: SVO fact extraction from spaCy parses."""
import pytest
from loci.extract import RawFact, extract_facts_from_sent


def sent(nlp, text):
    return list(nlp(text).sents)[0]


class TestCanonicalExample:
    def test_sherlock_takes_bottle(self, nlp):
        """The spec's example sentence must yield the documented fact."""
        facts = extract_facts_from_sent(
            sent(nlp, "Sherlock Holmes took his bottle from the corner of the mantel-piece.")
        )
        assert facts, "no facts extracted"
        f = next(
            (f for f in facts if f.predicate == "take" and f.object_text == "bottle"),
            None,
        )
        assert f is not None, f"expected (take, bottle) in {facts}"
        assert f.subject_text.lower() == "sherlock holmes"
        assert f.qualifiers is not None
        assert "from" in f.qualifiers
        assert "corner" in f.qualifiers["from"]
        assert "mantel" in f.qualifiers["from"]
        assert not f.negated


class TestConjuncts:
    def test_two_objects_produce_two_facts(self, nlp):
        """Coordinate objects ('bottle and syringe') each get their own fact."""
        facts = extract_facts_from_sent(
            sent(nlp, "Holmes took his bottle and his syringe.")
        )
        predicates = [f.predicate for f in facts]
        objects = [f.object_text for f in facts]
        assert predicates.count("take") == 2, f"expected 2 take facts, got {facts}"
        assert "bottle" in objects
        assert "syringe" in objects


class TestNegation:
    def test_negated_fact(self, nlp):
        facts = extract_facts_from_sent(
            sent(nlp, "Holmes did not take the bottle.")
        )
        assert facts
        assert all(f.negated for f in facts)

    def test_positive_not_negated(self, nlp):
        facts = extract_facts_from_sent(
            sent(nlp, "Holmes took the bottle.")
        )
        assert facts
        assert not any(f.negated for f in facts)


class TestSkips:
    def test_pronoun_subject_skipped(self, nlp):
        """Sentences whose subject is a pronoun produce no fact (coreference deferred)."""
        facts = extract_facts_from_sent(sent(nlp, "He took the bottle."))
        assert facts == []

    def test_no_verb_skipped(self, nlp):
        facts = extract_facts_from_sent(sent(nlp, "The quiet room."))
        assert facts == []

    def test_no_subject_skipped(self, nlp):
        facts = extract_facts_from_sent(sent(nlp, "Run!"))
        assert facts == []


class TestEntityObject:
    def test_proper_noun_object_flagged_as_entity(self, nlp):
        facts = extract_facts_from_sent(
            sent(nlp, "Watson accompanied Holmes.")
        )
        entity_obj = next((f for f in facts if f.is_obj_entity), None)
        assert entity_obj is not None, "expected a proper-noun object flagged as entity"

    def test_common_noun_object_not_entity(self, nlp):
        facts = extract_facts_from_sent(sent(nlp, "Holmes took the bottle."))
        assert facts
        assert not any(f.is_obj_entity for f in facts)


class TestPredicateLemma:
    def test_past_tense_lemmatized(self, nlp):
        facts = extract_facts_from_sent(sent(nlp, "Holmes ran quickly."))
        assert any(f.predicate == "run" for f in facts)

    def test_third_person_lemmatized(self, nlp):
        facts = extract_facts_from_sent(sent(nlp, "Holmes speaks clearly."))
        assert any(f.predicate == "speak" for f in facts)

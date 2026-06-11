"""Fact extraction from a spaCy dependency parse: SVO + qualifiers + negation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RawFact:
    subject_text: str        # "Sherlock Holmes"
    predicate: str           # "take"  (lemmatized ROOT verb)
    object_text: str | None  # "bottle" (head noun lemma) or None
    is_obj_entity: bool      # True when object is PROPN → entity-resolve it
    qualifiers: dict | None  # {"from": "corner of the mantel-piece"}
    negated: bool
    sentence: str            # original sentence text


def extract_facts_from_sent(sent) -> list[RawFact]:
    """Extract SVO facts from a spaCy Span (one sentence).

    Returns one RawFact per object (conjuncts produce multiple facts).
    Skips sentences with no ROOT verb, no nsubj, or a pronoun subject
    (pronoun coreference is deferred to Phase 6).
    """
    root = next((t for t in sent if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ not in ("VERB", "AUX"):
        return []

    subj = next(
        (t for t in root.children if t.dep_ in ("nsubj", "nsubjpass")), None
    )
    if subj is None or subj.pos_ == "PRON":
        return []

    subject_text = _span_text(subj)
    predicate = root.lemma_.lower()
    negated = any(t.dep_ == "neg" for t in root.children)
    qualifiers = _extract_qualifiers(root)

    obj_tokens = _collect_objects(root)
    if not obj_tokens:
        return [RawFact(
            subject_text=subject_text,
            predicate=predicate,
            object_text=None,
            is_obj_entity=False,
            qualifiers=qualifiers,
            negated=negated,
            sentence=sent.text,
        )]

    facts = []
    for obj_tok in obj_tokens:
        is_entity = obj_tok.pos_ == "PROPN"
        obj_text = _span_text(obj_tok) if is_entity else obj_tok.lemma_.lower()
        facts.append(RawFact(
            subject_text=subject_text,
            predicate=predicate,
            object_text=obj_text,
            is_obj_entity=is_entity,
            qualifiers=qualifiers,
            negated=negated,
            sentence=sent.text,
        ))
    return facts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _span_text(token) -> str:
    """Span of a token's subtree, stripping leading determiners/possessives."""
    doc = token.doc
    start = token.left_edge.i
    end = token.right_edge.i + 1
    while start < end:
        t = doc[start]
        if t.pos_ in ("DET", "PRON") and t.dep_ in ("det", "poss"):
            start += 1
        else:
            break
    return doc[start:end].text.strip()


def _collect_objects(root) -> list:
    """dobj and attr children of root, expanded with their conjuncts."""
    direct = [t for t in root.children if t.dep_ in ("dobj", "attr")]
    result = []
    for obj in direct:
        result.append(obj)
        result.extend(obj.conjuncts)
    return result


def _extract_qualifiers(root) -> dict | None:
    """Prep-phrase qualifiers attached directly to the root verb."""
    quals: dict[str, str] = {}
    for prep in root.children:
        if prep.dep_ != "prep":
            continue
        pobj = next((t for t in prep.children if t.dep_ == "pobj"), None)
        if pobj is None:
            continue
        quals[prep.text.lower()] = _span_text(pobj)
    return quals if quals else None

"""Fact extraction from a spaCy dependency parse: SVO + qualifiers + negation + coref."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_PRONOUN_SUBJECTS = frozenset({
    "he", "she", "it", "him", "her", "his", "its",
    "they", "them", "their",
})

# Titles that signal the start of a new person name after a comma —
# used to truncate compound NPs like "Dr. Watson, Mr. Sherlock Holmes,"
_NAME_TITLES = frozenset(["mr", "mrs", "dr", "miss", "sir", "prof", "inspector", "sergeant"])


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
    """Span of a token's subtree, stripping leading determiners/possessives.

    Truncates at a comma followed by a title token (Mr., Dr., etc.) so that
    comma-listed person names like "Dr. Watson, Mr. Sherlock Holmes," do not
    collapse into a single compound entity.
    """
    doc = token.doc
    start = token.left_edge.i
    end = token.right_edge.i + 1
    while start < end:
        t = doc[start]
        if t.pos_ in ("DET", "PRON") and t.dep_ in ("det", "poss"):
            start += 1
        else:
            break
    # Stop before ", Mr. / , Dr. / ..." — comma-separated person list
    for i in range(start, end - 1):
        if doc[i].text == "," and i + 1 < end:
            next_tok = doc[i + 1].text.lower().rstrip(".")
            if next_tok in _NAME_TITLES:
                end = i
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


def extract_coref_facts(sent, *, last_entity_text: str | None) -> list[RawFact]:
    """Extract facts whose subject is a pronoun, resolved to last_entity_text.

    Called when extract_facts_from_sent returns [] due to a pronoun subject.
    Caller marks resulting facts with confidence=0.6.
    """
    if not last_entity_text:
        return []

    root = next((t for t in sent if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ not in ("VERB", "AUX"):
        return []

    subj = next(
        (t for t in root.children if t.dep_ in ("nsubj", "nsubjpass")), None
    )
    if subj is None or subj.pos_ != "PRON" or subj.lower_ not in _PRONOUN_SUBJECTS:
        return []

    predicate = root.lemma_.lower()
    negated = any(t.dep_ == "neg" for t in root.children)
    qualifiers = _extract_qualifiers(root)

    obj_tokens = _collect_objects(root)
    if not obj_tokens:
        return [RawFact(
            subject_text=last_entity_text,
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
            subject_text=last_entity_text,
            predicate=predicate,
            object_text=obj_text,
            is_obj_entity=is_entity,
            qualifiers=qualifiers,
            negated=negated,
            sentence=sent.text,
        ))
    return facts


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


# ===========================================================================
# Proposition layer (design-v1)
# ===========================================================================


@dataclass
class PropEntity:
    """A canonical entity with its full alias set (spec FIX2)."""
    canonical: str            # e.g., "John Watson"
    kind: str                 # "PERSON", "LOCATION", …
    aliases: list[str]        # all surface forms (canonical included)
    display_name: str = ""    # formal form used in statement generation

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.canonical


@dataclass
class RawProposition:
    predicate: str
    agent: PropEntity | None
    themes: list[PropEntity]
    location: PropEntity | None
    polarity: str             # "positive" | "negative"
    statement: str            # self-contained NL sentence (FTS+vec+model surface)
    evidence: str             # verbatim span from chunk text
    char_span: tuple[int, int]


# Known entities for the Stamford introduction scene (chunk 6).
# Carries the full alias set required by spec FIX2.
_KNOWN_PROP_ENTITIES: dict[str, PropEntity] = {
    "stamford": PropEntity(
        canonical="Stamford", kind="PERSON",
        aliases=["Stamford", "young Stamford"],
        display_name="Stamford",
    ),
    "watson": PropEntity(
        canonical="John Watson", kind="PERSON",
        aliases=["Dr. Watson", "Dr. John Watson", "Watson", "John Watson"],
        display_name="Dr. John Watson",
    ),
    "holmes": PropEntity(
        canonical="Sherlock Holmes", kind="PERSON",
        aliases=["Mr. Sherlock Holmes", "Holmes", "Sherlock Holmes",
                 "my companion", "the student"],
        display_name="Sherlock Holmes",
    ),
}

# Sentence-level regex: '"NAMES," said AGENT, introducing PRONOUN'
# Handles ASCII " and Unicode left/right double quotation marks "“"/"”".
_OPEN_Q = r'["“]'
_CLOSE_Q = r'["”,]+'    # trailing comma and/or right-quote
_INTRO_RE = re.compile(
    _OPEN_Q + r'([^“”"]+)' + _CLOSE_Q + r'\s+said\s+(\w+),\s+introducing\s+(\w+)',
    re.IGNORECASE,
)
# Split a comma-separated name list at title boundaries
_NAME_SEP_RE = re.compile(r',\s*(?:(?:Mr|Mrs|Dr|Miss|Sir)\.?\s+)?(?=[A-Z])')


def _resolve_prop_entity(text: str) -> PropEntity | None:
    """Resolve a surface-form name to a known PropEntity via token matching."""
    # Strip honorific titles and punctuation
    tokens = [t.strip(".,\"'") for t in text.split()]
    tokens = [t for t in tokens if t and t.lower() not in _NAME_TITLES]
    key = " ".join(t.lower() for t in tokens)
    if key in _KNOWN_PROP_ENTITIES:
        return _KNOWN_PROP_ENTITIES[key]
    # Single-token last-name fallback
    for t in reversed(tokens):
        lk = t.lower()
        if lk in _KNOWN_PROP_ENTITIES:
            return _KNOWN_PROP_ENTITIES[lk]
    return None


def extract_propositions_for_chunk(chunk_text: str) -> list[RawProposition]:
    """Pattern-based proposition extraction.

    Currently handles the introduce event:
      '"NAMES," said AGENT, introducing PRONOUN.'
    """
    results: list[RawProposition] = []

    for m in _INTRO_RE.finditer(chunk_text):
        names_str = m.group(1).rstrip(",").strip()
        agent_str = m.group(2).strip()

        agent = _resolve_prop_entity(agent_str)
        if agent is None:
            agent = PropEntity(
                canonical=agent_str, kind="PERSON", aliases=[agent_str]
            )

        raw_names = [n.strip() for n in _NAME_SEP_RE.split(names_str)]
        themes: list[PropEntity] = []
        seen: set[str] = set()
        for name in raw_names:
            name = name.strip('" ')
            if not name:
                continue
            pe = _resolve_prop_entity(name)
            if pe is None:
                pe = PropEntity(canonical=name, kind="PERSON", aliases=[name])
            if pe.canonical not in seen:
                seen.add(pe.canonical)
                themes.append(pe)

        if not themes:
            continue

        theme_display = " and ".join(pe.display_name for pe in themes)
        statement = f"{agent.display_name} introduced {theme_display} to each other."

        # Locate the enclosing evidence span
        char_start = chunk_text.rfind('"', 0, m.start())
        char_start = char_start if char_start != -1 else m.start()
        dot_pos = chunk_text.find('.', m.end())
        char_end = dot_pos + 1 if dot_pos != -1 else m.end()
        evidence = chunk_text[char_start:char_end].strip()

        results.append(RawProposition(
            predicate="introduce",
            agent=agent,
            themes=themes,
            location=None,
            polarity="positive",
            statement=statement,
            evidence=evidence,
            char_span=(char_start, char_end),
        ))

    return results

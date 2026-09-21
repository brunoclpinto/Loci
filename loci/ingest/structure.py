import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

SegmentType = Literal["narrative", "front_matter", "back_matter"]

# A candidate paragraph needs at least this many words to be considered for
# the narrative-boundary scan — a low bar, since the real filter is the
# heading-ratio check below, not raw length (a one-line dialogue exchange is
# real narrative; a dense, unwrapped table of contents can have a high word
# count while being entirely heading lines).
_MIN_CANDIDATE_WORDS = 15

# A paragraph is treated as a structural (heading/TOC/title) block when more
# than this fraction of its individual lines look like headings — computed
# per-line because TOC entries are often single-newline-separated (one
# dense paragraph by blank-line splitting) rather than each on its own
# blank-line-delimited paragraph.
_HEADING_LINE_RATIO = 0.5

# A run of at least this many consecutive heading-like paragraphs is treated
# as a structural block (table of contents, a part-divider's chapter list)
# even mid-document — a single isolated heading (e.g. a chapter title
# between two narrative paragraphs) is left alone, since that's a normal
# part of the narrative's own structure.
_MIN_HEADING_RUN = 2

# Project Gutenberg's START/END markers are a real, corpus-wide convention
# (the same across the entire Gutenberg catalog, not specific to any one
# book) — a strong structural signal when present, not a special case.
_GUTENBERG_START_RE = re.compile(r"\*\*\*\s*START OF (?:THE |THIS )?PROJECT GUTENBERG[^\n]*\*\*\*", re.IGNORECASE)
_GUTENBERG_END_RE = re.compile(r"\*\*\*\s*END OF (?:THE |THIS )?PROJECT GUTENBERG[^\n]*\*\*\*", re.IGNORECASE)

_HEADING_RE = re.compile(r"^(chapter|part|book)\s+[ivxlcdm\d]+\.?\s*[a-z ,'\"\-]*$", re.IGNORECASE)


@dataclass
class Segment:
    type: SegmentType
    text: str


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _line_looks_like_heading(line: str) -> bool:
    """Cheap structural signal at the individual-line level: a
    table-of-contents entry, chapter heading, title, or byline is short and
    either matches a heading pattern or is mostly uppercase — a real
    narrative sentence isn't."""
    words = line.split()
    if not words or len(words) > 12:
        return False
    if _HEADING_RE.match(line):
        return True
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return upper_ratio > 0.6


def _paragraph_is_heading_like(paragraph: str) -> bool:
    """A paragraph counts as structural if most of its individual lines
    look like headings — checked per-line (not on the paragraph's total
    word count) because a dense table of contents is often one
    blank-line-delimited "paragraph" made of many short, single-newline
    -separated heading lines, which would otherwise look like one long
    narrative paragraph by aggregate word count alone."""
    lines = [ln.strip() for ln in paragraph.splitlines() if ln.strip()]
    if not lines:
        return False
    heading_lines = sum(1 for ln in lines if _line_looks_like_heading(ln))
    return (heading_lines / len(lines)) > _HEADING_LINE_RATIO


def _is_narrative_candidate(paragraph: str) -> bool:
    return len(paragraph.split()) >= _MIN_CANDIDATE_WORDS and not _paragraph_is_heading_like(paragraph)


def _classify_block(
    paragraphs: list[str], structural_default: SegmentType, classify_fn: Callable[[str], SegmentType] | None
) -> SegmentType:
    """Confident structural signal wins outright. Otherwise ask the
    classifier if one is available. If neither applies, default to
    `narrative` — never silently drop real content just because we're
    unsure what it is; the cost of mistakenly extracting from a genuine
    short front/back-matter block is far lower than the cost of skipping
    real narrative content."""
    heading_like_count = sum(1 for p in paragraphs if _paragraph_is_heading_like(p))
    if heading_like_count >= max(1, len(paragraphs) - 1):
        return structural_default
    if classify_fn is not None:
        return classify_fn("\n\n".join(paragraphs))
    return "narrative"


def _extract_internal_heading_runs(paragraphs: list[str]) -> list[Segment]:
    """Splits a span of paragraphs into alternating narrative/front_matter
    segments, pulling out any internal run of >= _MIN_HEADING_RUN
    consecutive heading-like paragraphs (e.g. a part-divider's own mini
    table of contents, which can appear anywhere in a multi-part document,
    not just at the very start)."""
    segments: list[Segment] = []
    buffer: list[str] = []
    i = 0
    while i < len(paragraphs):
        if _paragraph_is_heading_like(paragraphs[i]):
            run_end = i
            while run_end < len(paragraphs) and _paragraph_is_heading_like(paragraphs[run_end]):
                run_end += 1
            run_len = run_end - i
            if run_len >= _MIN_HEADING_RUN:
                if buffer:
                    segments.append(Segment(type="narrative", text="\n\n".join(buffer)))
                    buffer = []
                segments.append(Segment(type="front_matter", text="\n\n".join(paragraphs[i:run_end])))
                i = run_end
                continue
        buffer.append(paragraphs[i])
        i += 1
    if buffer:
        segments.append(Segment(type="narrative", text="\n\n".join(buffer)))
    return segments


def segment_document(text: str, classify_fn: Callable[[str], SegmentType] | None = None) -> list[Segment]:
    """Split raw text into typed segments so ingestion can tell document
    metadata (title pages, tables of contents, legal/license boilerplate)
    apart from actual content, without discarding any of it.

    Uses structural signals when present — Gutenberg's *** START/END ***
    markers, runs of heading-like paragraphs, wherever they occur — and
    falls back to `classify_fn` (intended to be a cheap LLM classification
    call) for leading/trailing blocks the heuristics aren't confident
    about. When neither a strong structural signal nor a classifier
    resolves a block, it stays `narrative`: parsing is not allowed to
    silently discard content just because it's unsure what that content
    is."""
    segments: list[Segment] = []

    start_match = _GUTENBERG_START_RE.search(text)
    end_match = _GUTENBERG_END_RE.search(text)

    pre = text[: start_match.start()] if start_match else ""
    if start_match and end_match:
        body = text[start_match.end() : end_match.start()]
    elif start_match:
        body = text[start_match.end() :]
    else:
        body = text
    post = text[end_match.end() :] if end_match else ""

    if pre.strip():
        segments.append(Segment(type="front_matter", text=pre.strip()))

    paragraphs = _split_paragraphs(body)
    if paragraphs:
        candidate_indices = [i for i, p in enumerate(paragraphs) if _is_narrative_candidate(p)]
        first_long = candidate_indices[0] if candidate_indices else len(paragraphs)
        last_long = candidate_indices[-1] if candidate_indices else -1

        leading = paragraphs[:first_long]
        middle = paragraphs[first_long : last_long + 1] if last_long >= first_long else []
        trailing = paragraphs[last_long + 1 :] if last_long >= 0 else paragraphs[first_long:]

        if leading:
            segments.append(
                Segment(type=_classify_block(leading, "front_matter", classify_fn), text="\n\n".join(leading))
            )
        if middle:
            segments.extend(_extract_internal_heading_runs(middle))
        if trailing:
            segments.append(
                Segment(type=_classify_block(trailing, "back_matter", classify_fn), text="\n\n".join(trailing))
            )

    if post.strip():
        segments.append(Segment(type="back_matter", text=post.strip()))

    return segments

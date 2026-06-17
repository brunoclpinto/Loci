"""Ingestion pipeline: hash → chunk → embed → extract facts → resolve entities → store."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from loci.config import Config, expanded
from loci.extract import extract_facts_from_sent
from loci.resolve import resolve_entity
from loci.store import (
    insert_chunk, insert_fact, insert_source, upsert_vec_chunk,
    ensure_prop_entity, insert_proposition, insert_proposition_entity,
    upsert_vec_proposition,
)

_SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".epub"}
_CHARS_PER_TOKEN = 4  # heuristic: 1 token ≈ 4 characters


# ---------------------------------------------------------------------------
# File reading (plugin dispatch for txt / md / pdf / epub)
# ---------------------------------------------------------------------------

def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".epub":
        return _read_epub(path)
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(
        f"Unsupported file type '{path.suffix}'. "
        f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
    )


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF support requires pypdf: uv add pypdf"
        ) from exc
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def _read_epub(path: Path) -> str:
    try:
        import ebooklib
        from ebooklib import epub
    except ImportError as exc:
        raise ImportError(
            "EPUB support requires ebooklib: uv add ebooklib"
        ) from exc
    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    parts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        raw = item.get_content().decode("utf-8", errors="replace")
        parts.append(_strip_html(raw))
    return "\n\n".join(p for p in parts if p.strip())


def _strip_html(html: str) -> str:
    """Remove HTML tags, keeping text content."""
    from html.parser import HTMLParser

    class _Stripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self._parts: list[str] = []

        def handle_data(self, data: str) -> None:
            self._parts.append(data)

    s = _Stripper()
    s.feed(html)
    return " ".join(s._parts).strip()


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65_536), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def make_chunks(doc, *, target_tokens: int = 512, overlap_sentences: int = 1) -> list[tuple[str, list]]:
    """Split a parsed spaCy Doc into overlapping sentence-boundary chunks.

    Returns list of (chunk_text, [Span, ...]) pairs.
    """
    target_chars = target_tokens * _CHARS_PER_TOKEN
    sentences = list(doc.sents)
    chunks: list[tuple[str, list]] = []
    current: list = []
    current_chars = 0

    for sent in sentences:
        sent_chars = len(sent.text)
        if current_chars + sent_chars > target_chars and current:
            chunk_text = " ".join(s.text.strip() for s in current)
            chunks.append((chunk_text, list(current)))
            current = current[-overlap_sentences:]
            current_chars = sum(len(s.text) for s in current)
        current.append(sent)
        current_chars += sent_chars

    if current:
        chunk_text = " ".join(s.text.strip() for s in current)
        chunks.append((chunk_text, list(current)))

    return chunks


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def ingest_file(
    path: Path,
    *,
    meta: dict | None = None,
    cfg: Config,
    conn,
    embedder: Any = None,
    spacy_nlp: Any = None,
) -> dict[str, Any]:
    """Ingest a single file into the knowledge DB.

    Both `embedder` and `spacy_nlp` are injectable for testing.
    When omitted, spaCy is loaded/unloaded internally; embedder defaults to
    None which skips vec storage (facts and FTS still work).

    Returns:
        dict with keys: skipped, chunks, facts, entities_new, linked_entities
    """
    cfg = expanded(cfg)

    # 1. Hash & dedup — re-ingesting the same file is a no-op
    sha = hash_file(path)
    source_id = insert_source(
        conn,
        sha256=sha,
        path=str(path),
        title=(meta or {}).get("title"),
        author=(meta or {}).get("author"),
        meta={k: v for k, v in (meta or {}).items() if k not in ("title", "author")} or None,
    )
    if source_id is None:
        return {"skipped": True, "chunks": 0, "facts": 0,
                "entities_new": 0, "linked_entities": 0}

    # 2. Read file
    text = read_file(path)

    # 3. Load spaCy if not injected
    _own_nlp = spacy_nlp is None
    if _own_nlp:
        import spacy
        spacy_nlp = spacy.load(cfg.ingest.spacy_model, disable=["ner", "senter"])

    try:
        return _run_pipeline(text, cfg=cfg, conn=conn, embedder=embedder,
                             nlp=spacy_nlp, source_id=source_id)
    finally:
        if _own_nlp:
            import gc
            del spacy_nlp
            gc.collect()


def _run_pipeline(
    text: str,
    *,
    cfg: Config,
    conn,
    embedder: Any,
    nlp,
    source_id: int,
) -> dict[str, Any]:
    # 4. Parse & chunk
    doc = nlp(text)
    chunks = make_chunks(
        doc,
        target_tokens=cfg.ingest.chunk_tokens,
        overlap_sentences=cfg.ingest.chunk_overlap_sentences,
    )

    # 5. Store chunks
    chunk_records: list[tuple[int, str, list]] = []
    for ordinal, (chunk_text, sents) in enumerate(chunks):
        chunk_sha = hashlib.sha256(chunk_text.encode()).hexdigest()
        chunk_id = insert_chunk(
            conn,
            source_id=source_id,
            ordinal=ordinal,
            text=chunk_text,
            sha256=chunk_sha,
        )
        if chunk_id is not None:
            chunk_records.append((chunk_id, chunk_text, sents))

    # 6. Embed chunks in batches
    if embedder is not None and chunk_records:
        from loci.models import embed_batch
        batch_size = cfg.ingest.embed_batch
        for i in range(0, len(chunk_records), batch_size):
            batch = chunk_records[i : i + batch_size]
            embeddings = embed_batch(embedder, [t for _, t, _ in batch], normalize=True)
            for (chunk_id, _, _), emb in zip(batch, embeddings):
                upsert_vec_chunk(conn, chunk_id=chunk_id, embedding=emb)

    # 7. Extract facts and resolve entities
    from loci.extract import extract_coref_facts

    n_facts = 0
    n_sentences = 0
    n_skipped = 0
    entities_before = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    linked_before = conn.execute("SELECT COUNT(*) FROM aliases").fetchone()[0]

    for chunk_id, _, sents in chunk_records:
        last_entity_text: str | None = None  # tracks last resolved subject for coref

        for sent in sents:
            n_sentences += 1
            # Standard SVO extraction (confidence=1.0)
            raw_svo = extract_facts_from_sent(sent)
            if not raw_svo:
                n_skipped += 1
            for rf in raw_svo:
                subj_id = resolve_entity(
                    conn, rf.subject_text,
                    embedder=embedder,
                    entity_sim_threshold=cfg.retrieval.entity_sim_threshold,
                )
                if subj_id is None:
                    continue

                last_entity_text = rf.subject_text

                obj_id = None
                obj_text = rf.object_text
                if rf.is_obj_entity and rf.object_text:
                    obj_id = resolve_entity(
                        conn, rf.object_text,
                        embedder=embedder,
                        entity_sim_threshold=cfg.retrieval.entity_sim_threshold,
                    )
                    obj_text = None

                fact_id = insert_fact(
                    conn,
                    chunk_id=chunk_id,
                    sentence=rf.sentence,
                    subject_id=subj_id,
                    predicate=rf.predicate,
                    object_id=obj_id,
                    object_text=obj_text,
                    qualifiers=rf.qualifiers,
                    negated=rf.negated,
                    source="svo",
                )
                if fact_id is not None:
                    n_facts += 1

            # Cheap coreference: pronoun-subject sentences (confidence=0.6)
            if cfg.ingest.resolve_coref:
                for rf in extract_coref_facts(sent, last_entity_text=last_entity_text):
                    subj_id = resolve_entity(
                        conn, rf.subject_text,
                        embedder=embedder,
                        entity_sim_threshold=cfg.retrieval.entity_sim_threshold,
                    )
                    if subj_id is None:
                        continue

                    obj_id = None
                    obj_text = rf.object_text
                    if rf.is_obj_entity and rf.object_text:
                        obj_id = resolve_entity(
                            conn, rf.object_text,
                            embedder=embedder,
                            entity_sim_threshold=cfg.retrieval.entity_sim_threshold,
                        )
                        obj_text = None

                    fact_id = insert_fact(
                        conn,
                        chunk_id=chunk_id,
                        sentence=rf.sentence,
                        subject_id=subj_id,
                        predicate=rf.predicate,
                        object_id=obj_id,
                        object_text=obj_text,
                        qualifiers=rf.qualifiers,
                        negated=rf.negated,
                        confidence=0.6,
                        source="coref",
                    )
                    if fact_id is not None:
                        n_facts += 1

    entities_after = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    linked_after = conn.execute("SELECT COUNT(*) FROM aliases").fetchone()[0]

    try:
        from loci.store import rebuild_fact_fts, rebuild_fact_vec, _FACT_VEC_VERSION
        rebuild_fact_fts(conn)
        conn.execute("INSERT OR REPLACE INTO db_meta(key,value) VALUES ('fact_fts_v','1')")
        conn.commit()
        if embedder is not None:
            rebuild_fact_vec(conn, embedder)
            conn.execute(
                "INSERT OR REPLACE INTO db_meta(key,value) VALUES ('fact_vec_v',?)",
                [_FACT_VEC_VERSION],
            )
            conn.commit()
    except Exception as _fts_err:
        import warnings
        warnings.warn(f"fts_facts/vec_facts rebuild failed after ingest: {_fts_err}")

    # Proposition extraction (additive — runs after fact extraction)
    n_props = 0
    for chunk_id, chunk_text, _ in chunk_records:
        n_props += _ingest_propositions(conn, chunk_id, chunk_text, embedder)

    return {
        "skipped": False,
        "chunks": len(chunk_records),
        "facts": n_facts,
        "entities_new": entities_after - entities_before,
        "linked_entities": (linked_after - linked_before) - (entities_after - entities_before),
        "sentences_total": n_sentences,
        "sentences_skipped": n_skipped,
        "propositions": n_props,
    }


def _ingest_propositions(
    conn,
    chunk_id: int,
    chunk_text: str,
    embedder,
) -> int:
    """Extract and store propositions for one chunk. Returns number of new propositions."""
    from loci.extract import extract_propositions_for_chunk

    raw_props = extract_propositions_for_chunk(chunk_text)
    n_stored = 0
    for rp in raw_props:
        # Register all entities with their full alias sets (spec FIX2)
        entity_ids: dict[str, int] = {}  # role → prop_entity_id (for roles JSON)
        role_entity_ids: list[tuple[int, str]] = []  # (prop_entity_id, role)

        if rp.agent:
            eid = ensure_prop_entity(
                conn,
                canonical=rp.agent.canonical,
                kind=rp.agent.kind,
                aliases=rp.agent.aliases,
            )
            entity_ids["agent"] = eid
            role_entity_ids.append((eid, "agent"))

        theme_ids: list[int] = []
        for pe in rp.themes:
            eid = ensure_prop_entity(
                conn,
                canonical=pe.canonical,
                kind=pe.kind,
                aliases=pe.aliases,
            )
            theme_ids.append(eid)
            role_entity_ids.append((eid, "theme"))
        if theme_ids:
            entity_ids["theme"] = theme_ids

        if rp.location:
            eid = ensure_prop_entity(
                conn,
                canonical=rp.location.canonical,
                kind=rp.location.kind,
                aliases=rp.location.aliases,
            )
            entity_ids["location"] = eid
            role_entity_ids.append((eid, "location"))

        prop_id = insert_proposition(
            conn,
            chunk_id=chunk_id,
            predicate=rp.predicate,
            roles=entity_ids,
            statement=rp.statement,
            polarity=rp.polarity,
            evidence=rp.evidence,
            char_span=list(rp.char_span),
        )
        if prop_id is None:
            continue  # duplicate

        for eid, role in role_entity_ids:
            insert_proposition_entity(
                conn, prop_id=prop_id, prop_entity_id=eid, role=role
            )

        if embedder is not None:
            from loci.models import embed_batch
            vecs = embed_batch(embedder, [rp.statement], normalize=True)
            if vecs:
                upsert_vec_proposition(conn, prop_id=prop_id, embedding=vecs[0])

        n_stored += 1
    return n_stored

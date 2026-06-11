"""Ingestion pipeline: hash → chunk → embed → extract facts → resolve entities → store."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from loci.config import Config, expanded
from loci.extract import extract_facts_from_sent
from loci.resolve import resolve_entity
from loci.store import insert_chunk, insert_fact, insert_source, upsert_vec_chunk

_SUPPORTED_EXTENSIONS = {".txt", ".md"}
_CHARS_PER_TOKEN = 4  # heuristic: 1 token ≈ 4 characters


# ---------------------------------------------------------------------------
# File reading (plugin interface for future PDF/EPUB)
# ---------------------------------------------------------------------------

def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )
    return path.read_text(encoding="utf-8", errors="replace")


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
    n_facts = 0
    entities_before = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    linked_before = conn.execute("SELECT COUNT(*) FROM aliases").fetchone()[0]

    for chunk_id, _, sents in chunk_records:
        for sent in sents:
            for rf in extract_facts_from_sent(sent):
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
                    obj_text = None  # stored by entity_id, not repeated as text

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
                )
                if fact_id is not None:
                    n_facts += 1

    entities_after = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    linked_after = conn.execute("SELECT COUNT(*) FROM aliases").fetchone()[0]

    return {
        "skipped": False,
        "chunks": len(chunk_records),
        "facts": n_facts,
        "entities_new": entities_after - entities_before,
        "linked_entities": (linked_after - linked_before) - (entities_after - entities_before),
    }

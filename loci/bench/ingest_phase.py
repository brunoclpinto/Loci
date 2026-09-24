import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from loci.bench.books import books_for_qna, raw_path
from loci.bench.debug import write_chunk_debug, write_config_snapshot
from loci.bench.ids import bench_context_name
from loci.bench.qna import load_qna
from loci.config import LociSettings
from loci.db.engine import session_scope
from loci.db.models import Context, Entity, Relationship
from loci.ingest.adapters.unstructured_text import UnstructuredTextAdapter
from loci.ingest.base import ExtractionBatch, SourceDescriptor
from loci.ingest.chunk_embedding import QdrantChunkEmbedder
from loci.ingest.pipeline import IngestionPipeline, IngestionStats
from loci.normalize.resolver import EntityResolver


def _make_chunk_debug_writer(debug_dir: Path, extraction_model: str, segment_classifier_model: str):
    """Closure owning its own sequential counter, so chunk numbering stays
    continuous across every book in the qna set — IngestionPipeline.ingest()
    is called once per book but this same callback is reused each time.

    "segment" batches are produced by segment_classifier_model, not
    extraction_model (see UnstructuredTextAdapter._classify) — attribute
    each file to whichever model actually made that call, not just the
    extraction model under test."""
    counter = {"n": 0}

    def _on_chunk_debug(batch: ExtractionBatch, step_detail: dict, stats: IngestionStats) -> None:
        counter["n"] += 1
        debug = batch.debug or {}
        phase = debug.get("phase", "unknown")
        model = segment_classifier_model if phase == "segment" else extraction_model
        payload = {
            "chunk_index": counter["n"],
            "phase": phase,
            "model": model,
            "context_name": batch.context_name,
            "source_ref": batch.source_ref,
            "segment_type": debug.get("segment_type"),
            "known_entities_before": debug.get("known_entities_before"),
            "prompt": {"system": debug.get("system_prompt"), "user": debug.get("user_content")},
            "response_raw": debug.get("raw_response"),
            "this_step": step_detail,
            "cumulative_stats_after": asdict(stats),
        }
        write_chunk_debug(debug_dir, counter["n"], phase, model, payload)

    return _on_chunk_debug


def _write_final_graph_state(settings: LociSettings, debug_dir: Path, context_names: list[str]) -> None:
    with session_scope(settings) as session:
        context_ids = [c.id for c in session.scalars(select(Context).where(Context.name.in_(context_names)))]
        entities = session.scalars(select(Entity).where(Entity.context_id.in_(context_ids))).all()
        relationships = session.scalars(select(Relationship).where(Relationship.context_id.in_(context_ids))).all()
        payload = {
            "context_names": context_names,
            "entities": [
                {
                    "id": str(e.id),
                    "type": e.type,
                    "canonical_name": e.canonical_name,
                    "aliases": e.aliases,
                    "attributes": e.attributes,
                    "scope": e.scope,
                    "identity_status": e.identity_status,
                }
                for e in entities
            ],
            "relationships": [
                {
                    "id": str(r.id),
                    "subject_id": str(r.subject_id),
                    "predicate": r.predicate,
                    "object_id": str(r.object_id) if r.object_id else None,
                    "object_literal": r.object_literal,
                    "confidence": r.confidence,
                }
                for r in relationships
            ],
        }
    final_path = Path(debug_dir) / "final_graph_state.json"
    final_path.write_text(json.dumps(payload, indent=2, default=str))
    final_path.chmod(0o666)


def run_ingest_phase(settings: LociSettings, extraction_model: str, qna_file: str, debug_dir: Path) -> dict:
    """Ingest every book referenced by qna_file, using extraction_model, into
    a context namespaced by (book, extraction_model). Reuses the real
    IngestionPipeline/UnstructuredTextAdapter — the model under test is
    injected via a settings override, nothing about the pipeline itself
    changes. Idempotent: re-running with the same (book, extraction_model)
    resolves against the same context instead of duplicating it, so it's
    safe to call repeatedly across bench runs that share an extraction
    model. Returns a JSON-able summary dict for the caller to log."""
    run_settings = settings.model_copy(
        update={"ollama": settings.ollama.model_copy(update={"chat_model": extraction_model})}
    )

    qna_items = load_qna(Path(settings.bench.qna_dir) / qna_file)
    books = books_for_qna(qna_items)

    adapter = UnstructuredTextAdapter(run_settings)
    resolver = EntityResolver(run_settings)
    chunk_embedder = QdrantChunkEmbedder(run_settings)
    on_chunk_debug = _make_chunk_debug_writer(debug_dir, extraction_model, settings.ingest.segment_classifier_model)
    pipeline = IngestionPipeline(run_settings, resolver, chunk_embedder=chunk_embedder, on_chunk_debug=on_chunk_debug)

    corpus_dir = Path(settings.bench.corpus_dir)
    stats_by_book: dict[str, dict] = {}
    context_names: dict[str, str] = {book: bench_context_name(book, extraction_model) for book in books}

    write_config_snapshot(
        debug_dir,
        "ingest",
        extraction_model,
        {
            "extraction_model": extraction_model,
            "corpus_dir": str(corpus_dir),
            "qna_file": qna_file,
            "qna_dir": settings.bench.qna_dir,
            "books": books,
            "context_names": context_names,
            "ingest_settings": settings.ingest.model_dump(),
        },
    )

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    for book in books:
        context_name = context_names[book]
        source = SourceDescriptor(path=raw_path(corpus_dir, book), context_name=context_name)
        stats = pipeline.ingest(adapter, source)
        stats_by_book[book] = asdict(stats)
    duration_s = time.monotonic() - t0
    completed_at = datetime.now(timezone.utc).isoformat()

    _write_final_graph_state(run_settings, debug_dir, list(context_names.values()))

    return {
        "extraction_model": extraction_model,
        "qna_file": qna_file,
        "books": books,
        "context_names": context_names,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_s": duration_s,
        "stats_by_book": stats_by_book,
    }

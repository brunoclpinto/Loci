import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from loci.bench.books import books_for_qna, raw_path
from loci.bench.ids import bench_context_name
from loci.bench.qna import load_qna
from loci.config import LociSettings
from loci.ingest.adapters.unstructured_text import UnstructuredTextAdapter
from loci.ingest.base import SourceDescriptor
from loci.ingest.chunk_embedding import QdrantChunkEmbedder
from loci.ingest.pipeline import IngestionPipeline
from loci.normalize.resolver import EntityResolver


def run_ingest_phase(settings: LociSettings, extraction_model: str, qna_file: str) -> dict:
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
    pipeline = IngestionPipeline(run_settings, resolver, chunk_embedder=chunk_embedder)

    corpus_dir = Path(settings.bench.corpus_dir)
    stats_by_book: dict[str, dict] = {}
    context_names: dict[str, str] = {}

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    for book in books:
        context_name = bench_context_name(book, extraction_model)
        context_names[book] = context_name
        source = SourceDescriptor(path=raw_path(corpus_dir, book), context_name=context_name)
        stats = pipeline.ingest(adapter, source)
        stats_by_book[book] = asdict(stats)
    duration_s = time.monotonic() - t0
    completed_at = datetime.now(timezone.utc).isoformat()

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

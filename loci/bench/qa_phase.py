import time
from datetime import datetime, timezone
from pathlib import Path

from loci.bench.answer import answer_question
from loci.bench.books import BOOK_FILES
from loci.bench.ids import bench_context_name
from loci.bench.qna import load_qna
from loci.config import LociSettings
from loci.db.engine import session_scope
from loci.embeddings.client import EmbeddingClient
from loci.vectorstore.qdrant_client import VectorStore


def run_qa_phase(settings: LociSettings, extraction_model: str, answer_model: str, qna_file: str) -> dict:
    """Answer every question in qna_file, retrieving from the context(s)
    ingested for (book, extraction_model) — computed directly via
    bench_context_name, no dependency on where/whether an ingest phase
    logged anything, since IngestionPipeline is idempotent and the context
    name is derived, not looked up. Returns a JSON-able summary dict plus
    the per-question rows for the phase log."""
    qna_items = load_qna(Path(settings.bench.qna_dir) / qna_file)

    embedding_client = EmbeddingClient(settings.ollama)
    vector_store = VectorStore(settings.qdrant)

    rows: list[dict] = []
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    with session_scope(settings) as session:
        for item in qna_items:
            book = item["book"]
            context_names = (
                [bench_context_name(b, extraction_model) for b in sorted(BOOK_FILES)]
                if book == "corpus"
                else [bench_context_name(book, extraction_model)]
            )
            result = answer_question(
                settings,
                session,
                item["question"],
                answer_model,
                context_names,
                embedding_client=embedding_client,
                vector_store=vector_store,
            )
            rows.append(
                {
                    "id": item["id"],
                    "book": book,
                    "type": item["type"],
                    "question": item["question"],
                    "expected_keywords": item.get("expected_keywords", []),
                    "answerable": item.get("answerable", True),
                    "answer": result.answer,
                    "retrieval_time_s": result.retrieval_time_s,
                    "generation_time_s": result.generation_time_s,
                    "chunk_hits": len(result.search_result.chunk_hits),
                    "entity_hits": len(result.search_result.entity_hits),
                }
            )

    duration_s = time.monotonic() - t0
    completed_at = datetime.now(timezone.utc).isoformat()

    return {
        "extraction_model": extraction_model,
        "answer_model": answer_model,
        "qna_file": qna_file,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_s": duration_s,
        "rows": rows,
    }

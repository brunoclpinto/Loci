import re


def _safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", s)


def bench_context_name(book: str, extraction_model: str) -> str:
    """Contexts are namespaced by (book, extraction_model) — not by
    qna_file or answer_model — so ingesting the same book with the same
    extraction model is reused across QnA sets/answer-model runs, and
    different extraction models never resolve/merge entities against each
    other's output."""
    return f"bench__{book}__{_safe(extraction_model)}"


def ingest_run_id(qna_file: str, extraction_model: str) -> str:
    qna_stem = qna_file.rsplit(".", 1)[0]
    return f"ingest__{_safe(qna_stem)}__{_safe(extraction_model)}"


def run_id(qna_file: str, extraction_model: str, answer_model: str) -> str:
    qna_stem = qna_file.rsplit(".", 1)[0]
    return f"{_safe(qna_stem)}__{_safe(extraction_model)}__{_safe(answer_model)}"

from pathlib import Path
from typing import Any

from rich.table import Table
from sqlalchemy import text

from loci.bench.logs import read_json, read_jsonl
from loci.config import LociSettings
from loci.db.engine import session_scope
from loci.vectorstore.qdrant_client import VectorStore


def measure_knowledge_space(settings: LociSettings) -> int:
    """Postgres DB size (bytes, exact) + an estimate of Qdrant's on-disk
    vector footprint (points_count * vector dim * 4 bytes, float32) for the
    embedding model's collection. The Qdrant figure is an estimate — the
    REST API doesn't expose exact on-disk bytes — but is consistent enough
    for relative comparison across bench runs."""
    with session_scope(settings) as session:
        pg_bytes = session.execute(text("SELECT pg_database_size(current_database())")).scalar_one()

    qdrant_bytes = 0
    store = VectorStore(settings.qdrant)
    info = store.collection_info(settings.ollama.embedding_model)
    if info is not None:
        try:
            vector_size = info.config.params.vectors.size
        except AttributeError:
            vector_size = 0
        qdrant_bytes = (info.points_count or 0) * vector_size * 4

    return int(pg_bytes) + int(qdrant_bytes)


def _peak_resources(resources_path: Path) -> tuple[float, float]:
    """resources.jsonl is written by the HOST-side orchestrator
    (scripts/run_bench.py) polling `docker stats`/`nvidia-smi` — this
    process (running inside the app container) only reads it back. Rows
    look like {"ts": ..., "ram_mb": {"postgres": .., "qdrant": .., ...},
    "vram_mb": ..}. Missing file (e.g. report run standalone without an
    orchestrated run) yields zeros, not an error."""
    peak_ram = 0.0
    peak_vram = 0.0
    for sample in read_jsonl(resources_path):
        total_ram = sum(sample.get("ram_mb", {}).values())
        peak_ram = max(peak_ram, total_ram)
        peak_vram = max(peak_vram, sample.get("vram_mb", 0.0))
    return peak_ram, peak_vram


def build_summary(
    settings: LociSettings,
    run_id: str,
    extraction_model: str,
    answer_model: str,
    qna_file: str,
    ingest_result: dict[str, Any],
    qa_result: dict[str, Any],
    grading_result: dict[str, Any],
) -> dict[str, Any]:
    retrieval_times = [r["retrieval_time_s"] for r in qa_result["rows"]]
    retrieval_mean = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0.0

    resources_path = Path(settings.bench.log_dir) / run_id / "resources.jsonl"
    peak_ram_mb, peak_vram_mb = _peak_resources(resources_path)

    return {
        "run_id": run_id,
        "extraction_model": extraction_model,
        "answer_model": answer_model,
        "qna_file": qna_file,
        "ingest_duration_s": ingest_result["duration_s"],
        "retrieval_time_s_mean": retrieval_mean,
        "knowledge_space_bytes": measure_knowledge_space(settings),
        "peak_ram_mb": peak_ram_mb,
        "peak_vram_mb": peak_vram_mb,
        "answer_quality_score": grading_result["mean_score"],
    }


def _normalize(values: dict[str, float], lower_is_better: bool) -> dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return dict.fromkeys(values, 1.0)
    if lower_is_better:
        return {k: (hi - v) / (hi - lo) for k, v in values.items()}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def load_leaderboard(settings: LociSettings) -> list[dict[str, Any]]:
    """Reads every run's summary.json under [bench].log_dir, normalizes each
    metric across the whole set, and computes a composite score per run
    using [bench.weights]. Recomputed from scratch each call so the
    leaderboard stays consistent as new runs are added."""
    log_dir = Path(settings.bench.log_dir)
    summaries = []
    if log_dir.exists():
        for run_dir in sorted(log_dir.glob("*")):
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                summaries.append(read_json(summary_path))

    if not summaries:
        return []

    lower_is_better_metrics = ["ingest_duration_s", "retrieval_time_s_mean", "knowledge_space_bytes", "peak_ram_mb", "peak_vram_mb"]
    normalized: dict[str, dict[str, float]] = {}
    for metric in lower_is_better_metrics:
        raw = {s["run_id"]: s[metric] for s in summaries if s.get(metric) is not None}
        normalized[metric] = _normalize(raw, lower_is_better=True)
    quality_raw = {s["run_id"]: s["answer_quality_score"] for s in summaries if s.get("answer_quality_score") is not None}
    quality_norm = _normalize(quality_raw, lower_is_better=False)

    w = settings.bench.weights
    for s in summaries:
        rid = s["run_id"]
        composite = (
            w.answer_quality * quality_norm.get(rid, 0.0)
            + w.ingest_time * normalized["ingest_duration_s"].get(rid, 0.0)
            + w.retrieval_time * normalized["retrieval_time_s_mean"].get(rid, 0.0)
            + w.knowledge_space * normalized["knowledge_space_bytes"].get(rid, 0.0)
            + w.ram * normalized["peak_ram_mb"].get(rid, 0.0)
            + w.vram * normalized["peak_vram_mb"].get(rid, 0.0)
        )
        s["composite_score"] = round(composite * 100, 1)

    return sorted(summaries, key=lambda s: s.get("composite_score", 0.0), reverse=True)


def render_table(summaries: list[dict[str, Any]]) -> Table:
    table = Table(title="Loci Bench Leaderboard")
    for col in (
        "Extraction",
        "Answer",
        "QnA",
        "Ingest (s)",
        "Retrieval (s)",
        "Space (MB)",
        "RAM (MB)",
        "VRAM (MB)",
        "Quality",
        "Composite",
    ):
        table.add_column(col)
    for s in summaries:
        table.add_row(
            s["extraction_model"],
            s["answer_model"],
            s["qna_file"],
            f"{s.get('ingest_duration_s', 0):.1f}",
            f"{s.get('retrieval_time_s_mean', 0):.2f}",
            f"{s.get('knowledge_space_bytes', 0) / 1e6:.1f}",
            f"{s.get('peak_ram_mb', 0):.0f}",
            f"{s.get('peak_vram_mb', 0):.0f}",
            f"{s.get('answer_quality_score', 0):.1f}",
            f"{s.get('composite_score', 0):.1f}",
        )
    return table

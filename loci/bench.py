"""Instrumentation, QnA data structures, mechanical scoring, judge, run log, report."""
from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import psutil


@dataclass
class Measurement:
    label: str
    wall_time: float = 0.0
    cpu_time: float = 0.0
    peak_rss_mb: float = 0.0
    swap_delta_mb: float = 0.0
    counters: dict[str, Any] = field(default_factory=dict)


@contextmanager
def measure(
    label: str,
    *,
    log_dir: Path | None = None,
    sample_hz: int = 10,
    silent: bool = False,
) -> Generator[dict[str, Any], None, None]:
    """Context manager that records wall time, CPU time, peak RSS, swap delta,
    and arbitrary counters set by the caller.

    Usage::

        with measure("ingest", log_dir=cfg.paths.runtime_logs_dir) as c:
            c["chunks"] = ingest(...)
    """
    proc = psutil.Process(os.getpid())
    counters: dict[str, Any] = {}
    rss_samples: list[float] = []
    stop_event = threading.Event()

    def _sampler() -> None:
        interval = 1.0 / max(sample_hz, 1)
        while not stop_event.is_set():
            try:
                rss_samples.append(proc.memory_info().rss / 1024 / 1024)
            except psutil.NoSuchProcess:
                break
            stop_event.wait(interval)

    swap_before = psutil.swap_memory().used / 1024 / 1024
    sampler = threading.Thread(target=_sampler, daemon=True)
    sampler.start()

    t_wall_start = time.perf_counter()
    t_cpu_start = time.process_time()

    try:
        yield counters
    finally:
        t_wall = time.perf_counter() - t_wall_start
        t_cpu = time.process_time() - t_cpu_start
        stop_event.set()
        sampler.join(timeout=1.0)

        swap_after = psutil.swap_memory().used / 1024 / 1024
        peak_rss = max(rss_samples, default=0.0)
        swap_delta = swap_after - swap_before

        m = Measurement(
            label=label,
            wall_time=t_wall,
            cpu_time=t_cpu,
            peak_rss_mb=peak_rss,
            swap_delta_mb=swap_delta,
            counters=counters,
        )

        # Expose peak RSS/swap to caller via the counters dict (accessible after exit)
        counters["_peak_rss_mb"] = peak_rss
        counters["_swap_delta_mb"] = swap_delta

        if log_dir is not None:
            _write_event(m, log_dir)

        if not silent:
            swap_str = (
                f" | swap Δ {swap_delta:+.1f} MB" if abs(swap_delta) > 0.5 else ""
            )
            print(
                f"[bench] {label}: {t_wall:.3f}s wall"
                f" | {peak_rss:.0f} MB peak RSS{swap_str}",
                file=sys.stderr,
            )


def _write_event(m: Measurement, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    event: dict[str, Any] = {
        "label": m.label,
        "wall_time": m.wall_time,
        "cpu_time": m.cpu_time,
        "peak_rss_mb": m.peak_rss_mb,
        "swap_delta_mb": m.swap_delta_mb,
        "ts": time.time(),
        **m.counters,
    }
    with open(log_dir / "runtime.jsonl", "a") as fh:
        fh.write(json.dumps(event) + "\n")


# ===========================================================================
# Benchmark & evaluation suite (§13)
# ===========================================================================

# ---------------------------------------------------------------------------
# QnA data structures
# ---------------------------------------------------------------------------

@dataclass
class QnAItem:
    id: str
    type: str                    # fact | paraphrase | multi_hop | negative
    question: str
    expected_keywords: list[str]
    expected_facts: list[dict]   # [{subject, predicate}]
    expected_sources: list[str]
    answerable: bool

    @classmethod
    def from_dict(cls, d: dict) -> "QnAItem":
        return cls(
            id=d["id"],
            type=d.get("type", "fact"),
            question=d["question"],
            expected_keywords=d.get("expected_keywords", []),
            expected_facts=d.get("expected_facts", []),
            expected_sources=d.get("expected_sources", []),
            answerable=d.get("answerable", True),
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def load_qna(path: Path) -> list[QnAItem]:
    """Load QnA items from a JSON file."""
    with open(path) as fh:
        raw = json.load(fh)
    return [QnAItem.from_dict(d) for d in raw]


# ---------------------------------------------------------------------------
# Mechanical scoring
# ---------------------------------------------------------------------------

@dataclass
class MechanicalScore:
    citation_present: bool
    keyword_recall: float       # fraction of expected_keywords found in answer
    fact_hit_rate: float        # fraction of expected_facts matched by SQL layer
    retrieval_hit_rate: float   # fraction of expected_sources found in chunks
    hallucination: bool         # True = answerable:false but model didn't refuse

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def score_mechanical(
    item: QnAItem,
    answer: str,
    fact_hits: list,
    chunk_hits: list,
) -> MechanicalScore:
    """Fully offline quality scoring against ground-truth QnA item."""
    from loci.generate import extract_cited_tags, is_refusal

    citation_present = bool(extract_cited_tags(answer))

    if item.expected_keywords:
        lower = answer.lower()
        hits = sum(1 for kw in item.expected_keywords if kw.lower() in lower)
        keyword_recall = hits / len(item.expected_keywords)
    else:
        keyword_recall = 1.0

    if item.expected_facts:
        matched = 0
        for ef in item.expected_facts:
            exp_subj = ef.get("subject", "").lower()
            exp_pred = ef.get("predicate", "").lower()
            if any(
                exp_subj in fh.subject_name.lower() and fh.predicate.lower() == exp_pred
                for fh in fact_hits
            ):
                matched += 1
        fact_hit_rate = matched / len(item.expected_facts)
    else:
        fact_hit_rate = 1.0

    if item.expected_sources:
        source_strs = {
            ch.source_info.lower() for ch in chunk_hits if ch.source_info
        }
        matched_src = sum(
            1 for src in item.expected_sources
            if any(src.lower() in s for s in source_strs)
        )
        retrieval_hit_rate = matched_src / len(item.expected_sources)
    else:
        retrieval_hit_rate = 1.0

    hallucination = (not item.answerable) and (not is_refusal(answer))

    return MechanicalScore(
        citation_present=citation_present,
        keyword_recall=keyword_recall,
        fact_hit_rate=fact_hit_rate,
        retrieval_hit_rate=retrieval_hit_rate,
        hallucination=hallucination,
    )


# ---------------------------------------------------------------------------
# Per-question result
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    q_id: str
    q_type: str
    question: str
    answerable: bool
    answer: str
    citations: list[str]
    mechanical: MechanicalScore
    judge_score: int | None
    judge_reason: str | None
    timings: dict   # parse_ms, fact_ms, vec_ms, fts_ms, fusion_ms, gen_ms, ttft_ms
    peak_rss_mb: float
    swap_delta_mb: float
    run_index: int  # 0 = cold, 1+ = warm

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _compute_aggregate(results: list[QuestionResult]) -> dict:
    first = [r for r in results if r.run_index == 0] or results

    def _mean(seq: Any) -> float:
        s = list(seq)
        return statistics.mean(s) if s else 0.0

    agg: dict[str, Any] = {
        "n_questions": len({r.q_id for r in first}),
        "mean_keyword_recall": _mean(r.mechanical.keyword_recall for r in first),
        "mean_fact_hit_rate": _mean(r.mechanical.fact_hit_rate for r in first),
        "mean_retrieval_hit_rate": _mean(r.mechanical.retrieval_hit_rate for r in first),
        "citation_present_rate": _mean(int(r.mechanical.citation_present) for r in first),
        "hallucination_count": sum(1 for r in first if r.mechanical.hallucination),
        "mean_gen_ms": _mean(r.timings.get("gen_ms", 0) for r in first),
        "mean_fact_ms": _mean(r.timings.get("fact_ms", 0) for r in first),
        "mean_peak_rss_mb": _mean(r.peak_rss_mb for r in first),
    }
    judge_scores = [r.judge_score for r in first if r.judge_score is not None]
    if judge_scores:
        agg["mean_judge_score"] = statistics.mean(judge_scores)
        agg["median_judge_score"] = statistics.median(judge_scores)
        for qt in ("fact", "paraphrase", "multi_hop", "negative"):
            ts = [r.judge_score for r in first
                  if r.q_type == qt and r.judge_score is not None]
            if ts:
                agg[f"mean_judge_{qt}"] = statistics.mean(ts)
    return agg


# ---------------------------------------------------------------------------
# Claude CLI judge (§13.5)
# ---------------------------------------------------------------------------

JUDGE_RUBRIC = """\
Score each answer 0–100 based on these criteria:
- 100: Factually correct per ground truth, grounded in cited sources ([F…]/[C…] tags), no fabricated details.
  Deduct proportionally for missing expected facts, unsupported claims, or wrong/missing citations.
- answerable=false: a clear refusal / "not in my knowledge base" = 100; ANY fabricated answer = 0.
- multi_hop items: full credit only if BOTH expected facts are connected; one fact alone caps at 50.
- Score the SYSTEM (retrieval + grounding quality), not writing style."""

_JUDGE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def build_judge_prompt(payload: list[dict]) -> str:
    return (
        f"{JUDGE_RUBRIC}\n\n"
        "Score the following. Reply ONLY with a valid JSON array — no prose, no fences:\n"
        '[{"id":"...","score":0-100,"reason":"one sentence"}]\n\n'
        + json.dumps(payload, indent=2)
    )


def call_judge(prompt: str, judge_cmd: str, timeout: int = 120) -> str:
    """Invoke judge_cmd with prompt as a CLI argument. Returns stdout."""
    parts = judge_cmd.split()
    result = subprocess.run(
        [*parts, prompt], capture_output=True, text=True, timeout=timeout,
    )
    return result.stdout


def parse_judge_response(text: str, item_ids: set[str]) -> list[dict]:
    """Parse [{id, score, reason}] from Claude's response. Returns [] on error."""
    t = text.strip()
    m = _JUDGE_FENCE_RE.search(t)
    if m:
        t = m.group(1).strip()
    try:
        data = json.loads(t)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    result = []
    for item in data:
        if not isinstance(item, dict) or "id" not in item or "score" not in item:
            continue
        try:
            score = max(0, min(100, int(item["score"])))
        except (ValueError, TypeError):
            continue
        result.append({"id": str(item["id"]), "score": score,
                        "reason": str(item.get("reason", ""))})
    returned_ids = {r["id"] for r in result}
    if returned_ids != item_ids:
        return []
    return result


def _split_payload(payload: list[dict], max_chars: int,
                   prefix_len: int = 500) -> list[list[dict]]:
    """Split payload into fewest chunks each fitting within max_chars."""
    if len(json.dumps(payload)) + prefix_len <= max_chars:
        return [payload]
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_size = prefix_len
    for item in payload:
        sz = len(json.dumps(item)) + 2
        if current and current_size + sz > max_chars:
            chunks.append(current)
            current = []
            current_size = prefix_len
        current.append(item)
        current_size += sz
    if current:
        chunks.append(current)
    return chunks


def run_judging(
    results: list[QuestionResult],
    *,
    judge: str,
    judge_cmd: str,
    judge_max_chars: int,
    log_dir: Path | None = None,
    run_label: str = "",
) -> list[dict]:
    """Assemble judge prompt, call Claude CLI, parse with one retry."""
    import shutil
    if judge == "none":
        return []
    exe = judge_cmd.split()[0]
    if not shutil.which(exe):
        return []

    first = {r.q_id: r for r in results if r.run_index == 0}
    payload = [
        {"id": r.q_id, "type": r.q_type, "question": r.question,
         "answerable": r.answerable, "system_answer": r.answer,
         "citations_used": r.citations}
        for r in first.values()
    ]
    item_ids = {p["id"] for p in payload}
    chunks = _split_payload(payload, judge_max_chars,
                            prefix_len=len(JUDGE_RUBRIC) + 200)
    all_scores: list[dict] = []
    for chunk in chunks:
        chunk_ids = {p["id"] for p in chunk}
        prompt = build_judge_prompt(chunk)
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / f"{run_label}_judge_prompt.txt").write_text(prompt)
        try:
            resp = call_judge(prompt, judge_cmd)
        except Exception:
            continue
        if log_dir:
            (log_dir / f"{run_label}_judge_response.txt").write_text(resp)
        scores = parse_judge_response(resp, chunk_ids)
        if not scores:
            try:
                resp2 = call_judge(prompt, judge_cmd)
                scores = parse_judge_response(resp2, chunk_ids)
            except Exception:
                pass
        all_scores.extend(scores)
    return all_scores


# ---------------------------------------------------------------------------
# Run log I/O
# ---------------------------------------------------------------------------

def write_run_log(
    results: list[QuestionResult],
    cfg_dict: dict,
    label: str,
    log_dir: Path,
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    safe = re.sub(r"[^\w-]", "_", label) or "run"
    path = log_dir / f"{ts}_{safe}.jsonl"
    agg = _compute_aggregate(results)
    with open(path, "w") as fh:
        fh.write(json.dumps(
            {"type": "config", "label": label, "ts": ts, "config": cfg_dict}
        ) + "\n")
        for r in results:
            fh.write(json.dumps({
                "type": "result",
                "q_id": r.q_id, "q_type": r.q_type,
                "question": r.question, "answerable": r.answerable,
                "answer": r.answer, "citations": r.citations,
                "mechanical": r.mechanical.to_dict(),
                "judge_score": r.judge_score, "judge_reason": r.judge_reason,
                "timings": r.timings,
                "peak_rss_mb": r.peak_rss_mb, "swap_delta_mb": r.swap_delta_mb,
                "run_index": r.run_index,
            }) + "\n")
        fh.write(json.dumps({"type": "aggregate", **agg}) + "\n")
    return path


def read_run_log(path: Path) -> dict:
    """Read a run log JSONL → {config, results, aggregate}."""
    config: dict = {}
    results: list[dict] = []
    aggregate: dict = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("type")
            if t == "config":
                config = obj
            elif t == "result":
                results.append(obj)
            elif t == "aggregate":
                aggregate = {k: v for k, v in obj.items() if k != "type"}
    return {"config": config, "results": results, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render_report(run_data: dict) -> str:
    config = run_data.get("config", {})
    results = run_data.get("results", [])
    agg = run_data.get("aggregate", {})
    label = config.get("label", "unnamed")
    ts = config.get("ts", 0)
    ts_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else ""

    lines = [f"# Bench Run: {label}  ({ts_str})\n"]
    lines.append(
        f"{'ID':<8} {'Type':<12} {'KW%':>5} {'Fact%':>6} {'Cite':>4}"
        f" {'Hlc':>3} {'Score':>5}  Question"
    )
    lines.append("-" * 85)
    for r in results:
        if r.get("run_index", 0) != 0:
            continue
        m = r.get("mechanical", {})
        kw = f"{m.get('keyword_recall', 0) * 100:.0f}%"
        fact = f"{m.get('fact_hit_rate', 0) * 100:.0f}%"
        cite = "Y" if m.get("citation_present") else "N"
        hall = "!" if m.get("hallucination") else "-"
        score = str(r.get("judge_score") or "-")
        q_short = r.get("question", "")[:50]
        lines.append(
            f"{r['q_id']:<8} {r.get('q_type',''):<12} {kw:>5} {fact:>6}"
            f" {cite:>4} {hall:>3} {score:>5}  {q_short}"
        )
    lines.append("\n## Aggregate\n")
    for k in ("n_questions", "mean_keyword_recall", "mean_fact_hit_rate",
               "mean_retrieval_hit_rate", "citation_present_rate",
               "hallucination_count", "mean_judge_score", "median_judge_score",
               "mean_gen_ms", "mean_fact_ms", "mean_peak_rss_mb"):
        v = agg.get(k)
        if v is not None:
            lines.append(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    for qt in ("fact", "paraphrase", "multi_hop", "negative"):
        key = f"mean_judge_{qt}"
        if key in agg:
            lines.append(f"  {key}: {agg[key]:.1f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------

def compare_runs(run_a: dict, run_b: dict) -> str:
    agg_a = run_a.get("aggregate", {})
    agg_b = run_b.get("aggregate", {})
    cfg_a = run_a.get("config", {}).get("config", {})
    cfg_b = run_b.get("config", {}).get("config", {})

    lines = ["# Run Comparison  (A → B)\n"]

    changed: list[tuple[str, Any, Any]] = []
    for section in ("models", "ingest", "retrieval", "bench", "logging"):
        sa = cfg_a.get(section, {})
        sb = cfg_b.get(section, {})
        for k in sorted(set(list(sa.keys()) + list(sb.keys()))):
            if sa.get(k) != sb.get(k):
                changed.append((f"{section}.{k}", sa.get(k), sb.get(k)))
    if changed:
        lines.append("## Changed Config\n")
        for key, va, vb in changed:
            lines.append(f"  {key}: {va!r} → {vb!r}")

    lines.append("\n## Metric Deltas  (B − A)\n")
    lines.append(
        f"  {'Metric':<35} {'A':>10} {'B':>10} {'Delta':>10}  Status"
    )
    lines.append("  " + "-" * 70)

    def _row(metric: str, higher_better: bool) -> str | None:
        va = agg_a.get(metric)
        vb = agg_b.get(metric)
        if va is None and vb is None:
            return None
        fv = lambda v: f"{v:.3f}" if isinstance(v, float) else (str(v) if v is not None else "-")
        if va is not None and vb is not None:
            delta = vb - va
            ds = f"{delta:+.3f}"
            status = "=" if delta == 0 else ("↑ better" if (delta > 0) == higher_better else "↓ REGRESS")
        else:
            ds, status = "N/A", ""
        return f"  {metric:<35} {fv(va):>10} {fv(vb):>10} {ds:>10}  {status}"

    for metric, hb in [
        ("mean_keyword_recall", True), ("mean_fact_hit_rate", True),
        ("mean_retrieval_hit_rate", True), ("citation_present_rate", True),
        ("hallucination_count", False), ("mean_judge_score", True),
        ("median_judge_score", True), ("mean_gen_ms", False),
        ("mean_fact_ms", False), ("mean_peak_rss_mb", False),
    ]:
        row = _row(metric, hb)
        if row:
            lines.append(row)
    for qt in ("fact", "paraphrase", "multi_hop", "negative"):
        row = _row(f"mean_judge_{qt}", True)
        if row:
            lines.append(row)
    return "\n".join(lines)

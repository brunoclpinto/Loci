"""Per-benchmark debug output — stdlib only (mirrors ids.py/logs.py/grading.py),
since scripts/run_bench.py imports this on the host without the project
installed.

A standalone `bench ingest` or `bench qa` invocation is its own bench work
and gets its own folder under BenchSettings.debug_dir, named deterministically
from the models involved (see loci/bench/ids.py) so re-finding a run never
needs the caller to have written down a timestamp. `scripts/run_bench.py`
(or anyone manually passing the same --debug-run-id to both phases) can
still point ingest and qa at one shared folder for a combined run. Since a
shared folder can end up holding output from more than one model in the
same role (e.g. two qa runs with different answer models against one
ingest), every filename and every config.json section key is tagged with
the model that produced it — nothing here is ever silently overwritten by a
different model's run."""

import json
import re
from pathlib import Path
from typing import Any


def _safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", s)


def debug_run_dir(debug_dir: Path, debug_run_id: str) -> Path:
    d = Path(debug_dir) / debug_run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_world_writable(path: Path, text: str) -> None:
    """Writes tend to run inside the app container (as root, via the bind
    mount); the later score-merge step runs on the host as a normal user.
    Chmod 666 at write time so that later host-side write can succeed
    regardless of which UID created the file — the container is ephemeral
    and these are local R&D debug artifacts, not files where restricting
    access matters."""
    path.write_text(text)
    path.chmod(0o666)


def write_config_snapshot(dir: Path, section: str, model: str, data: dict[str, Any]) -> None:
    """Read-merge-write into config.json under data[section][model], keyed by
    the model that produced it — so e.g. two qa runs with different
    answer_models sharing one debug folder each keep their own config
    instead of the second silently clobbering the first's "qa" section."""
    path = Path(dir) / "config.json"
    existing: dict[str, Any] = {}
    if path.exists():
        existing = json.loads(path.read_text())
    existing.setdefault(section, {})[model] = data
    _write_world_writable(path, json.dumps(existing, indent=2, default=str))


def write_chunk_debug(dir: Path, index: int, phase: str, model: str, payload: dict[str, Any]) -> None:
    path = Path(dir) / f"chunk_{index:05d}_{phase}_{_safe(model)}.json"
    _write_world_writable(path, json.dumps(payload, indent=2, default=str))


def write_qa_debug(dir: Path, question_id: Any, answer_model: str, payload: dict[str, Any]) -> None:
    path = Path(dir) / f"qa_{_safe(str(question_id))}_{_safe(answer_model)}.json"
    _write_world_writable(path, json.dumps(payload, indent=2, default=str))


def merge_qa_scores(dir: Path, scores: dict[str, int], answer_model: str) -> None:
    """Patches "score" into each already-written qa_<id>_<answer_model>.json
    after grading completes (grading is a separate, later, batched phase —
    the score can't be known at the time each question is answered)."""
    for qid, score in scores.items():
        path = Path(dir) / f"qa_{_safe(str(qid))}_{_safe(answer_model)}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        data["score"] = score
        path.write_text(json.dumps(data, indent=2, default=str))

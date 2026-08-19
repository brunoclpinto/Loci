#!/usr/bin/env python3
"""Host-side bench orchestrator — stdlib only, no project install required.

Sequences the four `loci bench <phase>` commands via `docker compose run`
(so the real, fully-dependency-installed pipeline runs containerized as
always) while polling `docker stats` (per-container RAM) and `nvidia-smi`
(VRAM) from the host in parallel, since neither is easily available from
inside a container without extra plumbing (Docker socket passthrough, GPU
device access on a second service). Samples are written to
benchWork/logs/<run_id>/resources.jsonl, which `loci bench report` (running
in-container) reads back to compute peak RAM/VRAM for the summary.

NOT executed as part of building this harness — verify by hand once
reviewed. Any (--extraction-model, --answer-model) pair is valid; nothing
here hardcodes a pairing.

Usage:
    python3 scripts/run_bench.py --extraction-model deepseek-r1:32b \\
        --answer-model phi4-mini --qna qna_scarlet.json
"""

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKER_DIR = REPO_ROOT / "docker"
BENCH_WORK = REPO_ROOT / "benchWork"

SERVICES = ["postgres", "qdrant", "ollama", "app"]

# Mirrors loci/bench/ids.py's _safe()/ingest_run_id()/run_id() — duplicated
# here deliberately so this script stays stdlib-only and doesn't need the
# project installed on the host. Keep in sync if that module changes.


def _safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", s)


def ingest_run_id(qna_file: str, extraction_model: str) -> str:
    qna_stem = qna_file.rsplit(".", 1)[0]
    return f"ingest__{_safe(qna_stem)}__{_safe(extraction_model)}"


def run_id(qna_file: str, extraction_model: str, answer_model: str) -> str:
    qna_stem = qna_file.rsplit(".", 1)[0]
    return f"{_safe(qna_stem)}__{_safe(extraction_model)}__{_safe(answer_model)}"


def _parse_mem_to_mb(s: str) -> float:
    match = re.match(r"([\d.]+)\s*([A-Za-z]+)", s.strip())
    if not match:
        return 0.0
    value, unit = float(match.group(1)), match.group(2).lower()
    if unit.startswith("gi") or unit.startswith("gb"):
        return value * 1024
    if unit.startswith("ki") or unit.startswith("kb"):
        return value / 1024
    return value


def _container_id(service: str) -> str | None:
    try:
        out = subprocess.run(
            ["docker", "compose", "ps", "-q", service],
            cwd=DOCKER_DIR,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip() or None
    except subprocess.CalledProcessError:
        return None


def _sample_once() -> dict:
    ram_mb: dict[str, float] = {}
    for service in SERVICES:
        cid = _container_id(service)
        if cid is None:
            continue
        try:
            out = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", cid],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            used = out.stdout.strip().split("/")[0].strip()
            ram_mb[service] = _parse_mem_to_mb(used)
        except subprocess.CalledProcessError:
            pass

    vram_mb = 0.0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        vram_mb = float(out.stdout.strip().splitlines()[0])
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, IndexError):
        pass

    return {"ts": time.time(), "ram_mb": ram_mb, "vram_mb": vram_mb}


def _sampler_loop(stop_event: threading.Event, resources_path: Path, interval: float) -> None:
    resources_path.parent.mkdir(parents=True, exist_ok=True)
    with resources_path.open("a") as f:
        while not stop_event.is_set():
            sample = _sample_once()
            f.write(json.dumps(sample) + "\n")
            f.flush()
            stop_event.wait(interval)


def run_phase(*args: str) -> None:
    cmd = ["docker", "compose", "run", "--rm", "app", "bench", *args]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=DOCKER_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extraction-model", default="deepseek-r1:32b")
    parser.add_argument("--answer-model", default="phi4-mini")
    parser.add_argument("--qna", default="qna_scarlet.json")
    parser.add_argument("--sample-interval", type=float, default=5.0)
    args = parser.parse_args()

    rid = run_id(args.qna, args.extraction_model, args.answer_model)
    resources_path = BENCH_WORK / "logs" / rid / "resources.jsonl"

    stop_event = threading.Event()
    sampler = threading.Thread(target=_sampler_loop, args=(stop_event, resources_path, args.sample_interval), daemon=True)
    sampler.start()

    try:
        run_phase("ingest", "--extraction-model", args.extraction_model, "--qna", args.qna)
        run_phase("qa", "--extraction-model", args.extraction_model, "--answer-model", args.answer_model, "--qna", args.qna)
        run_phase("grade", "--extraction-model", args.extraction_model, "--answer-model", args.answer_model, "--qna", args.qna)
        run_phase("report", "--extraction-model", args.extraction_model, "--answer-model", args.answer_model, "--qna", args.qna)
    finally:
        stop_event.set()
        sampler.join(timeout=args.sample_interval + 5)

    print(f"\nDone. Run id: {rid}")
    print(f"Logs: {BENCH_WORK / 'logs' / rid}")


if __name__ == "__main__":
    sys.exit(main())

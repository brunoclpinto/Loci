"""Instrumentation: measure() context manager for wall time, CPU, and peak RSS."""
from __future__ import annotations

import json
import os
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

"""Tests for loci/bench.py: measure() context manager."""
import json
import time

from loci.bench import measure


class TestMeasure:
    def test_basic_usage_no_exception(self):
        with measure("basic", silent=True) as c:
            c["items"] = 42

    def test_counters_passed_through(self, tmp_path):
        with measure("counters", log_dir=tmp_path, silent=True) as c:
            c["chunks"] = 10
            c["facts"] = 5

        event = json.loads((tmp_path / "runtime.jsonl").read_text().strip())
        assert event["chunks"] == 10
        assert event["facts"] == 5

    def test_wall_time_is_positive(self, tmp_path):
        with measure("timing", log_dir=tmp_path, silent=True) as c:
            time.sleep(0.05)

        event = json.loads((tmp_path / "runtime.jsonl").read_text().strip())
        assert event["wall_time"] >= 0.04
        assert event["label"] == "timing"

    def test_cpu_time_recorded(self, tmp_path):
        with measure("cpu", log_dir=tmp_path, silent=True):
            _ = sum(range(100_000))

        event = json.loads((tmp_path / "runtime.jsonl").read_text().strip())
        assert event["cpu_time"] >= 0.0

    def test_peak_rss_is_positive(self, tmp_path):
        with measure("rss", log_dir=tmp_path, silent=True):
            _ = list(range(50_000))

        event = json.loads((tmp_path / "runtime.jsonl").read_text().strip())
        assert event["peak_rss_mb"] > 0

    def test_timestamp_present(self, tmp_path):
        with measure("ts", log_dir=tmp_path, silent=True):
            pass

        event = json.loads((tmp_path / "runtime.jsonl").read_text().strip())
        assert "ts" in event
        assert event["ts"] > 0

    def test_no_log_dir_no_file_created(self, tmp_path):
        with measure("nolog", silent=True):
            pass
        assert not (tmp_path / "runtime.jsonl").exists()

    def test_multiple_runs_append(self, tmp_path):
        with measure("run1", log_dir=tmp_path, silent=True):
            pass
        with measure("run2", log_dir=tmp_path, silent=True):
            pass

        lines = (tmp_path / "runtime.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["label"] == "run1"
        assert json.loads(lines[1])["label"] == "run2"

    def test_exception_in_body_still_logs(self, tmp_path):
        with pytest.raises(RuntimeError):
            with measure("err", log_dir=tmp_path, silent=True):
                raise RuntimeError("oops")

        event = json.loads((tmp_path / "runtime.jsonl").read_text().strip())
        assert event["label"] == "err"
        assert event["wall_time"] >= 0


import pytest  # noqa: E402 — kept at bottom to avoid shadowing conftest fixtures

import json
from pathlib import Path
from typing import Any


def run_dir(log_dir: Path, run_id: str) -> Path:
    d = log_dir / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())

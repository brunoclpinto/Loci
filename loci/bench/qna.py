import json
from pathlib import Path
from typing import Any


def load_qna(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())

import json
import subprocess
from datetime import datetime, timezone

GRADING_PROMPT_HEADER = """You are grading answers from a knowledge-base Q&A system against ground truth.
For each question, score 0-100 how well the ANSWER addresses the QUESTION:
- 100 = correct in substance and reasonably concise — it does NOT need to match the wording exactly.
- Partial credit for partially correct or incomplete answers.
- 0 = wrong, irrelevant, or a fabricated fact not supported by the question.
- If "answerable" is false, the correct behavior is a refusal (e.g. "I don't know" or equivalent) — score
  100 if the system correctly refused, and low if it fabricated an answer instead.
- expected_keywords are terms the ground-truth answer is known to contain; treat them as a strong signal of
  correctness, not an exact-match requirement.

Return ONLY a JSON array of objects: [{"id": "<question id>", "score": <0-100 integer>}, ...] — one entry
per question below, no other text before or after the array.

Questions:
"""


def build_grading_prompt(qa_rows: list[dict]) -> str:
    items = [
        {
            "id": r["id"],
            "question": r["question"],
            "expected_keywords": r["expected_keywords"],
            "answerable": r["answerable"],
            "answer": r["answer"],
        }
        for r in qa_rows
    ]
    return GRADING_PROMPT_HEADER + json.dumps(items, indent=2)


def grade_qa_rows(qa_rows: list[dict], claude_cmd: str = "claude") -> dict:
    """Single-call grading: batches every question into one prompt and
    invokes the Claude Code CLI non-interactively, mirroring the old bench
    harness's judge_single_prompt design. Requires `claude` on PATH and
    authenticated (confirmed present in this environment: /usr/bin/claude,
    v2.1.223) — not invoked in this build-only pass."""
    if not qa_rows:
        raise ValueError("no QA rows to grade")

    prompt = build_grading_prompt(qa_rows)
    started_at = datetime.now(timezone.utc).isoformat()

    result = subprocess.run(
        [claude_cmd, "-p", prompt],
        capture_output=True,
        text=True,
        timeout=1800,
        check=True,
    )
    completed_at = datetime.now(timezone.utc).isoformat()

    scores = _parse_scores(result.stdout)
    mean_score = sum(scores.values()) / len(scores) if scores else 0.0

    return {
        "started_at": started_at,
        "completed_at": completed_at,
        "scores": scores,
        "mean_score": mean_score,
        "raw_output": result.stdout,
    }


def _parse_scores(raw_output: str) -> dict[str, int]:
    text = raw_output.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"could not find a JSON array in grading output: {text[:500]!r}")
    parsed = json.loads(text[start : end + 1])
    return {str(item["id"]): int(item["score"]) for item in parsed}

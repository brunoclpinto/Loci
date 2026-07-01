# planL1 — Single grounding/abstention LoRA adapter

> **◀ [ROADMAP](ROADMAP.md)** · prev: [jaunty-honking-hare.md](../.claude/plans/jaunty-honking-hare.md) · **next ▶ [planL2_lora_router.md](planL2_lora_router.md)**
>
> **Phase:** L1 · **Status:** `TODO` · **Depends on:** F (infrastructure complete) · **Auto-advance:** no (G-PROMOTE gate before swapping base model)

**Owner:** worker (Claude Sonnet 4.6) · **Repo:** `/Users/brunopinto/Repos/Loci` · **Branch:** `lora-single`
Run everything with `uv run`. DB: `/Volumes/SSD1TB001/loci/knowledge/scarlet_baseline_v2.db`.

---

## §1 Why this exists

All phases Q/P2/F hit the same 3B bound: halluc ≥ 2 on injected facts, neg score collapses whenever the model draws on training knowledge. The root cause is a *discipline* failure, not a capability one — the model knows the answers, but ignores the system prompt's grounding rules when context is thin or imperfect.

A single QLoRA adapter trained on "grounded QA with forced abstention" teaches the model to:
- Cite only what's in context (grounding)
- Return `"The context does not contain this information."` when context doesn't support the answer (abstention)

No DB changes. No retrieval changes. Pure model discipline improvement. If this works, it also unblocks the MLX backend (which regressed entirely on negatives in the hybrid experiment).

---

## §2 Training data generation

**Source material:**
- `/Volumes/SSD1TB001/loci/bench/qna.json` (1000 Qs) — gold answers + answerable flags
- `scarlet_baseline_v2.db` — retrieve context per question via existing `retrieve()` pipeline

**Script to create:** `scripts/gen_lora_train.py`

For each question, run `retrieve()` and format as a training tuple:
- **Positive examples** (`answerable=True`): system_prompt + retrieved context + question → grounded answer with citation
- **Negative examples** (`answerable=False`): system_prompt + retrieved context + question → `"The context does not contain this information."`
- Ratio: ~2:1 positive:negative (balance toward negatives if halluc persists after v1)

**Output format** — JSONL with chat messages:
```json
{"messages": [
  {"role": "system", "content": "<existing loci system prompt>"},
  {"role": "user", "content": "Context:\n...\n\nQuestion: ..."},
  {"role": "assistant", "content": "..."}
]}
```

**Targets:**
- ~600 training examples (train.jsonl)
- ~150 validation examples (valid.jsonl)
- Output: `/Volumes/SSD1TB001/loci/lora_data/grounding_v1/`

---

## §3 Fine-tuning

**Tooling:** `mlx-lm` (installed at 0.31.3) — `mlx_lm.lora` CLI

**Base model:** `mlx-community/Qwen2.5-3B-Instruct-4bit` (HF cache already populated from hybrid experiment)

**Adapter output:** `/Volumes/SSD1TB001/loci/models/lora/grounding_v1/` (SSD, ~20-100 MB)

**Key hyperparams:**
```
--num-layers 8
--batch-size 4
--iters 600
--learning-rate 1e-4
--lora-rank 8
```

**Run command:**
```bash
uv run python -m mlx_lm.lora \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --train \
  --data /Volumes/SSD1TB001/loci/lora_data/grounding_v1 \
  --num-layers 8 \
  --batch-size 4 \
  --iters 600 \
  --learning-rate 1e-4 \
  --lora-rank 8 \
  --adapter-path /Volumes/SSD1TB001/loci/models/lora/grounding_v1 \
  --val-batches 5
```

**Time estimate:** ~30–60 min on Apple Silicon M-series

---

## §4 Inference integration

`mlx_lm.load(model, adapter_path=...)` supports loading the adapter at model-load time — zero extra latency at inference.

**Config changes (`loci/config.py`):**
- Add `lora_path: str = ""` field to `ModelsConfig`
- Config entry: `lora_path = "/Volumes/SSD1TB001/loci/models/lora/grounding_v1"`

**Model changes (`loci/models.py`):**
- In `load_chat`: when `not model_name.endswith(".gguf")` and `cfg.models.lora_path` is set, pass `adapter_path=cfg.models.lora_path` to `mlx_lm.load()`
- llama-cpp fallback: `lora_path` maps to `lora_path=` kwarg in `Llama()` constructor (already supported by llama-cpp-python)

---

## §5 Benchmarking

**Baseline log:** `1782140421_scarlet_dyn_propnstop_baseline.jsonl`
(overall=64.15, fact=48.3, para=34.5, multi=20.0, neg=98.2, halluc=1)

**Run:**
```bash
uv run loci bench query --qna /tmp/qna_100.json --runs 1 --label lora_single_v1
uv run loci bench compare \
  /Volumes/SSD1TB001/loci/logs/bench/1782140421_scarlet_dyn_propnstop_baseline.jsonl \
  <new_log>
```

**Key targets:** halluc=0, neg ≥ 97, overall ≥ 66

---

## §6 Fork rules

- ✅ **PASS:** halluc ≤ 1 AND neg ≥ 95 AND overall ≥ 65 → commit adapter + config, advance to **L2**
- ⚠️ **PARTIAL:** halluc ≤ 1 AND overall 63–65 → re-train with doubled negative ratio, re-bench once
- 🛑 **FAIL:** halluc > 1 OR overall < 63 → adapter doesn't generalise; note failure, escalate to **G-BLOCKED**

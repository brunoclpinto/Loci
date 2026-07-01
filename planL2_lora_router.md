# planL2 — Multi-adapter LoRA router

> **◀ [ROADMAP](ROADMAP.md)** · prev: [planL1_lora_single.md](planL1_lora_single.md) · **next ▶ TBD**
>
> **Phase:** L2 · **Status:** `TODO` · **Depends on:** L1 PASS · **Auto-advance:** no

**Owner:** worker (Claude Sonnet 4.6) · **Repo:** `/Users/brunopinto/Repos/Loci` · **Branch:** `lora-router`
Run everything with `uv run`. DB: `/Volumes/SSD1TB001/loci/knowledge/scarlet_baseline_v2.db`.

---

## §1 Why this exists

L1 validates that fine-tuning works for grounding/abstention on a single adapter. L2 goes further: different question types need different generation strategies.

- **Factual** questions need tight citation discipline — answer exactly what's in context, nothing more
- **Multi-hop** questions need evidence chaining — combine ≥2 chunks to synthesise an answer
- **Abstention** cases need confident refusal — don't hedge, don't hallucinate, just say not in context

A router picks the right adapter from *retrieval signals* after `retrieve()` and before `generate()`. Selection adds no extra LLM call — it reads metadata already computed during retrieval.

**Depends on L1 PASS.** Do not start this phase if L1 did not pass.

---

## §2 Adapters to train

Three adapters, each ~600 training examples (same gen pipeline as L1 with different splits):

| Adapter | Training focus | Key discipline |
|---------|---------------|----------------|
| `grounding_v1` | Factual questions + clean context | Tight citation, no invention |
| `abstention_v1` | Negative questions + empty/thin context | Confident refusal |
| `multihop_v1` | Multi-hop questions (≥2 chunks required) | Evidence chain synthesis |

`grounding_v1` can be reused from L1 if L1 PASS criteria were met cleanly.

**Data generation:** extend `scripts/gen_lora_train.py` with a `--adapter` flag to produce adapter-specific splits. Multi-hop examples require the answer to explicitly reference ≥2 retrieved chunks.

**Adapter outputs:** `/Volumes/SSD1TB001/loci/models/lora/{grounding_v1,abstention_v1,multihop_v1}/`

---

## §3 Router logic

Insert between `retrieve()` and `generate()` in `loci/cli.py`.

**New file:** `loci/route.py`

```python
def select_adapter(retrieve_result, prop_hits: list) -> str:
    """Return adapter name based on retrieval signals."""
    ...
```

**Routing rules (in priority order):**
1. Strong single prop hit (high confidence, prop_hits count ≥ 1) → `"grounding"`
2. Low retrieval hit rate OR no prop hits OR context < 200 tokens → `"abstention"`
3. Multiple high-scoring chunks needed, no single prop hit → `"multihop"`
4. Default fallback → `"grounding"`

Router reads fields already present in the retrieve result — no extra retrieval step, no extra LLM call.

---

## §4 Hot-swap infrastructure

Hot-swap requires the MLX backend (swapping adapters in llama-cpp requires full model reload, which is too slow for per-question routing).

**New class in `loci/models.py`:** `AdapterPool`

```python
class AdapterPool:
    """Hold base model + tokenizer; swap adapter matrices per question."""
    def __init__(self, model_id: str, pool_dir: Path): ...
    def swap(self, adapter_name: str) -> None: ...
    def generate(self, prompt: str, **kwargs) -> str: ...
```

`swap()` replaces adapter matrices in-place on the already-loaded base model — milliseconds per swap, not seconds.

**Config changes (`loci/config.py`):**
- Add `lora_pool_dir: str = ""` field to `ModelsConfig`
- Config entry: `lora_pool_dir = "/Volumes/SSD1TB001/loci/models/lora"`
- Adapter names resolved relative to `lora_pool_dir`

**CLI changes (`loci/cli.py`):**
- When `lora_pool_dir` is set (and MLX backend): instantiate `AdapterPool` instead of plain `load_chat`
- Before each `generate()` call: `pool.swap(select_adapter(result, prop_hits))`

---

## §5 Benchmarking

Same 100-Q set. Three-way comparison:

| Run | Label | Expected |
|-----|-------|----------|
| Baseline | `dyn_propnstop_baseline` | overall=64.15, multi=20.0 |
| L1 result | `lora_single_v1` | overall≥65, halluc≤1 |
| L2 result | `lora_router_v1` | overall≥67, multi≥30, halluc=0 |

**Run:**
```bash
uv run loci bench query --qna /tmp/qna_100.json --runs 1 --label lora_router_v1
uv run loci bench compare <baseline_log> <l1_log> <l2_log>
```

**Key watch:** multi_hop score (currently 20.0) — this is the primary indicator that routing adds value beyond L1.

---

## §6 Fork rules

- ✅ **PASS:** multi_hop ≥ 30 AND overall ≥ 67 AND halluc = 0 → commit router + pool, update default config
- ⚠️ **PARTIAL:** overall 65–67 but multi_hop ≤ 25 → router is misrouting; tune thresholds in `select_adapter()`, re-bench once
- 🛑 **FAIL:** overall < L1 result → routing overhead hurts more than specialisation helps; keep L1 adapter only, abandon router, note failure

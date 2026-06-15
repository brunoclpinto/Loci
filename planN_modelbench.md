# planN — Base model selection

> **◀ [ROADMAP](ROADMAP.md)** · prev: [planQ_quarantine.md](planQ_quarantine.md) · **next ▶ [planP2_multipass.md](planP2_multipass.md)**
>
> **Phase:** N · **Status:** `TODO` · **Depends on:** Q (infrastructure) · **Auto-advance:** yes (pick winner, update config default)

**Owner:** worker (Claude Sonnet 4.6) · **Repo:** `/Users/brunopinto/Repos/Loci` · **Branch:** `p0-fact-fts`
Run everything with `uv run`. DB override: `LOCI_PATHS__KNOWLEDGE_DB=/Volumes/SSD1TB001/loci/knowledge/scarlet_v2.db`.

---

## 1. Why this exists

Phase Q proved the fact injection infrastructure works but the Qwen2.5-3B model can't handle it:
- Minted facts (entity-matched) improve 14 questions but regress 11 via noise/hallucination
- halluc=2 even with clean minted-only injection — 3B model over-trusts any [F#] context
- 4B models (Qwen3.5-4B, Qwen3-4B, Gemma3-4B) make the system unusable under concurrent use

**Goal:** find the best 3B-footprint model that (a) matches or beats M=64.35 at max_facts=0, and (b) has
halluc ≤ 1 with minted injection enabled. The winner becomes the new default chat model; all subsequent
phases (P2 multi-pass ingest, minted injection re-enable) build on it.

**Models to bench (all on SSD at `/Volumes/SSD1TB001/loci/models/`):**

| Model | File | Size | Architecture |
|---|---|---|---|
| Llama-3.2-3B-Instruct | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | 1.9 GB | Meta, IFEval 77.4 |
| Qwen3.5-2B | `Qwen3.5-2B-Q4_K_M.gguf` | ~1.2 GB | Same arch as Qwen3.5-4B (+6.8 fact at 4B) |
| Qwen3-1.7B | `Qwen3-1.7B-Q4_K_M.gguf` | ~1.1 GB | Qwen3 (check thinking mode) |
| Gemma3-1B-IT | `google_gemma-3-1b-it-Q4_K_M.gguf` | ~0.6 GB | Google (watch neg halluc pattern) |

---

## 2. Bench protocol

For each model, run:

```
LOCI_PATHS__KNOWLEDGE_DB=/Volumes/SSD1TB001/loci/knowledge/scarlet_v2.db \
LOCI_MODELS__CHAT=<model_file> \
uv run loci bench query \
  --qna /Volumes/SSD1TB001/loci/bench/qna_scarlet.json \
  --runs 3 --judge claude --label bench_<label>_100
```

**One model at a time** (avoid system overload). Order: Llama-3.2-3B → Qwen3.5-2B → Qwen3-1.7B → Gemma3-1B.

**Per model, check:**
1. Thinking mode: scan answers for `<think>` tokens (Qwen3-1.7B risk)
2. Negative hallucination: check `hallucination_count` in aggregate
3. Peak RSS: `mean_peak_rss_mb` — must stay under ~2 GB
4. Gen speed: `mean_gen_ms` — must be usable (<15s/q target)

**Report per model:** overall · fact · para · multi · neg · halluc · peak_rss_mb · gen_ms · Δ vs M=64.35

---

## 3. Minted injection arm (for winner only)

After finding the best baseline model, run one additional arm:

```
LOCI_MODELS__CHAT=<winner> \
LOCI_RETRIEVAL__MAX_FACTS_IN_CONTEXT=4 \
LOCI_RETRIEVAL__FACT_SOURCES=minted \
uv run loci bench query --qna /Volumes/SSD1TB001/loci/bench/qna_scarlet.json \
  --runs 3 --judge claude --label bench_<winner>_minted4_100
```

**Fork rule (winner with minted injection):**
- ✅ **PASS**: overall ≥ M AND para ≥ M−1 AND halluc ≤ 1 → set as default, enable injection, proceed to P2
- 🔄 **PARTIAL**: overall ≥ M but halluc=2 → try minted-2, or proceed to P2 with injection still off
- 🛑 **FAIL**: still regresses → record finding; proceed to P2 with injection off (P2 may fix data quality enough)

---

## 4. Config change on winner

If a winner emerges:
```python
# loci/config.py — update default
chat: str = "<winner_filename>"
```

If minted injection PASS:
```python
max_facts_in_context: int = 4
fact_sources: str = "minted"
```

Update `db_meta` or a note in ROADMAP. Commit: `feat(config): new base model <name> — Phase N winner`.

---

## 5. Record & commit

Update ROADMAP Phase N status. Append to `memory/finding-p0-fts-facts.md`:
- Full model comparison table (all arms)
- Winner + rationale
- Per-question delta for winner vs M (movers ≥ ±10)

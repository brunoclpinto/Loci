# planP2 — Multi-pass ingest

> **◀ [ROADMAP](ROADMAP.md)** · prev: [planN_modelbench.md](planN_modelbench.md) · **next ▶ TBD**
>
> **Phase:** P2 · **Status:** `TODO` · **Depends on:** N (winning base model), P1 (existing taxonomy facts as baseline) · **Auto-advance:** yes

**Owner:** worker (Claude Sonnet 4.6) · **Repo:** `/Users/brunopinto/Repos/Loci` · **Branch:** `p0-fact-fts`
Run everything with `uv run`. DB override: `LOCI_PATHS__KNOWLEDGE_DB=/Volumes/SSD1TB001/loci/knowledge/scarlet_v2.db`.

> ⚠️ **G-DATA-MUTATION gate:** before re-extracting facts into scarlet_v2.db, make a timestamped backup:
> `cp /Volumes/SSD1TB001/loci/knowledge/scarlet_v2.db /Volumes/SSD1TB001/loci/knowledge/scarlet_v2_pre_p2_$(date +%s).db`

---

## 1. Why this exists

P1 minted 63 taxonomy facts via one LLM pass per chapter. Quality is good (`source='llm'`, confidence=0.7)
but coverage is thin — 63 facts across a full novel produces a 26:1 SVO:minted ratio, so minted facts
rarely win FTS slots. The remaining zero-score questions all have a missing-fact root cause:

| Question | Missing fact | Root cause |
|---|---|---|
| "Who is the landlady at 221B?" | (Mrs Hudson, role, landlady) | Coref: "our landlady" in ch1, "Mrs Hudson" in ch4 |
| "What occupation was Jefferson Hope?" | (Jefferson Hope, occupation, cab driver) | Archaic vocab: "jarvey" not decoded |
| "Who introduced Watson to Holmes?" | (Stamford, introduce, Watson+Holmes) | FTS stem gap: "introduced" ≠ "introduce" |
| "What does RACHE mean?" | (RACHE, means, revenge) | German word not decoded — P1 minted but coref failed |
| Multi-hop chains | various | Requires connecting 2+ facts not co-located in one chunk |

**Three extraction passes** fix these systematically without overfitting to the 100-Q set:

---

## 2. Pass design

### Pass 1 — Entity-centric extraction (cross-chunk coref)

**What:** For each named entity in `entities` table, gather all sentences mentioning it across the whole
book (via `entity_mentions` + `chunks`), concatenate as a single context window, ask the LLM:

```
"Given these passages about [ENTITY], extract all facts about them with these predicates:
profession/occupation/role, full_name/alias, relationship_to, introduced_by, located_at,
resides_at, affiliation, motive, cause_of_death, means. Use canonical names for objects.
Only extract what the text directly supports."
```

**Why better than P1:** P1 ran per-chunk — "our landlady" in one chunk has no name. Entity-centric pass
sees all mentions together, resolves "our landlady" → "Mrs Hudson" naturally.

**Implementation:** new function in `enhance.py`, e.g. `extract_entity_facts(conn, llm, entity_id)`.
Iterate over entities where `entities.proper_noun = 1`. Insert via `insert_fact(source='llm')`.
Guard with `db_meta` key `p2_entity_v` to allow re-run safety.

### Pass 2 — Implication extraction (archaic/indirect facts)

**What:** For each chunk, ask the LLM:

```
"What facts are implied but not stated directly? Focus on: occupations described by job action
(e.g. 'drove a cab' → occupation=cab driver), translations of foreign words, decoded archaic
terms, aliases used for characters. Output only high-confidence implications."
```

**Why:** Jefferson Hope's "jarvey" is described by action ("drove"), not named. RACHE is German.
These are systematically missed by vocab-matching extraction.

**Implementation:** new pass in `enhance.py`, `extract_implied_facts(conn, llm, chunk_id)`.
Source = `'llm'`, lower confidence threshold acceptable (0.65).

### Pass 3 — Question-pattern extraction

**What:** Run a targeted pass using the known question taxonomy as a prompt template. For each entity,
ask one focused prompt per predicate category:

```
"For [ENTITY] in this text, answer if supported:
- What is their profession or occupation?
- What is their full name?
- What is their relationship to other characters?
- What is their motive?
- What alias or nickname do they use?"
```

**Why:** Directly mints facts whose predicate vocabulary matches question vocabulary — bridges the
lexical gap that FTS can't cross (M-baseline leaves fact at 44.4, all phases failed to improve it).

---

## 3. Anti-overfitting guard

**Hard rule:** do not inspect `qna_scarlet.json` question text when designing extraction prompts.
The three passes above are grounded in *linguistic analysis of the source text gaps*, not in the
question set. Validate on held-out `qna_20.json` first; `qna_scarlet.json` is the deciding number.

---

## 4. Implementation plan

1. **Backup DB** (G-DATA-MUTATION gate — always do this first).
2. Add `p2_entity_v`, `p2_implied_v`, `p2_qpattern_v` keys to `db_meta` for idempotency.
3. Implement Pass 1 in `enhance.py`: `extract_entity_facts(conn, llm, cfg)`.
4. Implement Pass 2 in `enhance.py`: `extract_implied_facts(conn, llm, cfg)`.
5. Implement Pass 3 in `enhance.py`: `extract_qpattern_facts(conn, llm, cfg)`.
6. Wire all three to `loci enhance` CLI command (flag: `--passes entity,implied,qpattern`).
7. Run on `scarlet_v2.db`. Log new fact counts per pass and per entity.
8. `rebuild_fact_fts(conn)` after all passes.
9. Bench: `max_facts=4, fact_sources=minted` with N's winning model.

---

## 5. Success criteria

Compare vs N's winning-model baseline (max_facts=0):

- ✅ **PASS**: overall ≥ winner_baseline + 3 AND fact ≥ winner_fact + 5 AND halluc ≤ 1
- 🔄 **PARTIAL**: fact improves but overall within noise band — run per-Q analysis and decide
- 🛑 **FAIL**: no improvement despite more facts → bottleneck is retrieval mechanism, not data

---

## 6. Tests

Add to `tests/test_enhance.py`:
- `test_entity_facts_uses_cross_chunk_coref`: mock LLM, verify entity mention aggregation
- `test_implied_facts_source_is_llm`: verify new facts have `source='llm'`
- `test_p2_idempotent`: running twice doesn't duplicate facts (db_meta guard)

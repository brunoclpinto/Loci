#!/usr/bin/env bash
# Ingest A Study in Scarlet into three separate DBs (one per embedder),
# then bench each and print a comparison.
#
# Usage:
#   bash bench/embedder_sweep.sh [--skip-ingest]
#
# --skip-ingest: skip ingest+enhance and go straight to bench (if DBs exist)

set -euo pipefail

SCARLET="/Volumes/SSD1TB001/loci/raw/pg244_a_study_in_scarlet.txt"
DB_DIR="/Volumes/SSD1TB001/loci/knowledge"
LOG_DIR="/Volumes/SSD1TB001/loci/logs/bench"
QNA="bench/qna.json"

SKIP_INGEST=false
for arg in "$@"; do [[ "$arg" == "--skip-ingest" ]] && SKIP_INGEST=true; done

run_variant() {
  local VARIANT="$1"
  local EMBEDDER="$2"
  local DIM="$3"
  local DB="${DB_DIR}/scarlet_${VARIANT}.db"

  echo ""
  echo "════════════════════════════════════════════════════"
  echo "  Variant: bge-${VARIANT}  (${DIM}-dim)  →  ${DB}"
  echo "════════════════════════════════════════════════════"

  if [[ "$SKIP_INGEST" == "false" ]]; then
    echo "→ Ingest..."
    LOCI_MODELS__EMBEDDER="$EMBEDDER" \
    LOCI_MODELS__VEC_DIM="$DIM" \
    LOCI_PATHS__KNOWLEDGE_DB="$DB" \
      uv run loci ingest "$SCARLET"

    echo "→ Enhance..."
    LOCI_MODELS__EMBEDDER="$EMBEDDER" \
    LOCI_MODELS__VEC_DIM="$DIM" \
    LOCI_PATHS__KNOWLEDGE_DB="$DB" \
      uv run loci enhance
  fi

  echo "→ Bench..."
  LOCI_MODELS__EMBEDDER="$EMBEDDER" \
  LOCI_MODELS__VEC_DIM="$DIM" \
  LOCI_PATHS__KNOWLEDGE_DB="$DB" \
    uv run loci bench query --qna "$QNA" --runs 1 --label "emb_${VARIANT}"
}

run_variant "small" "bge-small-en-v1.5-q8_0.gguf" "384"
run_variant "base"  "bge-base-en-v1.5-q8_0.gguf"  "768"
run_variant "large" "bge-large-en-v1.5-q8_0.gguf" "1024"

echo ""
echo "════════════════════════════════════════════════════"
echo "  Pairwise comparisons"
echo "════════════════════════════════════════════════════"

LOG_SMALL=$(ls -t "${LOG_DIR}"/*emb_small*.jsonl 2>/dev/null | grep -v judge | head -1)
LOG_BASE=$(ls  -t "${LOG_DIR}"/*emb_base*.jsonl  2>/dev/null | grep -v judge | head -1)
LOG_LARGE=$(ls -t "${LOG_DIR}"/*emb_large*.jsonl 2>/dev/null | grep -v judge | head -1)

if [[ -f "$LOG_SMALL" && -f "$LOG_BASE" ]]; then
  echo ""
  echo "── small vs base ──"
  uv run loci bench compare "$LOG_SMALL" "$LOG_BASE"
fi
if [[ -f "$LOG_SMALL" && -f "$LOG_LARGE" ]]; then
  echo ""
  echo "── small vs large ──"
  uv run loci bench compare "$LOG_SMALL" "$LOG_LARGE"
fi
if [[ -f "$LOG_BASE" && -f "$LOG_LARGE" ]]; then
  echo ""
  echo "── base vs large ──"
  uv run loci bench compare "$LOG_BASE" "$LOG_LARGE"
fi

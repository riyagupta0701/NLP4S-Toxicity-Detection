#!/bin/bash
# =============================================================================
# End-to-end pipeline: data prep -> synthetic generation -> encoder train/infer
#                      -> LLM prompting -> evaluation notebook.
# =============================================================================
# Runs every stage in dependency order, each writing its outputs to disk so a
# crash (or a deliberate stage skip) can be resumed from the next checkpoint.
#
# Stages (in order):
#   prep      nlp4s prep      -> data/processed/{mhc,train}.jsonl, coverage.json
#   generate  nlp4s generate  -> data/synthetic/implicit.jsonl   (needs COHERE_API_KEY)
#   merge     merge_hate_datasets -> data/processed/train_split/hate_{train,val}.csv
#   train     nlp4s train     -> outputs/encoder_baseline/
#   infer     nlp4s infer     -> outputs/encoder_out/predictions.jsonl
#   llm       LLM sweep       -> outputs/llm/predictions.jsonl  (+ metrics)
#   eval      execute notebook-> outputs/eval/   (via jupyter nbconvert)
#
# Usage:
#   scripts/run_pipeline.sh [options]
#
# Options:
#   --only "STAGE ..."    run only the named stages (space-separated)
#   --skip "STAGE ..."    run all stages except these
#   --from STAGE          run from STAGE to the end
#   --llm-sub N           eval rows per cell for the LLM sweep (default 25)
#   --llm-strats "..."    few-shot strategies for the LLM sweep (default: all)
#   --no-eval             alias for --skip eval (skip the notebook step)
#   -h | --help           show this help
#
# Examples:
#   scripts/run_pipeline.sh                       # full pipeline, LLM sweep @ 25 rows/cell
#   scripts/run_pipeline.sh --llm-sub 3           # fast smoke run end-to-end
#   scripts/run_pipeline.sh --from train          # resume from encoder training
#   scripts/run_pipeline.sh --skip "generate"     # skip synthetic generation
#   scripts/run_pipeline.sh --only "infer llm"    # just the two prediction stages
#
# Notes:
#   * `generate` needs COHERE_API_KEY; it is skipped with a warning if unset.
#   * `llm` delegates to scripts/run_llm_experiments.sh (memory-safe, Ollama).
#   * `eval` executes evaluation.ipynb in place; needs jupyter + bert-score.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

NLP=.venv/bin/nlp4s
PYTHON=.venv/bin/python
# Fall back to PATH if the venv binaries are absent.
[ -x "$NLP" ]    || NLP=nlp4s
[ -x "$PYTHON" ] || PYTHON=python

ALL_STAGES=(prep generate merge train infer llm eval)
LLM_SUB=25
LLM_STRATS=""
RUN_ONLY=""
SKIP=""
FROM=""

usage() { sed -n '2,41p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --only)       RUN_ONLY="$2"; shift 2;;
    --skip)       SKIP="$2"; shift 2;;
    --from)       FROM="$2"; shift 2;;
    --llm-sub)    LLM_SUB="$2"; shift 2;;
    --llm-strats) LLM_STRATS="$2"; shift 2;;
    --no-eval)    SKIP="$SKIP eval"; shift;;
    -h|--help)    usage 0;;
    *) echo "[pipeline] unknown option: $1" >&2; usage 1;;
  esac
done

# Decide which stages run, preserving canonical order.
stage_selected() {
  local s="$1"
  if [ -n "$RUN_ONLY" ]; then
    [[ " $RUN_ONLY " == *" $s "* ]] && return 0 || return 1
  fi
  if [ -n "$FROM" ]; then
    local seen=0 x
    for x in "${ALL_STAGES[@]}"; do
      [ "$x" = "$FROM" ] && seen=1
      [ "$x" = "$s" ] && { [ "$seen" = 1 ] && break || return 1; }
    done
  fi
  [[ " $SKIP " == *" $s "* ]] && return 1
  return 0
}

run_stage() { echo ""; echo "========== [pipeline] STAGE: $1 @ $(date +%T) =========="; }
ok()        { echo "[pipeline] $1 OK @ $(date +%T)"; }
die()       { echo "[pipeline] FATAL: $1" >&2; exit 1; }

echo "[pipeline] start @ $(date) ; llm_sub=$LLM_SUB"
echo "[pipeline] using NLP=$NLP PYTHON=$PYTHON"

# ---- prep -------------------------------------------------------------------
if stage_selected prep; then
  run_stage prep
  "$NLP" prep --config configs/data.yaml || die "prep failed"
  ok prep
fi

# ---- generate (optional; needs COHERE_API_KEY) ------------------------------
if stage_selected generate; then
  run_stage generate
  if [ -z "${COHERE_API_KEY:-}" ] && ! grep -q '^COHERE_API_KEY=.\+' .env 2>/dev/null; then
    echo "[pipeline] COHERE_API_KEY not set and not in .env -> skipping synthetic generation"
  else
    "$NLP" generate --config configs/data.yaml || die "generate failed"
    ok generate
  fi
fi

# ---- merge (build encoder training CSVs) ------------------------------------
if stage_selected merge; then
  run_stage merge
  "$PYTHON" -m nlp4s.data.merge_hate_datasets || die "merge_hate_datasets failed"
  ok merge
fi

# ---- train ------------------------------------------------------------------
if stage_selected train; then
  run_stage train
  "$NLP" train --config configs/encoder.yaml || die "train failed"
  ok train
fi

# ---- infer ------------------------------------------------------------------
if stage_selected infer; then
  run_stage infer
  "$NLP" infer --config configs/encoder.yaml || die "infer failed"
  ok infer
fi

# ---- llm --------------------------------------------------------------------
if stage_selected llm; then
  run_stage llm
  if [ -n "$LLM_STRATS" ]; then
    scripts/run_llm_experiments.sh "$LLM_SUB" $LLM_STRATS || die "llm sweep failed"
  else
    scripts/run_llm_experiments.sh "$LLM_SUB" || die "llm sweep failed"
  fi
  ok llm
fi

# ---- eval (execute the notebook) --------------------------------------------
if stage_selected eval; then
  run_stage eval
  [ -f outputs/encoder_out/predictions.jsonl ] || echo "[pipeline] WARN: encoder predictions missing"
  [ -f outputs/llm/predictions.jsonl ]         || echo "[pipeline] WARN: LLM predictions missing"
  mkdir -p outputs/eval
  "$PYTHON" -m jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=1800 evaluation.ipynb \
    || die "notebook execution failed (need jupyter + bert-score; see README Step 7)"
  ok eval
fi

echo ""
echo "[pipeline] DONE @ $(date)"
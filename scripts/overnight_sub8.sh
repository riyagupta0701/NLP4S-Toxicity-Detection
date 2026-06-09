#!/bin/bash
# =============================================================================
# Overnight RQ2/RQ3 LLM experiment run — eval_subsample_per_cell = 8 (recommended)
# =============================================================================
# Full matrix over LOCAL Ollama:
#   3 models  x  {random, bm25, target_group}  x  {no_explanation, explanation}
#
# Answers:
#   RQ3.1-3.3  implicit vs explicit, explanation vs not, across models & languages
#   RQ3.4      few-shot demonstration selection (random vs selective retrieval)
#   RQ2.1-2.3  explanation -> accuracy; explanation quality via BERTScore
#
# Safety/robustness:
#   * ONE model loaded at a time (forced `ollama stop` between models) -> safe on 16 GB.
#   * `caffeinate` prevents idle-sleep for the lifetime of this script.
#   * Results merged + re-scored after EVERY (model, strategy) leg, so an
#     interrupted run still leaves usable, up-to-date outputs.
#   * The SQLite response cache makes a resume cheap (only new cells cost time).
#
# Estimated: ~9.5k LLM calls, ~10 hours.
#
# Run:        bash scripts/overnight_sub8.sh
# Monitor:    watch -n5 'ollama ps; tail -3 outputs/llm/experiments.log; cat outputs/llm/metrics.txt'
# Stop:       Ctrl-C, then  ollama stop mistral:7b-instruct llama3:8b aya:8b-23
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

SUB=8
PY=.venv/bin/nlp4s          # console script (alias below for python helpers)
NLP=.venv/bin/nlp4s
PYTHON=.venv/bin/python
LOG="outputs/llm/experiments.log"
mkdir -p outputs/llm

# Few-shot selection strategies (RQ3.4) and models (small -> large).
STRATS=(random bm25 target_group)
MODELS=("mistral|mistral:7b-instruct" "llama3|llama3:8b" "aya|aya:8b-23")

# Keep the machine awake only while this script is alive.
caffeinate -i -w $$ &
CAFFEINATE_PID=$!
cleanup() {
  kill "$CAFFEINATE_PID" 2>/dev/null
  for t in mistral:7b-instruct llama3:8b aya:8b-23; do ollama stop "$t" 2>/dev/null; done
}
trap cleanup EXIT INT TERM

{
  echo "[overnight] subsample=$SUB strategies=${STRATS[*]} @ $(date)"

  # Clean slate: unload models and clear stale per-leg prediction files so the
  # incremental merge only ever combines THIS run's outputs.
  for t in mistral:7b-instruct llama3:8b aya:8b-23 qwen2.5:7b-instruct; do ollama stop "$t" 2>/dev/null; done
  rm -f outputs/llm/pred_*.jsonl

  for entry in "${MODELS[@]}"; do
    key="${entry%%|*}"; tag="${entry##*|}"
    case "$key" in
      mistral) base="mistral-7b";;
      llama3)  base="llama-3-8b";;
      aya)     base="aya-23";;
    esac

    for strat in "${STRATS[@]}"; do
      cfg="configs/_exp_${key}_${strat}.yaml"
      cat > "$cfg" <<EOF
models:
  - name: "${base}::${strat}"          # selection encoded in name for RQ3.4 scoring
    provider: "openai_compatible"
    model_id: "$tag"
prompting:
  conditions: ["no_explanation", "explanation"]
  shots: 4
  fewshot_selection: "${strat}"
  temperature: 0.0
  json_mode: true
  max_tokens: 256
  max_tokens_by_condition:
    explanation: 512
  seed: 42
  mhc_path: "data/processed/mhc.jsonl"
  synthetic_path: "data/synthetic/implicit.jsonl"
  pool_per_cell: 5
  eval_subsample_per_cell: $SUB
  cache_path: "outputs/llm/cache.sqlite"
  predictions_out: "outputs/llm/pred_${key}_${strat}.jsonl"
EOF
      echo "[overnight] ===== ${base} / ${strat} @ $(date +%T) ====="
      "$NLP" llm --config "$cfg"
      echo "[overnight] ${base}/${strat} exit=$?"
      rm -f "$cfg"

      echo "[overnight] merge + score classification @ $(date +%T)"
      cat outputs/llm/pred_*.jsonl > outputs/llm/predictions.jsonl 2>/dev/null
      wc -l outputs/llm/predictions.jsonl
      "$PYTHON" scripts/llm_metrics.py outputs/llm/predictions.jsonl outputs/llm/metrics || true
    done

    echo "[overnight] unloading $tag"
    ollama stop "$tag" 2>/dev/null
  done

  echo "[overnight] scoring explanation quality (BERTScore) @ $(date +%T)"
  "$PYTHON" scripts/explanation_quality.py outputs/llm/predictions.jsonl outputs/llm/explanation_quality || true

  echo "[overnight] DONE @ $(date)"
} 2>&1 | tee "$LOG"

#!/bin/bash
# Full RQ2/RQ3 LLM experiment matrix over LOCAL Ollama models, memory-safe.
#
# Sweeps:  models x few-shot selection {random,bm25,target_group} x {no_explanation,explanation}
# Answers: RQ3.1-3.3 (implicit vs explicit, explanation vs not, across models/langs)
#          RQ3.4 (demonstration selection: random vs selective retrieval)
#          RQ2.1-2.3 (explanation -> accuracy; explanation quality via BERTScore)
#
# Memory: ONE model loaded at a time. Each model runs all its strategies/conditions
# while resident, then is force-unloaded before the next (safe on 16 GB).
#
# Usage:
#   scripts/run_llm_experiments.sh [SUBSAMPLE] [STRATEGY ...]
#     SUBSAMPLE   eval rows per (language x functionality) cell. Default 10.
#     STRATEGY    subset of: random bm25 target_group. Default: all three.
#
# Examples:
#   scripts/run_llm_experiments.sh 10                 # full sweep, ~13 h
#   scripts/run_llm_experiments.sh 25 random          # main RQ3.1-3.3 only, ~10 h
#   scripts/run_llm_experiments.sh 3                  # quick smoke of the whole matrix
#
# Total LLM calls ~= (#models) x (#strategies) x 2 conditions x (66 cells x SUBSAMPLE).
# Results are merged + re-scored after EVERY (model,strategy) leg, so partial runs
# still leave usable outputs. The response cache makes resuming cheap.
set -uo pipefail
cd "$(dirname "$0")/.."

SUB="${1:-10}"; shift || true
STRATS=("$@"); [ ${#STRATS[@]} -eq 0 ] && STRATS=(random bm25 target_group)
PY=.venv/bin/python
NLP=.venv/bin/nlp4s
MODELS=("mistral|mistral:7b-instruct" "llama3|llama3:8b" "aya|aya:8b-23")

echo "[exp] subsample=$SUB ; strategies=${STRATS[*]} ; @ $(date)"
for t in mistral:7b-instruct llama3:8b aya:8b-23 qwen2.5:7b-instruct; do ollama stop "$t" 2>/dev/null; done
rm -f outputs/llm/pred_*.jsonl   # start clean so the incremental merge is consistent

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
    echo "[exp] ===== ${base} / ${strat} @ $(date +%T) ====="
    $NLP llm --config "$cfg"
    echo "[exp] ${base}/${strat} exit=$?"
    rm -f "$cfg"

    echo "[exp] merge + score classification @ $(date +%T)"
    cat outputs/llm/pred_*.jsonl > outputs/llm/predictions.jsonl 2>/dev/null
    wc -l outputs/llm/predictions.jsonl
    $PY scripts/llm_metrics.py outputs/llm/predictions.jsonl outputs/llm/metrics || true
  done
  echo "[exp] unloading $tag"
  ollama stop "$tag" 2>/dev/null
done

echo "[exp] scoring explanation quality (BERTScore) @ $(date +%T)"
$PY scripts/explanation_quality.py outputs/llm/predictions.jsonl outputs/llm/explanation_quality || true

echo "[exp] DONE @ $(date)"

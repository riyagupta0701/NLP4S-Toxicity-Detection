#!/bin/bash
# Memory-safe LLM run over local Ollama models, ONE MODEL AT A TIME.
#
# Why one at a time: on a 16 GB Mac, Ollama's default keep-alive stacks each
# finished 8B model (~5.5 GB) while loading the next -> 3 models at once swaps
# the machine to death. Running each model in its own process and forcing
# `ollama stop` between them caps peak RAM at a single model.
#
# Usage:
#   scripts/run_llm_ollama.sh [SUBSAMPLE_PER_CELL]
#       SUBSAMPLE_PER_CELL  eval rows kept per (language x functionality) cell.
#                           3 = quick test (~198 rows/model). 25 = full benchmark.
#                           Default: 25.
#
# Outputs:
#   outputs/llm/predictions.jsonl   merged predictions (all models)
#   outputs/llm/metrics.{csv,txt}   scored accuracy + JSON-health table
#   outputs/llm/cache.sqlite        response cache (reused across runs)
#
# Re-runs are cheap: anything already in the cache is not re-charged.
set -uo pipefail
cd "$(dirname "$0")/.."
SUB="${1:-25}"
PY=.venv/bin/python
NLP=.venv/bin/nlp4s

# model key | ollama tag, ordered small -> large
MODELS=("mistral|mistral:7b-instruct" "llama3|llama3:8b" "aya|aya:8b-23")

echo "[run] subsample_per_cell=$SUB ; one model at a time with forced unload"
for t in mistral:7b-instruct llama3:8b aya:8b-23 qwen2.5:7b-instruct; do ollama stop "$t" 2>/dev/null; done

# Clear stale per-model prediction files so the incremental merge only ever
# combines this run's outputs (not a previous run's).
rm -f outputs/llm/pred_mistral.jsonl outputs/llm/pred_llama3.jsonl outputs/llm/pred_aya.jsonl

for entry in "${MODELS[@]}"; do
  key="${entry%%|*}"; tag="${entry##*|}"
  case "$key" in
    mistral) name="mistral-7b";;
    llama3)  name="llama-3-8b";;
    aya)     name="aya-23";;
  esac
  cfg="configs/_run_${key}.yaml"
  cat > "$cfg" <<EOF
models:
  - name: "$name"
    provider: "openai_compatible"
    model_id: "$tag"
prompting:
  conditions: ["no_explanation", "explanation"]
  shots: 4
  fewshot_selection: "random"
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
  predictions_out: "outputs/llm/pred_${key}.jsonl"
EOF
  echo "[run] ===== $name ($tag) @ $(date +%T) ====="
  $NLP llm --config "$cfg"
  echo "[run] $name exit=$? ; unloading $tag"
  ollama stop "$tag" 2>/dev/null
  rm -f "$cfg"

  # Merge + score after EACH model so results persist even if a later model
  # is interrupted (the cache makes a resume cheap).
  echo "[run] merging + scoring after $name @ $(date +%T)"
  cat outputs/llm/pred_*.jsonl > outputs/llm/predictions.jsonl 2>/dev/null
  wc -l outputs/llm/predictions.jsonl
  $PY scripts/llm_metrics.py outputs/llm/predictions.jsonl outputs/llm/metrics || true
done

echo "[run] DONE @ $(date +%T)"

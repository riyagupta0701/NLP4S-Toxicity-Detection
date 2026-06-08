# NLP4S — Implicit vs Explicit Hate Speech Detection (Multilingual)

DSAIT4100 NLP for Society, Group 3. Decomposing hate-speech-detection performance into **implicit** vs **explicit** hate across the 11 Multilingual HateCheck (MHC) languages.

See [`docs/assignment.md`](docs/assignment.md) for the research design (RQs, datasets, methods) and [`docs/implementation_plan.md`](docs/implementation_plan.md) for the workplan and the four parallel roles.

> Status: **skeleton** — module signatures, configs, and a fixture are in place; domain logic is marked `TODO` for the role owners to implement.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env   # then fill in LLM API keys
```

## Repository layout

```
configs/        YAML config per workstream (data / encoder / llm / eval)
data/           raw / processed / synthetic  (gitignored; structure kept via .gitkeep)
outputs/        predictions, metrics, figures (gitignored)
src/nlp4s/      the package (see role mapping below)
tests/          fixtures + tests
docs/           proposal, research design, implementation plan
```

### Role mapping (claim one in `docs/implementation_plan.md`)

| Role | Focus | Code |
|------|-------|------|
| A | Data & synthetic generation | `src/nlp4s/data/`, `src/nlp4s/generation/` |
| B | Encoder baseline (XLM-RoBERTa) | `src/nlp4s/encoder/` |
| C | LLM prompting (Aya / Mistral / Llama) | `src/nlp4s/llm/` |
| D | Evaluation & analysis | `src/nlp4s/eval/` |

Shared contracts (agreed first): `schema.py`, `functionalities.py`, `io_utils.py`, `config.py`.

## The frozen data schema

Every data and prediction record uses one schema so the four workstreams stay decoupled and can develop against fixtures from day one:

```
text, language, label, functionality, split
```

Defined in [`src/nlp4s/schema.py`](src/nlp4s/schema.py). Do not diverge from it silently.

## CLI

```bash
nlp4s --help          # or: python -m nlp4s.cli --help
nlp4s prep --config configs/data.yaml
nlp4s generate --config configs/data.yaml
nlp4s train --config configs/encoder.yaml
nlp4s infer --config configs/encoder.yaml
nlp4s llm --config configs/llm.yaml
nlp4s eval --config configs/eval.yaml
```

`prep` and `generate` are implemented (Role A); the rest still call into `TODO` functions owned by their roles.

## Role A — data & synthetic generation

### Required raw layout

MHC is downloaded automatically from HuggingFace. HASOC is **not redistributable**, so you obtain the per-edition files from the shared-task organisers and drop them under `data/raw/hasoc/<lang>/`:

```
data/raw/
  mhc/                       # HuggingFace cache (created on first run)
  hasoc/
    en/   hasoc_2019_en_train.tsv, hasoc_2020_en_train.tsv, ...
    de/   hasoc_2019_de_train.tsv, ...
    hi/   hasoc_2019_hi_train.tsv, ...
```

The loader auto-detects `.tsv` / `.csv` / `.txt`, sniffs delimiters, and accepts column-name variants (`text`/`tweet`/`comment`/`post`, `task_1`/`task1`/`label`/...). Per-language directories that are missing produce a warning and are skipped — a partial download still yields a usable training corpus.

### `nlp4s prep` — assemble training corpus + coverage report

```bash
nlp4s prep --config configs/data.yaml
```

Produces:

| Path | Contents |
|------|----------|
| `data/processed/mhc.jsonl` | MHC eval set in the frozen schema, filtered to the 6 studied functionalities (`derog_neg_attrib_h, derog_dehum_h, derog_impl_h, slur_h, profanity_h, profanity_nh`). |
| `data/processed/train.jsonl` | HASOC training corpus for the configured languages (deduped on normalised text, validated against the schema). |
| `outputs/data/coverage.json` | Per-language example counts across all 11 MHC languages, plus the `missing_languages` list — the Phase-2 decision input. |

The MHC step is best-effort: if HuggingFace is unreachable, a warning is emitted and corpus assembly continues so prep stays useful offline.

### `nlp4s generate` — synthetic implicit-multilingual examples

```bash
nlp4s generate --config configs/data.yaml
```

Generates matched (implicit-hateful, non-hateful-control) pairs in the target languages using a multilingual LLM (Aya-23 via `CohereClient` once Role C ships it). Target-language resolution is layered:

1. `generation.target_languages` in the config (explicit, set by you after the baseline lands).
2. `missing_languages` from `outputs/data/coverage.json` (zero-coverage MHC languages).
3. Static fallback `languages_without_training_data()` with a warning.

Each generated example uses the frozen schema with `split="synthetic"`, `functionality="derog_impl_h"` for hateful and `functionality="profanity_nh"` for the matched control. Output goes to `data/synthetic/implicit.jsonl`, after dedup + length + label/functionality-consistency + language-ID filtering.

### Cross-role dependency

Role A waits on **Role B's baseline matrix** before committing on `generation.target_languages`. Until then, the coverage report drives target selection automatically — start with the 8 MHC languages that have no HASOC training data (`ar, nl, fr, it, zh, pl, pt, es`) and revise once the baseline reveals which functionalities × languages most need help.

## Role C — multi-LLM prompting (RQ3)

```bash
nlp4s llm --config configs/llm.yaml
```

Prompts API-hosted LLMs over MHC (no fine-tuning) and writes `Prediction` records to `outputs/llm/predictions.jsonl`. Three backends are wired up via `configs/llm.yaml`:

- **Aya-23** — Cohere API (`provider: cohere`, uses `COHERE_API_KEY`).
- **Mistral-7B** and **Llama-3-8B** — any OpenAI-compatible endpoint (`provider: openai_compatible`, uses `OPENAI_COMPATIBLE_API_KEY` / `OPENAI_COMPATIBLE_BASE_URL`; point it at a local vLLM / Ollama / LM Studio server).

Pipeline:

1. Read `data/processed/mhc.jsonl` (Role A's `prep` output).
2. Split deterministically into **eval** and a few-shot **pool**, stratified per (language × functionality) cell (`pool_per_cell`); then cap eval per cell (`eval_subsample_per_cell`) for budget. Pool rows are never scored, so there is no demo↔eval leakage. Role A's synthetic implicit data is folded into the pool when present, extending coverage to the 8 MHC languages without HASOC.
3. For each `model × condition` (`no_explanation`, `explanation`) build prompts — with `shots` few-shot demonstrations selected by `fewshot_selection` (`random` / `bm25` / `target_group`) — call the model, parse the response, and emit predictions.

Key knobs in `configs/llm.yaml`: `shots`, `fewshot_selection`, `conditions`, `max_tokens` (plus `max_tokens_by_condition` to give the explanation condition a larger budget), and `seed`. Every call is cached in `outputs/llm/cache.sqlite` keyed by `(model_id, prompt, temperature, max_tokens)` — delete the file to force re-inference.

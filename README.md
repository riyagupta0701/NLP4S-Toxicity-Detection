# Implicit vs Explicit Hate Speech Detection in a Multilingual Setting

Full pipeline for studying how well encoder-based and LLM-based hate-speech classifiers handle **implicit** vs **explicit** hate across 7 languages.

**Research questions:**
- **RQ1** — Does XLM-RoBERTa perform worse on implicit hate speech than explicit, and how does the gap vary across languages?
- **RQ2** — Does prompting an LLM to produce a rationale (explanation condition) improve classification accuracy for implicit hate, and how does explanation quality correlate with correctness?
- **RQ3** — How do three LLMs (Aya-23, Mistral-7B, Llama-3-8B) and three few-shot selection strategies (random, BM25, target-group) interact with the implicit/explicit distinction across languages?


## Repository layout

```
configs/          YAML config for each pipeline step
data/             raw / processed / synthetic
outputs/          predictions, metrics, figures
src/nlp4s/        Python package
  data/           MHC + HASOC loading and corpus assembly
  generation/     ToxiGen-style synthetic implicit-hate generation
  encoder/        XLM-RoBERTa fine-tuning and inference
  llm/            Multi-LLM prompting (few-shot, caching, parsing)
scripts/          Experiment sweep scripts and quick-scoring utilities
tests/            Unit tests and fixtures
docs/             Project proposal and research design
evaluation.ipynb  Full evaluation notebook (RQ1–RQ3)
```


## Reproducing the full pipeline

Run the steps below in order. Each step writes its outputs to disk so you can resume from any checkpoint.

### One-command run

After the environment setup (Step 1), the whole pipeline — prep → synthetic generation → encoder train/infer → LLM prompting → evaluation notebook — can be run end-to-end with a single orchestrator:

```bash
scripts/run_pipeline.sh                 # full pipeline, LLM sweep @ 25 rows/cell
scripts/run_pipeline.sh --llm-sub 3     # fast end-to-end smoke test
scripts/run_pipeline.sh --from train    # resume from encoder training
scripts/run_pipeline.sh --only "infer llm"   # just the two prediction stages
scripts/run_pipeline.sh --no-eval       # everything but the notebook
scripts/run_pipeline.sh --help          # all options
```

Stages run in dependency order, each writing its outputs to disk, so `--from`, `--only`, and `--skip` let you resume from any checkpoint. `generate` is skipped automatically when `COHERE_API_KEY` is unset; `llm` delegates to the memory-safe `scripts/run_llm_experiments.sh`; `eval` executes `evaluation.ipynb` in place (needs `jupyter` + `bert-score`). The steps below document each stage individually.

### Step 1 — Environment setup

**Requirements:** Python ≥ 3.10, ~6 GB disk for model weights, and a compute device for encoder training:

- **Apple Silicon (M-series):** training auto-uses the MPS backend.
- **NVIDIA GPU (Linux/Windows):** CUDA is used automatically if a CUDA build of PyTorch is installed.
- **CPU only:** works but encoder training is slow (hours rather than ~20 min).

> **Platform note:** the sweep scripts under `scripts/` (`run_pipeline.sh`, `run_llm_experiments.sh`, …) are **bash** scripts. On macOS/Linux they run as shown. On Windows, run them under WSL or Git Bash, or invoke the underlying `nlp4s` CLI commands directly.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install all dependencies (covers everything: training, LLM clients, the notebook)
pip install -r requirements.txt
pip install -e .

# Configure API keys
cp .env.example .env
```

> Install the **full** `requirements.txt` before running any stage. A partially-installed environment surfaces as mid-run `ImportError`/`ModuleNotFoundError` for declared packages (e.g. `accelerate`, `openpyxl`, `jupyter`) — see [Troubleshooting](#troubleshooting).

Edit `.env` and fill in the following keys as needed:

| Key | Used for |
|-----|---------|
| `COHERE_API_KEY` | Aya-23 synthetic generation and LLM prompting via Cohere API |
| `OPENAI_COMPATIBLE_BASE_URL` | Local LLM server endpoint (e.g. `http://localhost:11434/v1` for Ollama) |
| `OPENAI_COMPATIBLE_API_KEY` | Any non-empty string for local servers; real key for hosted endpoints |
| `LLM_JUDGE_MODEL` | Model name for the LLM-as-judge cell in the notebook (e.g. `llama3`, `command-r-08-2024`) |

### Step 2 — Prepare data

**MHC** is downloaded automatically from HuggingFace on first run.

**HASOC** (2019/2020, English/German/Hindi) is not redistributable. Obtain the files from the [shared-task organisers](https://hasocfire.github.io/) and place them under:

```
data/raw/hasoc/
  en/   hasoc_2019_en_train.tsv  hasoc_2020_en_train.tsv  ...
  de/   hasoc_2019_de_train.tsv  ...
  hi/   hasoc_2019_hi_train.tsv  ...
```

The loader auto-detects `.tsv`/`.csv`/`.txt` files and common column-name variants. Missing language directories produce a warning and are skipped gracefully.

```bash
nlp4s prep --config configs/data.yaml
```

Outputs:

| Path | Contents |
|------|----------|
| `data/processed/mhc.jsonl` | MHC evaluation set (6 studied functionalities) |
| `data/processed/train.jsonl` | HASOC training corpus (en/de/hi, deduped) |
| `outputs/data/coverage.json` | Per-language example counts; identifies synthetic-generation targets |

### Step 3 — Generate synthetic data (optional)

Produces ToxiGen-style matched (implicit-hateful, non-hateful) pairs for MHC languages not covered by HASOC, using Aya-23 via the Cohere API. Requires `COHERE_API_KEY`.

```bash
nlp4s generate --config configs/data.yaml
```

Output: `data/synthetic/implicit.jsonl` (~5,300 examples across 10 languages).

Set `generation.target_languages` in `configs/data.yaml` to specific languages, or leave it empty to auto-target zero-coverage languages from the coverage report.

### Step 4 — Train the encoder baseline

Fine-tunes `xlm-roberta-base` on the HASOC corpus for binary hate speech classification. Training takes ~20 min on an M-series Mac (MPS) or a comparable GPU.

```bash
# Build the training/validation split first (one-time; writes data/processed/train_split/)
python -m nlp4s.data.merge_hate_datasets

# Then train
nlp4s train --config configs/encoder.yaml
```

The merge step **downloads over the network** — Multi3Hate and MLMA from HuggingFace, and HASOC-2020 from a public GitHub mirror — and reads the HASOC `.xlsx` files via `openpyxl` and detects MLMA languages via `langdetect` (both in `requirements.txt`). It is independent of the `data/raw/hasoc/` files used by Step 2.

The fine-tuned model is saved to `outputs/encoder_baseline/`.

### Step 5 — Run encoder inference

Runs the fine-tuned encoder over the 7 MHC languages that overlap with the training data (ar, de, es, fr, hi, zh, en) and writes binary predictions.

```bash
nlp4s infer --config configs/encoder.yaml
```

Output: `outputs/encoder_out/predictions.jsonl` (~5,528 predictions).

### Step 6 — Run LLM prompting experiments

#### Option A — Local Ollama (recommended for a full sweep)

**Set up Ollama first:**

1. Install it from [ollama.com/download](https://ollama.com/download) (macOS/Linux/Windows).
2. Make sure the Ollama server is running — the desktop app starts it automatically, or run `ollama serve` in a separate terminal. It listens on `http://localhost:11434`.
3. Point the LLM stage at it in `.env`:
   ```
   OPENAI_COMPATIBLE_BASE_URL=http://localhost:11434/v1
   OPENAI_COMPATIBLE_API_KEY=ollama          # any non-empty string works locally
   ```
4. Pull the three models:
   ```bash
   ollama pull mistral:7b-instruct
   ollama pull llama3:8b
   ollama pull aya:8b-23
   ```

> **16 GB RAM:** each 8B model is ~5.5 GB resident, so the scripts deliberately load **one model at a time** and force `ollama stop` between them. Running all three concurrently will swap the machine to a halt.

Run the full RQ2/RQ3 matrix (3 models × 3 selection strategies × 2 conditions), loading one model at a time to stay within 16 GB RAM:

```bash
scripts/run_llm_experiments.sh           # full benchmark (~10 h, 25 rows/cell)
scripts/run_llm_experiments.sh 3         # quick smoke test (~30 min, 3 rows/cell)
scripts/run_llm_experiments.sh 10 random # one strategy only
```

For an unattended overnight run:

```bash
bash scripts/overnight_sub8.sh          # 8 rows/cell, logs to outputs/llm/experiments.log
```

#### Option B — Single run via CLI

```bash
nlp4s llm --config configs/llm.yaml
```

Edit `configs/llm.yaml` to select models, conditions, shot count, and selection strategy.

Output: `outputs/llm/predictions.jsonl` (merged across models and strategies).

#### Quick scoring (without the notebook)

```bash
python -m scripts.llm_metrics outputs/llm/predictions.jsonl outputs/llm/metrics
python scripts/explanation_quality.py outputs/llm/predictions.jsonl outputs/llm/expl_quality
```

### Step 7 — Evaluate (notebook)

Open `evaluation.ipynb` from the project root and run all cells. The notebook addresses all three research questions and writes results to `outputs/eval/`.

```bash
jupyter notebook evaluation.ipynb
# or: code evaluation.ipynb
```

`jupyter`, `nbconvert`, and `bert-score` (used for the BERTScore cells in Section 4.1) are all installed by `requirements.txt` — no extra install needed.

To run the notebook non-interactively (this is what the pipeline's `eval` stage does):

```bash
jupyter nbconvert --to notebook --execute --inplace evaluation.ipynb
```

**Prerequisite checks:**

```bash
ls outputs/encoder_out/predictions.jsonl   # must exist (Step 5)
ls outputs/llm/predictions.jsonl           # must exist (Step 6)
```


## Datasets

The encoder training corpus is built by merging three hate-speech datasets with `python src/nlp4s/data/merge_hate_datasets.py`. MHC is the evaluation-only benchmark and is downloaded automatically.

| Dataset | Purpose | Size | Languages |
|---------|---------|------|-----------|
| **MHC** (Multilingual HateCheck) | Evaluation benchmark | 8,806 examples | ar, nl, fr, de, hi, it, zh, pl, pt, es, en |
| **Multi3Hate** | Encoder training | ~6,000 examples | en, de, es, zh, hi |
| **MLMA** (Multilingual and Multilabel Hate Speech) | Encoder training | ~6,000 examples | en, ar, fr |
| **HASOC 2019/2020** | Encoder training | 25,162 train + 3,808 val | en, de, hi |
| **Synthetic** (Aya-23 generated) | Few-shot pool augmentation | ~5,308 examples | 10 MHC languages excl. zh |

**MHC functionality breakdown:**

| Functionality | Group | Count |
|--------------|-------|-------|
| `derog_neg_attrib_h` | explicit | 1,540 |
| `derog_dehum_h` | explicit | 1,540 |
| `slur_h` | explicit | 1,545 |
| `profanity_h` | explicit | 1,533 |
| `derog_impl_h` | **implicit** | 1,548 |
| `profanity_nh` | control | 1,100 |


## Key findings

| Finding | Value |
|---------|-------|
| Encoder overall F1 | 0.775 |
| Encoder explicit F1 | 0.863 |
| Encoder implicit F1 | 0.542 |
| Implicit–explicit gap (chi-square) | p = 1.7 × 10⁻¹²⁰ *** |
| Best LLM implicit F1 (Aya-23, random, explanation) | 0.977 |
| Explanation effect significance (McNemar) | n.s. for most conditions |
| Model differences (Friedman, explicit, no_explanation) | p = 1.2 × 10⁻⁴ *** |


## CLI reference

```bash
nlp4s --help
nlp4s prep     --config configs/data.yaml      # load MHC + HASOC, build corpus
nlp4s generate --config configs/data.yaml      # generate synthetic implicit examples
nlp4s train    --config configs/encoder.yaml   # fine-tune XLM-RoBERTa
nlp4s infer    --config configs/encoder.yaml   # encoder inference over MHC
nlp4s llm      --config configs/llm.yaml       # multi-LLM prompting
nlp4s eval     --config configs/eval.yaml      # prints a pointer to evaluation.ipynb
```

All commands load `.env` automatically via `python-dotenv` if present.


## Running tests

```bash
pytest tests/
```


## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError` / `ImportError` for `accelerate`, `openpyxl`, `jupyter`, etc. mid-run | venv was created before `requirements.txt` was fully installed (or before a dependency was added) | Re-sync the environment: `pip install -r requirements.txt` |
| `ImportError: ... requires accelerate>=1.1.0` at the `train` stage | `accelerate` missing or too old | `pip install -r requirements.txt` (it pins `accelerate>=1.1.0`) |
| `llm` stage hangs or errors connecting to the model | Ollama server not running, or `OPENAI_COMPATIBLE_BASE_URL` not set | Start Ollama (`ollama serve`) and set the `.env` keys — see [Step 6, Option A](#option-a--local-ollama-recommended-for-a-full-sweep) |
| Machine swaps / freezes during the LLM sweep | Multiple 8B models resident at once on ≤16 GB RAM | Use the provided scripts (they load one model at a time); don't run models concurrently |
| `eval` stage / notebook fails with `No module named jupyter` | `jupyter`/`nbconvert` not installed | `pip install -r requirements.txt` |
| `scripts/*.sh` won't run on Windows | scripts are bash | Run under WSL or Git Bash, or call the `nlp4s` CLI commands directly |
| `Warning: ... set a HF_TOKEN` during downloads | unauthenticated HuggingFace access | Harmless (rate limits only); optionally `export HF_TOKEN=...` for faster/higher-limit downloads |

The pipeline writes every stage's output to disk, so after fixing an environment issue you can resume rather than restart — e.g. `scripts/run_pipeline.sh --from train`.

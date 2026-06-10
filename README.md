# Implicit vs Explicit Hate Speech Detection in a Multilingual Setting

Full pipeline for studying how well encoder-based and LLM-based hate-speech classifiers handle **implicit** vs **explicit** hate across 7 languages, using [Multilingual HateCheck (MHC)](https://huggingface.co/datasets/mteb/multi-hatecheck) as the evaluation benchmark.

**Research questions:**
- **RQ1** — Does XLM-RoBERTa perform worse on implicit hate speech than explicit, and how does the gap vary across languages?
- **RQ2** — Does prompting an LLM to produce a rationale (explanation condition) improve classification accuracy for implicit hate, and how does explanation quality correlate with correctness?
- **RQ3** — How do three LLMs (Aya-23, Mistral-7B, Llama-3-8B) and three few-shot selection strategies (random, BM25, target-group) interact with the implicit/explicit distinction across languages?

---

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

---

## Reproducing the full pipeline

Run the steps below in order. Each step writes its outputs to disk so you can resume from any checkpoint.

### Step 1 — Environment setup

**Requirements:** Python ≥ 3.10, ~6 GB disk for model weights, a GPU or Apple Silicon MPS device for encoder training.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
pip install -e .

# Configure API keys
cp .env.example .env
```

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
python src/nlp4s/data/merge_hate_datasets.py

# Then train
nlp4s train --config configs/encoder.yaml
```

The fine-tuned model is saved to `outputs/encoder_baseline/`.

### Step 5 — Run encoder inference

Runs the fine-tuned encoder over the 7 MHC languages that overlap with the training data (ar, de, es, fr, hi, zh, en) and writes binary predictions.

```bash
nlp4s infer --config configs/encoder.yaml
```

Output: `outputs/encoder_out/predictions.jsonl` (~5,528 predictions).

### Step 6 — Run LLM prompting experiments

#### Option A — Local Ollama (recommended for a full sweep)

Pull the three models first:

```bash
ollama pull mistral:7b-instruct
ollama pull llama3:8b
ollama pull aya:8b-23
```

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

**Prerequisite checks:**

```bash
ls outputs/encoder_out/predictions.jsonl   # must exist (Step 5)
ls outputs/llm/predictions.jsonl           # must exist (Step 6)
pip install bert-score                     # required for Section 4.1 (BERTScore)
```

---

## Datasets

| Dataset | Purpose | Size | Languages |
|---------|---------|------|-----------|
| **MHC** (Multilingual HateCheck) | Evaluation benchmark | 8,806 examples | ar, nl, fr, de, hi, it, zh, pl, pt, es, en |
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

---

## Key findings

| Finding | Value |
|---------|-------|
| Encoder overall F1 | 0.775 |
| Encoder explicit F1 | 0.864 |
| Encoder implicit F1 | 0.542 |
| Implicit–explicit gap (chi-square) | p = 1.7 × 10⁻¹²⁰ *** |
| Best LLM implicit F1 (Aya-23, random, explanation) | 0.977 |
| Explanation effect significance (McNemar) | n.s. for most conditions |
| Model differences (Friedman, explicit, no_explanation) | p = 1.2 × 10⁻⁴ *** |

---

## CLI reference

```bash
nlp4s --help
nlp4s prep     --config configs/data.yaml      # load MHC + HASOC, build corpus
nlp4s generate --config configs/data.yaml      # generate synthetic implicit examples
nlp4s train    --config configs/encoder.yaml   # fine-tune XLM-RoBERTa
nlp4s infer    --config configs/encoder.yaml   # encoder inference over MHC
nlp4s llm      --config configs/llm.yaml       # multi-LLM prompting
nlp4s eval     --config configs/eval.yaml      # evaluate predictions (stub)
```

All commands load `.env` automatically via `python-dotenv` if present.

---

## LLM-as-Judge (notebook Section 4.2)

A judge LLM scores each explanation on a 3-point rubric: **3** = correct label + specific linguistic evidence; **2** = correct but vague; **1** = wrong or irrelevant. Point-biserial correlation with correctness is reported per model (50 examples × 3 models = 150 total).

Supported backends (set via `.env`):

| Backend | `OPENAI_COMPATIBLE_BASE_URL` | `OPENAI_COMPATIBLE_API_KEY` | `LLM_JUDGE_MODEL` |
|---------|------------------------------|-----------------------------|--------------------|
| Local Ollama | `http://localhost:11434/v1` | `ollama` | `llama3` or `mistral` |
| Cohere API | `https://api.cohere.ai/compatibility/v1` | your `COHERE_API_KEY` | `command-r-08-2024` |

The Cohere trial key is rate-limited to 20 calls/min; the notebook inserts a 3 s delay automatically (~7.5 min total).

---

## Running tests

```bash
pytest tests/
```

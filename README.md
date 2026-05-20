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

(Subcommands currently parse args and call into `TODO` functions.)

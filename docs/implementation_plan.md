# Implementation Plan — Implicit vs Explicit Hate Speech Detection (Multilingual)

Companion to [assignment.md](assignment.md) (research design, RQs, datasets, references). This file holds the execution plan and the parallelisable role split.

## Team roster (assign roles below)

Roles are intentionally **unassigned** — each member selects one of the four workstreams.

- [ ] Member 1: _______________
- [ ] Member 2: _______________
- [ ] Member 3: _______________
- [ ] Member 4: _______________

## Workplan

The plan is organised so the four workstreams run **in parallel** after a short shared setup. A frozen **data interface** (a single schema: `text, language, label, functionality, split`) is agreed in week 1 so each member can develop and test against fixtures before real data flows between components.

**Phase 0 — Shared setup (everyone, ~few days).** Repository, environment, the data-format contract above, and a shared evaluation spec (metrics, per-functionality × per-language reporting). Literature review is split and runs throughout.

**1. Datasets and baseline.** Download/preprocess MHC; verify the explicit/implicit/control functionality grouping; assemble the HASOC training corpus and document the **language-coverage gap** (which MHC languages have no training data). In parallel, stand up the XLM-RoBERTa fine-tuning + inference pipeline and produce a **first baseline** on MHC to confirm the pipeline end-to-end. *Deliverable: baseline functionality × language results matrix.*

**2. Based on the baseline, plan for the rest (decision point).** Read the baseline matrix to decide scope, rather than committing blindly up front:
   - *Which languages/functionalities are weakest?* → prioritise them for synthetic augmentation and for the cross-model LLM comparison.
   - *Is the implicit gap driven by data scarcity or by the encoder itself?* → decide how much synthetic implicit data to generate and in which languages (Aya-23 covers all 11).
   - *Is cross-lingual transfer already adequate for some languages?* → skip generation there to save budget.
   This converts the synthetic-generation and few-shot work from "do everything" into a targeted, baseline-justified plan.

**3. Parallel experiment tracks (run concurrently).**
   - *Encoder track:* augmented re-training of XLM-RoBERTa with synthetic data; full RQ1 matrix.
   - *LLM-prompting track:* multi-LLM zero/few-shot, explanation vs. no-explanation (RQ3), including the **few-shot sample-selection** comparison.
   - *Generation track:* produce, filter, and hand off synthetic implicit-multilingual data to the other two tracks.
   - *Evaluation track:* run the shared harness over all model outputs; explanation-quality scoring for RQ2.

**4. Analysis and report.** Cross-model and cross-language analysis, qualitative error analysis on implicit-hate failures, visualizations, report, and a reproducible replication package.

## Role division (parallelisable, 4 members)

Each role owns one vertical workstream with a clear interface to the others, so work proceeds in parallel after Phase 0. Hard dependencies are noted explicitly by **role**; everything else is independent.

### Role A — Data & synthetic generation lead
**Assignee:** _(select)_
- Own MHC preprocessing and the explicit/implicit/control functionality grouping; assemble the HASOC training corpus; document and quantify the MHC-vs-HASOC language-coverage gap.
- Build the **synthetic implicit-multilingual generation** pipeline (ToxiGen-style demonstration prompting with Aya-23/Aya-101, optionally Llama-3) targeting the languages identified at the Phase-2 decision point; quality-filter and version the generated data.
- *Interface out:* training corpus + synthetic data in the frozen schema.
- *Depends on:* baseline matrix (Role B) to prioritise target languages.

### Role B — Encoder baseline lead
**Assignee:** _(select)_
- Implement, fine-tune, and optimise the XLM-RoBERTa pipeline (tokenization, training, inference, reproducibility, hyperparameters).
- Deliver the **first baseline** (drives the Phase-2 decision) and the augmented re-training once synthetic data lands.
- *Interface out:* per-functionality × per-language predictions.
- *Depends on:* corpus from Role A; can start immediately on HASOC-only data using fixtures.

### Role C — LLM prompting lead
**Assignee:** _(select)_
- Stand up inference for **multiple LLMs** (Mistral-7B, Llama-3-8B, Aya-23), no fine-tuning.
- Design and refine **no-explanation vs. explanation** prompts and the **few-shot demonstration-selection** strategies (random vs. BM25/target-group selective retrieval) for RQ3 (and the explanation pipeline for RQ2).
- *Interface out:* model predictions + generated rationales in the frozen schema.
- *Depends on:* few-shot pool can use Role A's synthetic data but can bootstrap from MHC/HASOC meanwhile.

### Role D — Evaluation & analysis lead
**Assignee:** _(select)_
- Own the shared evaluation harness: F1/precision/recall per functionality and language for both encoder and LLM outputs.
- Evaluate generated explanations with reference-free metrics (BERTScore / G-Eval) and analyse correlation with classification correctness (RQ2.3).
- Cross-model / cross-language analysis, error analysis, and visualizations.
- *Interface in:* any predictions in the frozen schema, so evaluation can be built and tested against fixtures from day one.

### Shared
Literature review, discussion of findings, presentation preparation, and revision of the final report.

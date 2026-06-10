"""
End-to-end:
  1. Load Multilingual HateCheck (filtered to the 6 studied functionalities)
     and write it as JSONL to ``mhc.out_path``.
  2. Load HASOC for the configured languages from ``hasoc.root``.
  3. Assemble + dedupe the training corpus and write it to ``corpus.out_path``.
  4. Compute the per-language coverage report (the Phase-2 decision input) and
     persist it to ``coverage.out_path``.

Implemented as a plain function so it's importable from tests and from the CLI.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from nlp4s.data.corpus import assemble_corpus, coverage_report, missing_languages
from nlp4s.data.hasoc import load_hasoc
from nlp4s.data.mhc import load_mhc
from nlp4s.functionalities import MHC_LANGUAGES
from nlp4s.io_utils import write_jsonl
from nlp4s.schema import Example

_DEFAULT_MHC_OUT = "data/processed/mhc.jsonl"
_DEFAULT_COVERAGE_OUT = "outputs/data/coverage.json"


def _format_coverage_table(counts: dict[str, int]) -> str:
    width = max(len(lang) for lang in counts) + 2
    lines = ["language  count", "--------  -----"]
    for lang in MHC_LANGUAGES:
        marker = "" if counts.get(lang, 0) > 0 else "  <- missing (gen target)"
        lines.append(f"{lang:<{width}}{counts.get(lang, 0):>5}{marker}")
    return "\n".join(lines)


def run(cfg: dict[str, Any]) -> dict[str, int]:
    """Execute the prep pipeline. Returns the coverage report."""
    mhc_cfg = cfg.get("mhc", {})
    hasoc_cfg = cfg.get("hasoc", {})
    corpus_cfg = cfg.get("corpus", {})

    mhc_out = Path(mhc_cfg.get("out_path", _DEFAULT_MHC_OUT))
    coverage_out = Path(corpus_cfg.get("coverage_out", _DEFAULT_COVERAGE_OUT))
    corpus_out = Path(corpus_cfg["out_path"])
    languages = list(corpus_cfg.get("languages", []))

    # 1. MHC — best-effort: a missing dep / no-network shouldn't block corpus prep.
    try:
        mhc_examples = load_mhc(mhc_cfg["hf_dataset"], cache_dir=mhc_cfg.get("cache_dir"))
        write_jsonl(mhc_out, mhc_examples)
        print(f"[prep] MHC: wrote {len(mhc_examples)} examples to {mhc_out}")
    except Exception as exc:  # noqa: BLE001 — surface but don't abort prep
        warnings.warn(
            f"[prep] MHC load failed ({exc!r}); skipping MHC dump and continuing "
            "with training-corpus assembly.",
            stacklevel=2,
        )

    # 2. HASOC.
    hasoc_examples: list[Example] = []
    if languages:
        hasoc_examples = load_hasoc(hasoc_cfg["root"], languages)
        print(f"[prep] HASOC: loaded {len(hasoc_examples)} examples "
              f"across {len(languages)} requested languages.")
    else:
        warnings.warn(
            "[prep] corpus.languages is empty; assembling an empty training corpus.",
            stacklevel=2,
        )

    # 3. Assemble + write.
    corpus = assemble_corpus([hasoc_examples])
    write_jsonl(corpus_out, corpus)
    print(f"[prep] corpus: wrote {len(corpus)} examples to {corpus_out}")

    # 4. Coverage report.
    counts = coverage_report(corpus)
    missing = missing_languages(corpus)
    coverage_out.parent.mkdir(parents=True, exist_ok=True)
    coverage_out.write_text(
        json.dumps(
            {
                "counts": counts,
                "missing_languages": missing,
                "total": len(corpus),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[prep] coverage: wrote {coverage_out}")
    print(_format_coverage_table(counts))
    print(f"[prep] generation targets ({len(missing)}): {missing}")
    return counts

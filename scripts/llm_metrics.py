"""Score LLM predictions against MHC gold labels and persist the results.

Joins a predictions JSONL to the MHC dump and reports, per (model, condition),
overall accuracy plus the explicit / implicit / control breakdown and a
JSON-formatting health metric. Writes both a CSV (machine-readable) and a TXT
table (human-readable) so results are logged to disk rather than only printed.

Usage:
    python -m scripts.llm_metrics                         # defaults below
    python -m scripts.llm_metrics outputs/llm/predictions.jsonl outputs/llm/metrics
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from nlp4s.functionalities import group_of
from nlp4s.io_utils import read_jsonl, read_predictions
from nlp4s.llm.prompts import PARSE_FAILURE_RATIONALE

_MHC = "data/processed/mhc.jsonl"
_GROUPS = ("explicit", "implicit", "control")


def score(predictions_path: str, mhc_path: str = _MHC) -> list[dict]:
    gold = {e.id: e for e in read_jsonl(mhc_path)}
    preds = read_predictions(predictions_path)

    cells: dict[tuple[str, str], list] = defaultdict(list)
    for p in preds:
        cells[(p.model, p.condition)].append(p)

    rows: list[dict] = []
    for (model, condition), ps in sorted(cells.items()):
        n = len(ps)
        hit = defaultdict(lambda: [0, 0])  # group -> [correct, total]
        allc = [0, 0]
        true_fail = 0      # label unparseable -> defaulted to non-hateful
        missing_expl = 0   # valid label but explanation omitted (explanation cond only)
        for p in ps:
            flagged = bool(p.rationale and PARSE_FAILURE_RATIONALE in p.rationale)
            if flagged and p.pred_label == "hateful":
                missing_expl += 1            # label clearly parsed; only explanation missing
            elif flagged:
                true_fail += 1               # defaulted (label unparseable or genuinely non-hateful)
            e = gold.get(p.example_id)
            if not e:
                continue
            try:
                g = group_of(e.functionality)
            except KeyError:
                g = "other"
            ok = int(p.pred_label == e.label)
            hit[g][0] += ok
            hit[g][1] += 1
            allc[0] += ok
            allc[1] += 1

        def acc(grp: str) -> float | str:
            c = hit[grp]
            return round(100 * c[0] / c[1], 1) if c[1] else ""

        # The experiment runner encodes the few-shot strategy in the model name
        # as "<model>::<selection>" (RQ3.4). Split it back out for a clean table.
        model_name, _, selection = model.partition("::")
        rows.append({
            "model": model_name,
            "selection": selection or "-",
            "condition": condition,
            "n": n,
            "accuracy": round(100 * allc[0] / allc[1], 1) if allc[1] else "",
            "explicit": acc("explicit"),
            "implicit": acc("implicit"),
            "control": acc("control"),
            "true_parsefail_pct": round(100 * true_fail / n, 1),
            "missing_expl_pct": round(100 * missing_expl / n, 1),
        })
    return rows


def main() -> None:
    pred_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/llm/predictions.jsonl"
    out_stem = Path(sys.argv[2] if len(sys.argv) > 2 else "outputs/llm/metrics")
    rows = score(pred_path)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model", "selection", "condition", "n", "accuracy", "explicit",
            "implicit", "control", "true_parsefail_pct", "missing_expl_pct"]

    with (csv_path := out_stem.with_suffix(".csv")).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    header = (f"{'model':<12}{'select':<14}{'cond':<16}{'n':>5}{'acc':>7}"
              f"{'expl':>7}{'impl':>7}{'ctrl':>7}{'pfail%':>8}{'noexpl%':>9}")
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['model']:<12}{r['selection']:<14}{r['condition']:<16}{r['n']:>5}"
            f"{r['accuracy']:>7}{r['explicit']:>7}{r['implicit']:>7}{r['control']:>7}"
            f"{r['true_parsefail_pct']:>8}{r['missing_expl_pct']:>9}"
        )
    table = "\n".join(lines)
    (txt_path := out_stem.with_suffix(".txt")).write_text(table + "\n")

    print(table)
    print(f"\n[metrics] wrote {csv_path} and {txt_path}")


if __name__ == "__main__":
    main()

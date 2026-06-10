"""Explanation-quality scoring for RQ2.3 (reference-free, BERTScore).

For the ``explanation`` condition, scores each generated rationale and relates
its quality to classification correctness, broken down by implicit vs explicit
(RQ2.2 / RQ2.3).

Quality metric: multilingual **BERTScore-F1** between the generated explanation
(candidate) and the classified input text (reference). MHC carries no gold
rationales, so we use the input text as the reference — a reference-free
*groundedness* proxy: a good rationale should be semantically anchored in the
text it explains. Caveat: rationales are often English while the text is in the
target language, so absolute scores are noisy across languages; the
informative signal is the *within-model* gap between correctly- and
incorrectly-classified items (point-biserial correlation), not the raw value.
A stronger but costlier alternative is an LLM judge (G-Eval) — left as follow-up.

Usage:
    python scripts/explanation_quality.py [predictions.jsonl] [out_stem]
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
_BERT_MODEL = "bert-base-multilingual-cased"  # covers all MHC languages


def main() -> None:
    pred_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/llm/predictions.jsonl"
    out_stem = Path(sys.argv[2] if len(sys.argv) > 2 else "outputs/llm/explanation_quality")

    gold = {e.id: e for e in read_jsonl(_MHC)}
    preds = read_predictions(pred_path)

    # Keep only explanation-condition rows with a genuine (non-fallback) rationale.
    items = []
    for p in preds:
        if p.condition != "explanation":
            continue
        if not p.rationale or PARSE_FAILURE_RATIONALE in p.rationale:
            continue
        e = gold.get(p.example_id)
        if not e:
            continue
        items.append((p, e))

    if not items:
        print("[explquality] no scorable explanation rows found; run the "
              "explanation condition first.")
        return

    print(f"[explquality] scoring {len(items)} explanations with BERTScore "
          f"({_BERT_MODEL}); this loads a transformer and may take a few minutes...")
    from transformers import logging as hf_logging  # quiet the "unused weights" notice
    hf_logging.set_verbosity_error()
    from bert_score import score as bertscore  # heavy import

    cands = [p.rationale for p, _ in items]
    refs = [e.text for _, e in items]
    _, _, f1 = bertscore(
        cands, refs, model_type=_BERT_MODEL, num_layers=9, verbose=False, batch_size=64
    )
    f1 = f1.tolist()

    # Aggregate per (model, group); track correctness for the correlation.
    agg: dict[tuple[str, str], list] = defaultdict(list)
    per_model_pairs: dict[str, list] = defaultdict(list)
    for (p, e), bs in zip(items, f1):
        try:
            g = group_of(e.functionality)
        except KeyError:
            g = "other"
        correct = int(p.pred_label == e.label)
        model_name = p.model.partition("::")[0]
        sel = p.model.partition("::")[2] or "-"
        agg[(p.model, g)].append((bs, correct))
        per_model_pairs[p.model].append((bs, correct))

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else ""

    rows = []
    for (model, g), pairs in sorted(agg.items()):
        bs_vals = [b for b, _ in pairs]
        corr_vals = [b for b, c in pairs if c]
        inc_vals = [b for b, c in pairs if not c]
        model_name, _, sel = model.partition("::")
        rows.append({
            "model": model_name,
            "selection": sel or "-",
            "group": g,
            "n": len(pairs),
            "mean_bertscore_f1": mean(bs_vals),
            "bs_correct": mean(corr_vals),
            "bs_incorrect": mean(inc_vals),
        })

    # Point-biserial correlation (BERTScore vs correctness) per model, all groups.
    from scipy.stats import pointbiserialr
    corr_rows = []
    for model, pairs in sorted(per_model_pairs.items()):
        bs_vals = [b for b, _ in pairs]
        c_vals = [c for _, c in pairs]
        model_name, _, sel = model.partition("::")
        if len(set(c_vals)) < 2:  # need both correct & incorrect for a correlation
            r, pval = "", ""
        else:
            res = pointbiserialr(c_vals, bs_vals)
            r, pval = round(res.correlation, 4), round(res.pvalue, 4)
        corr_rows.append({"model": model_name, "selection": sel or "-",
                          "n": len(pairs), "pointbiserial_r": r, "p_value": pval})

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    with out_stem.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with Path(str(out_stem) + "_correlation.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(corr_rows[0].keys()))
        w.writeheader(); w.writerows(corr_rows)

    # Human-readable
    lines = ["Explanation quality (BERTScore-F1 of explanation vs. input text)", ""]
    h = f"{'model':<12}{'select':<14}{'group':<10}{'n':>5}{'meanBS':>9}{'BS|correct':>12}{'BS|wrong':>11}"
    lines += [h, "-" * len(h)]
    for r in rows:
        lines.append(f"{r['model']:<12}{r['selection']:<14}{r['group']:<10}{r['n']:>5}"
                     f"{r['mean_bertscore_f1']:>9}{r['bs_correct']:>12}{r['bs_incorrect']:>11}")
    lines += ["", "Correctness correlation (point-biserial r: BERTScore vs. correct)"]
    h2 = f"{'model':<12}{'select':<14}{'n':>5}{'r':>9}{'p':>9}"
    lines += [h2, "-" * len(h2)]
    for r in corr_rows:
        lines.append(f"{r['model']:<12}{r['selection']:<14}{r['n']:>5}{r['pointbiserial_r']:>9}{r['p_value']:>9}")
    table = "\n".join(lines)
    Path(str(out_stem) + ".txt").write_text(table + "\n")

    print(table)
    print(f"\n[explquality] wrote {out_stem.with_suffix('.csv')}, "
          f"{out_stem}_correlation.csv, {out_stem}.txt")


if __name__ == "__main__":
    main()

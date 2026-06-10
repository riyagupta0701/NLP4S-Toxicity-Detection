"""Load and normalise HASOC training data into the shared schema.

HASOC is a coarse binary HOF/NOT task; only English/German/Hindi overlap with
the MHC evaluation languages, and there is no implicit/explicit annotation
(``functionality`` is left empty on every loaded record).

Expected on-disk layout under ``root`` (HASOC files are not redistributable, so
obtain them from the shared-task organisers and place them here)::

    data/raw/hasoc/
      en/   hasoc_2019_en_train.tsv  hasoc_2020_en_train.tsv  ...
      de/   hasoc_2019_de_train.tsv  ...
      hi/   hasoc_2019_hi_train.tsv  ...

Each file must be tab- or comma-separated with at minimum:
    ``text`` column  — accepted variants: ``text``, ``tweet``, ``comment``, ``post``.
    ``label`` column — accepted variants: ``task_1``, ``task1``, ``label``,
                       ``subtask_1``, ``Task 1``. Values: ``HOF`` / ``NOT``.

Missing per-language directories are skipped with a warning; missing columns
raise ``ValueError`` so a bad file fails loudly rather than silently producing
empty data.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Iterable

from nlp4s.functionalities import HASOC_OVERLAP
from nlp4s.schema import Example

_TEXT_COLUMN_CANDIDATES: tuple[str, ...] = ("text", "tweet", "comment", "post", "Text")
_LABEL_COLUMN_CANDIDATES: tuple[str, ...] = (
    "task_1", "task1", "Task 1", "subtask_1", "label", "Label", "class",
)
_FILE_GLOBS: tuple[str, ...] = ("*.tsv", "*.csv", "*.txt")


def _pick_column(header: list[str], candidates: tuple[str, ...], *, role: str, path: Path) -> str:
    lookup = {name.strip().lower(): name for name in header}
    for cand in candidates:
        key = cand.strip().lower()
        if key in lookup:
            return lookup[key]
    raise ValueError(
        f"{path}: no {role} column found (looked for {list(candidates)}; "
        f"header was {header})"
    )


def _normalise_hof_label(raw: str) -> str | None:
    """Map a HASOC task-1 value to schema.LABELS, or None for unknown."""
    v = (raw or "").strip().upper()
    if v == "HOF":
        return "hateful"
    if v == "NOT":
        return "non-hateful"
    return None


def _iter_rows(path: Path) -> Iterable[dict[str, str]]:
    """Yield rows from a TSV/CSV file, auto-detecting the delimiter."""
    with path.open("r", encoding="utf-8", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
        except csv.Error:
            dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel
        reader = csv.DictReader(fh, dialect=dialect)
        for row in reader:
            yield row


def _load_file(path: Path, language: str) -> list[Example]:
    rows = list(_iter_rows(path))
    if not rows:
        return []
    header = list(rows[0].keys())
    text_col = _pick_column(header, _TEXT_COLUMN_CANDIDATES, role="text", path=path)
    label_col = _pick_column(header, _LABEL_COLUMN_CANDIDATES, role="label", path=path)

    out: list[Example] = []
    stem = path.stem
    for i, row in enumerate(rows):
        text = (row.get(text_col) or "").strip()
        if not text:
            continue
        label = _normalise_hof_label(row.get(label_col) or "")
        if label is None:
            continue
        out.append(
            Example(
                text=text,
                language=language,
                label=label,
                functionality="",
                split="train",
                id=f"hasoc-{language}-{stem}-{i}",
            )
        )
    return out


def _load_language(root: Path, language: str) -> list[Example]:
    lang_dir = root / language
    if not lang_dir.is_dir():
        warnings.warn(
            f"HASOC: no directory at {lang_dir} for language {language!r}; "
            "skipping (drop the per-edition TSV/CSV files here).",
            stacklevel=2,
        )
        return []
    files: list[Path] = []
    for pattern in _FILE_GLOBS:
        files.extend(sorted(lang_dir.glob(pattern)))
    if not files:
        warnings.warn(
            f"HASOC: no TSV/CSV/TXT files under {lang_dir}; skipping {language!r}.",
            stacklevel=2,
        )
        return []
    out: list[Example] = []
    for path in files:
        out.extend(_load_file(path, language))
    return out


def load_hasoc(root: str, languages: list[str]) -> list[Example]:
    """Load HASOC editions for the requested languages as Example records.

    Args:
        root: directory containing per-language HASOC subdirectories.
        languages: ISO 639-1 codes to load (typically the MHC overlap: en/de/hi).

    Returns:
        Example records with split="train" and functionality="". Missing
        per-language directories are skipped with a warning so a partial
        download still yields a usable training corpus.
    """
    root_path = Path(root)
    out: list[Example] = []
    for language in languages:
        if language not in HASOC_OVERLAP:
            warnings.warn(
                f"HASOC: language {language!r} is outside the MHC overlap "
                f"{sorted(HASOC_OVERLAP)}; loading anyway if files are present.",
                stacklevel=2,
            )
        out.extend(_load_language(root_path, language))
    return out

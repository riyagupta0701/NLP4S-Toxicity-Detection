"""Quality filtering and deduplication for synthetically generated examples.

Four filters, applied in order by ``quality_filter`` (cheap → expensive):
  1. Refusal detection — drop LLM safety-filter responses ("I cannot ...",
     "as an AI ...", etc.) that occasionally leak through despite the
     research-framed prompt.
  2. Length bounds (drop empty / one-word / runaway outputs).
  3. Label/functionality consistency (drop examples whose label contradicts
     the bucket their functionality belongs to).
  4. Language ID (drop generations that drifted off the requested language).
     Uses ``fasttext-langdetect`` when available; degrades gracefully to a
     no-op + warning if the dependency is missing.

Role A.
"""

from __future__ import annotations

import unicodedata
import warnings
from collections.abc import Iterable

from nlp4s.functionalities import CONTROL, EXPLICIT, IMPLICIT, SYNTHETIC, group_of
from nlp4s.schema import Example

# Length bounds in whitespace-separated tokens — matches the prompt's "5-30
# words" guidance with some slack.
_MIN_WORDS = 3
_MAX_WORDS = 50
_MAX_CHARS = 400


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().lower()
    return " ".join(text.split())


def deduplicate(examples: Iterable[Example]) -> list[Example]:
    """Remove exact / near-duplicate generated examples.

    Dedup key is ``(language, NFKC-normalised, lowercased, whitespace-collapsed text)``
    — matches the corpus-level dedupe in ``nlp4s.data.corpus``.
    """
    seen: set[tuple[str, str]] = set()
    out: list[Example] = []
    for example in examples:
        key = (example.language, _normalise(example.text))
        if key in seen:
            continue
        seen.add(key)
        out.append(example)
    return out


def _length_ok(text: str) -> bool:
    if len(text) > _MAX_CHARS:
        return False
    words = text.split()
    return _MIN_WORDS <= len(words) <= _MAX_WORDS


# Common English refusal/meta phrases from aligned LLMs. Matching is
# case-insensitive substring. The list intentionally targets distinctive
# fragments that don't appear in normal Spanish/French/etc. content, so a
# Spanish sentence containing the word "cannot" won't false-positive (Spanish
# wouldn't naturally include English phrases like "I cannot fulfill").
_REFUSAL_MARKERS: tuple[str, ...] = (
    "i cannot fulfill",
    "i cannot generate",
    "i cannot provide",
    "i can't fulfill",
    "i can't generate",
    "i can't provide",
    "i'm unable to",
    "i am unable to",
    "my purpose is to be helpful",
    "my purpose is to",
    "as an ai",
    "as a language model",
    "as a large language model",
    "i must decline",
    "i won't",
    "i will not generate",
    "harmful and inaccurate",
    "harmful and discriminatory",
    "goes against",
    "against this core principle",
    "promoting respect",
    "respectful and inclusive language",
    "i'm sorry, but",
    "i am sorry, but",
    "instead of generating",
    "instead of providing",
    "let's focus on",
    "i can provide you with resources",
)


def _looks_like_refusal(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _REFUSAL_MARKERS)


def _label_functionality_ok(example: Example) -> bool:
    """Hateful -> EXPLICIT|IMPLICIT; non-hateful -> CONTROL or SYNTHETIC."""
    func = example.functionality
    if not func:
        return True  # no functionality annotation -> can't check
    try:
        group_of(func)
    except KeyError:
        return False
    if example.label == "hateful":
        return func in EXPLICIT or func in IMPLICIT
    return func in CONTROL or func in SYNTHETIC  # non-hateful


# --- Language ID --------------------------------------------------------------

_LANGID_DETECTOR = None
_LANGID_WARNED = False


def _load_langid():
    """Lazy-load the fasttext-langdetect detector; cache the function."""
    global _LANGID_DETECTOR, _LANGID_WARNED
    if _LANGID_DETECTOR is not None:
        return _LANGID_DETECTOR
    try:
        from ftlangdetect import detect  # type: ignore
    except ImportError:
        if not _LANGID_WARNED:
            warnings.warn(
                "fasttext-langdetect not installed; skipping language-ID check "
                "in quality_filter. `pip install fasttext-langdetect` to enable.",
                stacklevel=2,
            )
            _LANGID_WARNED = True
        _LANGID_DETECTOR = False  # sentinel: "tried and failed"
        return False

    def _detect(text: str) -> str | None:
        # ftlangdetect dislikes newlines.
        cleaned = " ".join(text.split())
        if not cleaned:
            return None
        try:
            return detect(text=cleaned, low_memory=True)["lang"]
        except Exception:  # noqa: BLE001
            return None

    _LANGID_DETECTOR = _detect
    return _detect


def _language_ok(example: Example) -> bool:
    detector = _load_langid()
    if not detector:
        return True  # no detector available -> don't reject
    detected = detector(example.text)
    if detected is None:
        return True  # detection failed -> don't reject
    return detected == example.language


# --- Public entry point -------------------------------------------------------


def quality_filter(examples: Iterable[Example]) -> list[Example]:
    """Drop low-quality / off-target generations.

    Applies refusal, length, label/functionality-consistency, and language-ID
    checks in that order (cheap → expensive).
    """
    out: list[Example] = []
    for example in examples:
        if _looks_like_refusal(example.text):
            continue
        if not _length_ok(example.text):
            continue
        if not _label_functionality_ok(example):
            continue
        if not _language_ok(example):
            continue
        out.append(example)
    return out

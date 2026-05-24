"""Role-A tests: MHC row mapping, HASOC loading, corpus assembly + coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlp4s.data.corpus import assemble_corpus, coverage_report, missing_languages
from nlp4s.data.hasoc import load_hasoc
from nlp4s.data.mhc import KEEP_FUNCTIONALITIES, to_example
from nlp4s.functionalities import (
    CONTROL,
    EXPLICIT,
    IMPLICIT,
    MHC_LANGUAGES,
    languages_without_training_data,
)
from nlp4s.schema import Example

FIXTURES = Path(__file__).parent / "fixtures"
HASOC_ROOT = FIXTURES / "hasoc"


# --- MHC -----------------------------------------------------------------------


def test_keep_functionalities_is_the_six_studied_columns():
    assert KEEP_FUNCTIONALITIES == EXPLICIT | IMPLICIT | CONTROL
    assert KEEP_FUNCTIONALITIES == {
        "derog_neg_attrib_h",
        "derog_dehum_h",
        "derog_impl_h",
        "slur_h",
        "profanity_h",
        "profanity_nh",
    }


def test_to_example_normalises_iso6393_language_and_string_label():
    row = {
        "functionality": "derog_impl_h",
        "text": "We all know what those people are really like.",
        "is_hateful": "hateful",
        "lang": "eng",
    }
    ex = to_example(row, idx=0)
    assert ex.language == "en"
    assert ex.label == "hateful"
    assert ex.functionality == "derog_impl_h"
    assert ex.split == "test"
    assert ex.id == "mhc-en-0"


@pytest.mark.parametrize(
    "code3,code1",
    [("ara", "ar"), ("cmn", "zh"), ("deu", "de"), ("fra", "fr"),
     ("hin", "hi"), ("ita", "it"), ("nld", "nl"), ("pol", "pl"),
     ("por", "pt"), ("spa", "es"), ("eng", "en")],
)
def test_to_example_covers_all_mhc_languages(code3, code1):
    ex = to_example(
        {"functionality": "slur_h", "text": "x y z", "is_hateful": 1, "lang": code3},
        idx=1,
    )
    assert ex.language == code1
    assert ex.language in MHC_LANGUAGES


def test_to_example_accepts_integer_is_hateful():
    ex0 = to_example(
        {"functionality": "profanity_nh", "text": "x y z", "is_hateful": 0, "lang": "eng"},
        idx=2,
    )
    ex1 = to_example(
        {"functionality": "slur_h", "text": "x y z", "is_hateful": 1, "lang": "eng"},
        idx=3,
    )
    assert ex0.label == "non-hateful"
    assert ex1.label == "hateful"


def test_to_example_rejects_unknown_language():
    with pytest.raises(ValueError):
        to_example(
            {"functionality": "slur_h", "text": "x y z", "is_hateful": 1, "lang": "xxx"},
            idx=4,
        )


# --- HASOC ---------------------------------------------------------------------


def test_load_hasoc_parses_fixture_and_drops_bad_rows():
    examples = load_hasoc(str(HASOC_ROOT), ["en", "de"])
    # en fixture: 6 rows, 1 empty text, 1 unknown label -> 4 keepers
    # de fixture: 3 rows, all valid -> 3 keepers
    assert len(examples) == 7
    assert {e.language for e in examples} == {"en", "de"}
    assert {e.split for e in examples} == {"train"}
    assert all(e.functionality == "" for e in examples)
    assert {e.label for e in examples} == {"hateful", "non-hateful"}


def test_load_hasoc_warns_on_missing_language_directory():
    with pytest.warns(UserWarning, match="no directory"):
        examples = load_hasoc(str(HASOC_ROOT), ["en", "hi"])
    # 'hi' fixture doesn't exist -> still returns en data
    assert all(e.language == "en" for e in examples)


# --- Corpus assembly + coverage ----------------------------------------------


def test_assemble_corpus_dedupes_on_normalised_text():
    inputs = [
        Example(text="Hello world", language="en", label="non-hateful", split="train"),
        Example(text="  hello WORLD ", language="en", label="non-hateful", split="train"),
        Example(text="hello world", language="de", label="non-hateful", split="train"),  # different lang OK
    ]
    out = assemble_corpus([inputs])
    assert len(out) == 2  # second en is a dup; de is distinct
    assert {(e.language, e.text) for e in out} == {
        ("en", "Hello world"),
        ("de", "hello world"),
    }


def test_assemble_corpus_validates_records():
    bad = [Example(text="x", language="en", label="maybe", split="train")]
    with pytest.raises(ValueError):
        assemble_corpus([bad])


def test_coverage_report_defaults_zero_for_all_mhc_languages():
    counts = coverage_report([])
    assert set(counts.keys()) == set(MHC_LANGUAGES)
    assert all(v == 0 for v in counts.values())


def test_missing_languages_on_empty_corpus_matches_all_mhc():
    assert set(missing_languages([])) == set(MHC_LANGUAGES)


def test_missing_languages_on_hasoc_only_matches_static_gap():
    """Before any synthetic data is added, the gap == languages_without_training_data()."""
    hasoc_only = [
        Example(text="a b c", language="en", label="hateful", split="train"),
        Example(text="d e f", language="de", label="non-hateful", split="train"),
        Example(text="g h i", language="hi", label="hateful", split="train"),
    ]
    assert set(missing_languages(hasoc_only)) == set(languages_without_training_data())

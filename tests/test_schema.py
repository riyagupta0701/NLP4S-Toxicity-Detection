"""Tests for the shared contracts: schema, io_utils, functionalities.

Role-specific behaviour (data loading, training, prompting, metrics) is not
covered yet — TODO for each role to add tests alongside their implementation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nlp4s.functionalities import (
    HASOC_OVERLAP,
    MHC_LANGUAGES,
    group_of,
    languages_without_training_data,
)
from nlp4s.io_utils import read_jsonl
from nlp4s.schema import LABELS, Example, validate

FIXTURE = Path(__file__).parent / "fixtures" / "sample.jsonl"


def test_fixture_parses_into_schema():
    examples = read_jsonl(FIXTURE)
    assert len(examples) == 5
    assert all(isinstance(e, Example) for e in examples)
    assert all(e.label in LABELS for e in examples)
    assert all(e.language in MHC_LANGUAGES for e in examples)


def test_group_of_known_functionalities():
    assert group_of("derog_impl_h") == "implicit"
    assert group_of("slur_h") == "explicit"
    assert group_of("profanity_nh") == "control"


def test_group_of_unknown_raises():
    with pytest.raises(KeyError):
        group_of("not_a_functionality")


def test_validate_rejects_bad_label():
    with pytest.raises(ValueError):
        validate(Example(text="x", language="en", label="maybe"))


def test_coverage_gap_is_eight_languages():
    # 11 MHC languages minus the 3 HASOC overlaps (en/de/hi).
    assert len(MHC_LANGUAGES) == 11
    assert HASOC_OVERLAP == {"en", "de", "hi"}
    assert len(languages_without_training_data()) == 8

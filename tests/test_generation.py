"""Role-A tests: synthetic-generation filters and generator (stubbed LLM)."""

from __future__ import annotations

from nlp4s.generation.filters import deduplicate, quality_filter
from nlp4s.generation.generator import (
    build_prompt,
    generate_for_language,
    _parse_pairs,
)
from nlp4s.llm.client import LLMClient
from nlp4s.schema import Example


# --- Filters -------------------------------------------------------------------


def test_deduplicate_collapses_case_and_whitespace_variants():
    inputs = [
        Example(text="Esa gente arruina el barrio.", language="es",
                label="hateful", functionality="derog_impl_h", split="synthetic"),
        Example(text="  esa GENTE arruina el barrio. ", language="es",
                label="hateful", functionality="derog_impl_h", split="synthetic"),
        Example(text="Different sentence entirely.", language="es",
                label="hateful", functionality="derog_impl_h", split="synthetic"),
    ]
    out = deduplicate(inputs)
    assert len(out) == 2


def test_quality_filter_drops_short_long_and_inconsistent():
    examples = [
        # KEEP: clean implicit-hateful
        Example(text="Esa gente arruina el barrio cuando llega aqui siempre.",
                language="es", label="hateful",
                functionality="derog_impl_h", split="synthetic"),
        # KEEP: clean non-hateful
        Example(text="Mis nuevos vecinos trajeron galletas a la fiesta del barrio.",
                language="es", label="non-hateful",
                functionality="profanity_nh", split="synthetic"),
        # DROP: one word
        Example(text="malo", language="es", label="hateful",
                functionality="derog_impl_h", split="synthetic"),
        # DROP: hateful with CONTROL functionality
        Example(text="Mis vecinos trajeron galletas hoy por la manana.",
                language="es", label="hateful",
                functionality="profanity_nh", split="synthetic"),
        # DROP: non-hateful with IMPLICIT functionality
        Example(text="Mis vecinos organizaron una hermosa cena del barrio.",
                language="es", label="non-hateful",
                functionality="derog_impl_h", split="synthetic"),
        # DROP: unknown functionality
        Example(text="Esa gente arruina todo a su paso siempre.",
                language="es", label="hateful",
                functionality="not_a_real_func", split="synthetic"),
        # DROP: > 50 words
        Example(text=" ".join(["palabra"] * 80), language="es",
                label="hateful", functionality="derog_impl_h", split="synthetic"),
    ]
    kept = quality_filter(examples)
    assert len(kept) == 2
    assert all(e.functionality in {"derog_impl_h", "profanity_nh"} for e in kept)


# --- Generator (stubbed LLM) --------------------------------------------------


class _StubClient(LLMClient):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, prompt, *, temperature=0.0, max_tokens=512):
        self.calls += 1
        return self.responses.pop(0) if self.responses else ""


def _seed_demos():
    return [
        Example(text="We all know what those people are really like.",
                language="en", label="hateful",
                functionality="derog_impl_h", split="test"),
        Example(text="My new neighbours brought cookies yesterday.",
                language="en", label="non-hateful",
                functionality="profanity_nh", split="test"),
    ]


def test_build_prompt_uses_human_readable_language_name():
    p = build_prompt("zh", _seed_demos(), pairs=3)
    assert "Mandarin Chinese" in p
    assert "3 MATCHED PAIRS" in p


def test_parse_pairs_handles_fenced_json_and_extra_prose():
    raw = (
        "Sure, here you go:\n"
        "```json\n"
        '[{"hateful":"Esa gente arruina el barrio.","non_hateful":"Mis vecinos son amables."}]\n'
        "```\n"
        "Hope this helps!"
    )
    pairs = _parse_pairs(raw)
    assert pairs == [("Esa gente arruina el barrio.", "Mis vecinos son amables.")]


def test_parse_pairs_line_based_fallback():
    raw = (
        'hateful: "Diese Leute bringen nur Probleme."\n'
        'non_hateful: "Unsere neuen Nachbarn sind sehr freundlich."'
    )
    pairs = _parse_pairs(raw)
    assert pairs == [
        ("Diese Leute bringen nur Probleme.", "Unsere neuen Nachbarn sind sehr freundlich."),
    ]


def test_generate_for_language_emits_matched_pairs_with_right_functionality():
    client = _StubClient([
        '[{"hateful":"Esa gente arruina el barrio numero uno.",'
        ' "non_hateful":"Mis nuevos vecinos trajeron pan recien horneado uno."}]',
        '[{"hateful":"Esa gente arruina el barrio numero dos.",'
        ' "non_hateful":"Mis nuevos vecinos trajeron pan recien horneado dos."}]',
    ])
    out = generate_for_language(client, "es", _seed_demos(), n=4)
    assert len(out) == 4
    hateful = [e for e in out if e.label == "hateful"]
    non_hateful = [e for e in out if e.label == "non-hateful"]
    assert len(hateful) == 2 and len(non_hateful) == 2
    assert all(e.functionality == "derog_impl_h" for e in hateful)
    assert all(e.functionality == "profanity_nh" for e in non_hateful)
    assert all(e.split == "synthetic" for e in out)
    assert all(e.language == "es" for e in out)


def test_generate_for_language_terminates_on_empty_completion():
    client = _StubClient([""])
    out = generate_for_language(client, "fr", _seed_demos(), n=10)
    assert out == []
    assert client.calls == 1  # bailed after first empty response, no spin


def test_generate_for_language_returns_empty_for_nonpositive_n():
    client = _StubClient(["should not be called"])
    assert generate_for_language(client, "fr", _seed_demos(), n=0) == []
    assert client.calls == 0

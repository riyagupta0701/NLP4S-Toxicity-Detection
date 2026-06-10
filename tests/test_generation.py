"""Tests: synthetic-generation filters + hybrid ToxiGen generator (stubbed LLM)."""

from __future__ import annotations

from nlp4s.generation.filters import deduplicate, quality_filter
from nlp4s.generation.generator import (
    TARGETS,
    build_prompt,
    generate_for_language,
    generate_for_target,
    _parse_pairs,
    _select_demo_pool,
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


def test_quality_filter_drops_refusals_short_long_inconsistent():
    examples = [
        # KEEP: clean implicit-hateful
        Example(text="Esa gente arruina el barrio cuando llega aqui siempre.",
                language="es", label="hateful",
                functionality="derog_impl_h", split="synthetic"),
        # KEEP: clean non-hateful synthetic
        Example(text="Mis nuevos vecinos trajeron galletas a la fiesta del barrio.",
                language="es", label="non-hateful",
                functionality="neutral_mention_nh", split="synthetic"),
        # DROP: refusal text
        Example(text="I cannot fulfill your request to generate hateful content.",
                language="es", label="hateful",
                functionality="derog_impl_h", split="synthetic"),
        # DROP: refusal meta
        Example(text="My purpose is to be helpful and harmless.",
                language="es", label="non-hateful",
                functionality="neutral_mention_nh", split="synthetic"),
        # DROP: one word
        Example(text="malo", language="es", label="hateful",
                functionality="derog_impl_h", split="synthetic"),
        # DROP: hateful with CONTROL functionality
        Example(text="Mis vecinos trajeron galletas hoy por la manana.",
                language="es", label="hateful",
                functionality="profanity_nh", split="synthetic"),
    ]
    kept = quality_filter(examples)
    assert len(kept) == 2


def test_refusal_filter_is_case_insensitive_and_substring():
    examples = [
        Example(text="As an AI, I should clarify something here today.",
                language="es", label="hateful",
                functionality="derog_impl_h", split="synthetic"),
        Example(text="I CANNOT generate that kind of content for you.",
                language="es", label="hateful",
                functionality="derog_impl_h", split="synthetic"),
        Example(text="Una frase normal en espanol sobre un grupo cualquiera aqui.",
                language="es", label="hateful",
                functionality="derog_impl_h", split="synthetic"),
    ]
    kept = quality_filter(examples)
    assert len(kept) == 1
    assert "normal en espanol" in kept[0].text


# --- Prompt construction & parsing -------------------------------------------


def _seed_demos():
    return [
        Example(text="Aquellos siempre traen problemas al barrio.", language="es",
                label="hateful", functionality="derog_impl_h", split="test"),
        Example(text="Esa gente nunca aporta nada.", language="es",
                label="hateful", functionality="derog_impl_h", split="test"),
        Example(text="They never contribute anything.", language="en",
                label="hateful", functionality="derog_impl_h", split="test"),
        Example(text="My new neighbours brought cookies.", language="en",
                label="non-hateful", functionality="profanity_nh", split="test"),
    ]


def test_build_prompt_has_research_framing_and_target():
    """The hybrid prompt MUST include the research-purpose preamble that
    bypasses Aya's safety filter, plus the explicit target."""
    p = build_prompt("es", "Muslim people", _seed_demos(), pairs=3)
    assert "helping researchers build" in p
    assert "Hartvigsen" in p  # methodology citation
    assert "Muslim people" in p
    assert "Spanish" in p
    assert "3 matched pairs" in p
    # Pair structure is what bypasses refusal:
    assert "BIASED:" in p and "NEUTRAL:" in p
    # Softened phrasing ("bias / stereotyping" instead of "hateful"):
    assert "implicit bias" in p.lower() or "stereotyping" in p.lower()


def test_parse_pairs_handles_well_formed_numbered_output():
    raw = (
        "1. BIASED: Esa gente siempre trae problemas al barrio.\n"
        "   NEUTRAL: Mis nuevos vecinos organizaron una cena.\n"
        "2. BIASED: Aquellos no se integran y eso preocupa.\n"
        "   NEUTRAL: La comunidad aporta riqueza cultural.\n"
    )
    pairs = _parse_pairs(raw)
    assert len(pairs) == 2
    assert pairs[0] == (
        "Esa gente siempre trae problemas al barrio.",
        "Mis nuevos vecinos organizaron una cena.",
    )


def test_parse_pairs_tolerates_quote_marks_and_extra_prose():
    raw = (
        "Here you go:\n"
        '1. BIASED: "Esa gente siempre causa problemas en el barrio."\n'
        '   NEUTRAL: "Mis vecinos organizaron una cena maravillosa."\n'
        "Hope these help!\n"
    )
    pairs = _parse_pairs(raw)
    assert len(pairs) == 1
    assert pairs[0][0].startswith("Esa gente")
    assert pairs[0][1].startswith("Mis vecinos")


def test_parse_pairs_drops_orphaned_biased_without_neutral():
    raw = (
        "1. BIASED: First biased sentence here.\n"
        "2. BIASED: Second biased sentence without a neutral.\n"
        "   NEUTRAL: But this neutral pairs with the second.\n"
    )
    pairs = _parse_pairs(raw)
    # The orphaned first BIASED gets dropped; second forms a clean pair.
    assert len(pairs) == 1
    assert "Second biased sentence" in pairs[0][0]
    assert "this neutral pairs" in pairs[0][1]


# --- Demo selection -----------------------------------------------------------


def test_select_demo_pool_prefers_target_language_then_english():
    pool = _seed_demos()
    es_pool = _select_demo_pool(pool, "es")
    assert all(d.language == "es" and d.functionality == "derog_impl_h" for d in es_pool)
    de_pool = _select_demo_pool(pool, "de")  # no German demos -> falls back to English
    assert any(d.language == "en" for d in de_pool)
    # All demos returned are hateful (the side the model needs to *learn to write*)
    assert all(d.label == "hateful" for d in de_pool)


# --- Generator (stubbed LLM) --------------------------------------------------


class _StubClient(LLMClient):
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.prompts: list[str] = []

    def complete(self, prompt, *, temperature=0.0, max_tokens=512):
        self.calls += 1
        self.prompts.append(prompt)
        return self.response


def test_generate_for_target_emits_pairs_with_correct_tags():
    canned = (
        "1. BIASED: Esa gente nunca aporta nada a la sociedad civil de aqui.\n"
        "   NEUTRAL: La comunidad musulmana enriquece nuestra ciudad cada dia.\n"
        "2. BIASED: Aquellos siempre causan problemas en el barrio nuevo.\n"
        "   NEUTRAL: Mis vecinos musulmanes organizan eventos comunitarios encantadores.\n"
    )
    client = _StubClient(canned)
    out = generate_for_target(
        client, "es", "muslim", "Muslim people", _seed_demos(), pairs=2, seed=7,
    )
    assert len(out) == 4  # 2 pairs × 2 examples
    biased = [e for e in out if e.label == "hateful"]
    neutral = [e for e in out if e.label == "non-hateful"]
    assert len(biased) == 2 and len(neutral) == 2
    assert all(e.functionality == "derog_impl_h" for e in biased)
    assert all(e.functionality == "neutral_mention_nh" for e in neutral)
    assert all(e.split == "synthetic" for e in out)
    assert all("muslim" in (e.id or "") for e in out)
    # Pair IDs link the b/n halves: same hash before -b / -n suffix.
    biased_hashes = {e.id.rsplit("-", 1)[0] for e in biased}
    neutral_hashes = {e.id.rsplit("-", 1)[0] for e in neutral}
    assert biased_hashes == neutral_hashes


def test_generate_for_target_terminates_on_empty_completion():
    client = _StubClient("")
    out = generate_for_target(
        client, "fr", "muslim", "Muslim people", _seed_demos(), pairs=5, seed=1,
    )
    assert out == []
    assert client.calls == 1


def test_generate_for_language_loops_over_all_targets():
    canned = (
        "1. BIASED: Una frase con sesgo sobre el grupo en cuestion.\n"
        "   NEUTRAL: Una frase respetuosa sobre el mismo grupo de personas.\n"
    )
    client = _StubClient(canned)
    out = generate_for_language(
        client, "es", _seed_demos(), n=26, shots=4, seed=42,
    )
    # One call per target group with pairs_per_target=1 (26 examples / 13 groups / 2 = 1)
    assert client.calls == len(TARGETS) == 13
    assert len(out) == 26
    labels = {e.label for e in out}
    assert labels == {"hateful", "non-hateful"}
    # All 13 targets represented:
    target_ids = {e.id.split("-")[2] for e in out}
    assert target_ids == {t[0] for t in TARGETS}


def test_generate_for_language_returns_empty_for_nonpositive_n():
    client = _StubClient("nope")
    assert generate_for_language(client, "fr", _seed_demos(), n=0) == []
    assert client.calls == 0

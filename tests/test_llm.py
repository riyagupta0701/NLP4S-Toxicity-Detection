"""Role-C tests: prompts, parser, fewshot selection, pool split, cache, orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlp4s.io_utils import read_predictions, write_jsonl
from nlp4s.llm.cache import CachedLLMClient, ResponseCache
from nlp4s.llm.classify import run as classify_run
from nlp4s.llm.client import LLMClient
from nlp4s.llm.fewshot import select, select_bm25, select_random, select_target_group
from nlp4s.llm.pool import split_mhc, subsample_eval
from nlp4s.llm.prompts import (
    PARSE_FAILURE_FALLBACK_LABEL,
    PARSE_FAILURE_RATIONALE,
    build_prompt,
    parse_response,
)
from nlp4s.schema import Example


# --- Prompts ------------------------------------------------------------------


def test_build_prompt_no_explanation_has_strict_json_schema():
    p = build_prompt("Hello.", "en", "no_explanation")
    assert "JSON" in p
    assert '"label"' in p
    assert "explanation" not in p.lower().split("respond with json only,")[1][:200]


def test_build_prompt_explanation_requests_rationale():
    p = build_prompt("Hello.", "en", "explanation")
    assert '"explanation"' in p
    assert "one or two sentences" in p


def test_build_prompt_uses_language_name_not_code():
    p = build_prompt("Hola.", "es", "no_explanation")
    assert "Spanish" in p
    p_zh = build_prompt("你好", "zh", "no_explanation")
    assert "Mandarin Chinese" in p_zh


def test_build_prompt_includes_demonstrations_as_json_answers():
    demos = [
        Example(text="A slur-laden sentence.", language="en", label="hateful",
                functionality="slur_h", split="pool", id="d1"),
        Example(text="A friendly remark.", language="en", label="non-hateful",
                functionality="profanity_nh", split="pool", id="d2"),
    ]
    p = build_prompt("Some text.", "en", "no_explanation", demonstrations=demos)
    assert "A slur-laden sentence." in p
    assert '"label": "hateful"' in p
    assert '"label": "non-hateful"' in p


def test_build_prompt_rejects_unknown_condition():
    with pytest.raises(ValueError):
        build_prompt("x", "en", "nope")


# --- Parser -------------------------------------------------------------------


def test_parse_response_pure_json_no_explanation():
    label, rationale = parse_response('{"label": "hateful"}', "no_explanation")
    assert label == "hateful"
    assert rationale is None


def test_parse_response_pure_json_explanation():
    label, rationale = parse_response(
        '{"label": "non-hateful", "explanation": "It is a compliment."}',
        "explanation",
    )
    assert label == "non-hateful"
    assert rationale == "It is a compliment."


def test_parse_response_tolerates_chatty_preamble_and_code_fences():
    raw = (
        "Sure! Here's the answer:\n"
        "```json\n"
        '{"label": "hateful", "explanation": "Targets a group via stereotype."}\n'
        "```"
    )
    label, rationale = parse_response(raw, "explanation")
    assert label == "hateful"
    assert "stereotype" in rationale


def test_parse_response_label_variants_normalised():
    assert parse_response('{"label": "Hate"}', "no_explanation")[0] == "hateful"
    assert parse_response('{"label": "Non Hateful"}', "no_explanation")[0] == "non-hateful"
    assert parse_response('{"label": "neutral"}', "no_explanation")[0] == "non-hateful"


def test_parse_response_falls_back_on_unparseable_output():
    label, rationale = parse_response("I refuse to answer.", "explanation")
    assert label == PARSE_FAILURE_FALLBACK_LABEL
    assert rationale == PARSE_FAILURE_RATIONALE


def test_parse_response_falls_back_when_label_missing():
    label, rationale = parse_response('{"foo": "bar"}', "no_explanation")
    assert label == PARSE_FAILURE_FALLBACK_LABEL
    assert rationale is None  # no_explanation never returns a rationale


# --- Pool split + subsample ---------------------------------------------------


def _mk(i, lang, func):
    return Example(
        text=f"text-{lang}-{func}-{i}",
        language=lang,
        label="hateful" if "_h" == func[-2:] else "non-hateful",
        functionality=func,
        split="test",
        id=f"{lang}-{func}-{i}",
    )


def test_split_mhc_pulls_per_cell_pool_and_marks_split_pool():
    examples = []
    for lang in ("en", "es"):
        for func in ("derog_impl_h", "profanity_nh"):
            examples.extend(_mk(i, lang, func) for i in range(20))
    eval_, pool = split_mhc(examples, per_cell_pool_size=4, seed=7)
    # 4 cells * 4 = 16 in pool; remainder in eval.
    assert len(pool) == 16
    assert len(eval_) == 80 - 16
    assert all(e.split == "pool" for e in pool)
    # Each cell appears in both slices.
    cell_pool = {(e.language, e.functionality) for e in pool}
    cell_eval = {(e.language, e.functionality) for e in eval_}
    assert cell_pool == cell_eval == {
        ("en", "derog_impl_h"), ("en", "profanity_nh"),
        ("es", "derog_impl_h"), ("es", "profanity_nh"),
    }


def test_split_mhc_is_deterministic_under_same_seed():
    examples = [_mk(i, "en", "derog_impl_h") for i in range(30)]
    a_eval, a_pool = split_mhc(examples, per_cell_pool_size=5, seed=11)
    b_eval, b_pool = split_mhc(examples, per_cell_pool_size=5, seed=11)
    assert [e.id for e in a_pool] == [e.id for e in b_pool]
    assert [e.id for e in a_eval] == [e.id for e in b_eval]


def test_subsample_eval_caps_per_cell():
    examples = []
    for func in ("derog_impl_h", "slur_h"):
        examples.extend(_mk(i, "en", func) for i in range(20))
    out = subsample_eval(examples, per_cell=5, seed=42)
    assert len(out) == 10
    assert sum(1 for e in out if e.functionality == "derog_impl_h") == 5


def test_subsample_eval_noop_when_per_cell_nonpositive():
    examples = [_mk(i, "en", "slur_h") for i in range(7)]
    assert subsample_eval(examples, per_cell=0) == examples


# --- Few-shot selection -------------------------------------------------------


def _pool_with_targets():
    return [
        Example(text="women are bad drivers stereotype", language="en", label="hateful",
                functionality="derog_impl_h", split="pool", id="p1", target="women"),
        Example(text="women are great at sports", language="en", label="non-hateful",
                functionality="profanity_nh", split="pool", id="p2", target="women"),
        Example(text="men have privilege everywhere", language="en", label="hateful",
                functionality="derog_impl_h", split="pool", id="p3", target="men"),
        Example(text="totally unrelated nice weather today", language="de", label="non-hateful",
                functionality="profanity_nh", split="pool", id="p4", target=None),
    ]


def test_select_random_is_seeded_and_capped():
    pool = _pool_with_targets()
    a = select_random(pool, k=2, seed=1)
    b = select_random(pool, k=2, seed=1)
    assert [e.id for e in a] == [e.id for e in b]
    assert len(a) == 2
    assert len(select_random(pool, k=99, seed=1)) == len(pool)


def test_select_bm25_orders_by_overlap():
    pool = _pool_with_targets()
    picks = select_bm25(pool, query="women are amazing", k=2)
    assert {e.id for e in picks} <= {"p1", "p2"}


def test_select_target_group_prefers_matching_target_then_language():
    pool = _pool_with_targets()
    picks = select_target_group(
        pool, query="some query about women", k=2,
        query_target="women", query_language="en",
    )
    assert all(e.target == "women" for e in picks)


def test_select_target_group_fallback_to_language_when_no_target():
    pool = _pool_with_targets()
    picks = select_target_group(
        pool, query="just words", k=2,
        query_target=None, query_language="en",
    )
    # No target -> tier 1 is empty; tier 2 is same-language (en) regardless of target.
    assert all(e.language == "en" for e in picks)


def test_select_dispatcher_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        select("nope", _pool_with_targets(), "q", 2)


# --- Cache --------------------------------------------------------------------


class _CountingClient(LLMClient):
    def __init__(self, response: str = '{"label": "hateful"}'):
        self.response = response
        self.calls = 0

    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 512) -> str:
        self.calls += 1
        return self.response


def test_cached_client_returns_cached_response_on_repeat(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    inner = _CountingClient(response="abc")
    client = CachedLLMClient(inner, cache, model_id="m1")
    assert client.complete("hi") == "abc"
    assert client.complete("hi") == "abc"  # cache hit
    assert inner.calls == 1
    assert client.hits == 1 and client.misses == 1
    cache.close()


def test_cached_client_keys_separate_models(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    inner_a = _CountingClient(response="A")
    inner_b = _CountingClient(response="B")
    a = CachedLLMClient(inner_a, cache, model_id="m1")
    b = CachedLLMClient(inner_b, cache, model_id="m2")
    assert a.complete("hi") == "A"
    assert b.complete("hi") == "B"  # different model -> different cache key
    cache.close()


# --- Orchestrator (stub LLM) --------------------------------------------------


class _StubClient(LLMClient):
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    def complete(self, prompt, *, temperature=0.0, max_tokens=512):
        self.calls += 1
        return json.dumps(self.payload)


def _build_fake_mhc(path: Path) -> None:
    rows = []
    for lang in ("en", "es"):
        for func in ("derog_impl_h", "slur_h", "profanity_nh"):
            for i in range(8):
                rows.append(_mk(i, lang, func))
    write_jsonl(path, rows)


def test_classify_run_writes_predictions_for_each_model_and_condition(tmp_path):
    mhc_path = tmp_path / "mhc.jsonl"
    _build_fake_mhc(mhc_path)
    preds_out = tmp_path / "preds.jsonl"
    cache_path = tmp_path / "cache.sqlite"

    cfg = {
        "models": [
            {"name": "stub-a", "provider": "stub", "model_id": "stub-a-id"},
            {"name": "stub-b", "provider": "stub", "model_id": "stub-b-id"},
        ],
        "prompting": {
            "conditions": ["no_explanation", "explanation"],
            "shots": 0,
            "fewshot_selection": "random",
            "temperature": 0.0,
            "mhc_path": str(mhc_path),
            "synthetic_path": str(tmp_path / "no-such-synth.jsonl"),
            "pool_per_cell": 2,
            "eval_subsample_per_cell": 3,
            "cache_path": str(cache_path),
            "predictions_out": str(preds_out),
        },
    }

    def factory(provider, model_id):
        return _StubClient({"label": "hateful", "explanation": "stub reason"})

    classify_run(cfg, client_factory=factory)

    predictions = read_predictions(preds_out)
    # 2 langs * 3 functionalities = 6 cells; subsample_per_cell=3 -> 18 eval rows;
    # 2 models * 2 conditions -> 72 predictions.
    assert len(predictions) == 18 * 2 * 2
    models = {p.model for p in predictions}
    conds = {p.condition for p in predictions}
    assert models == {"stub-a", "stub-b"}
    assert conds == {"no_explanation", "explanation"}
    # no_explanation has no rationale; explanation has one.
    no_exp = [p for p in predictions if p.condition == "no_explanation"]
    exp = [p for p in predictions if p.condition == "explanation"]
    assert all(p.rationale is None for p in no_exp)
    assert all(p.rationale == "stub reason" for p in exp)


def test_classify_run_uses_demos_when_shots_positive(tmp_path):
    mhc_path = tmp_path / "mhc.jsonl"
    _build_fake_mhc(mhc_path)
    preds_out = tmp_path / "preds.jsonl"
    cache_path = tmp_path / "cache.sqlite"

    captured_prompts: list[str] = []

    class _Capturing(LLMClient):
        def complete(self, prompt, *, temperature=0.0, max_tokens=512):
            captured_prompts.append(prompt)
            return '{"label": "hateful"}'

    cfg = {
        "models": [{"name": "stub", "provider": "stub", "model_id": "stub-id"}],
        "prompting": {
            "conditions": ["no_explanation"],
            "shots": 2,
            "fewshot_selection": "random",
            "mhc_path": str(mhc_path),
            "synthetic_path": str(tmp_path / "no-such.jsonl"),
            "pool_per_cell": 3,
            "eval_subsample_per_cell": 2,
            "cache_path": str(cache_path),
            "predictions_out": str(preds_out),
        },
    }

    classify_run(cfg, client_factory=lambda *_: _Capturing())
    assert captured_prompts, "should have called the client at least once"
    # Demos rendered as `Text: ...` lines inside the prompt.
    assert any("Examples (input -> JSON answer):" in p for p in captured_prompts)


def test_classify_run_errors_without_mhc(tmp_path):
    cfg = {
        "models": [{"name": "s", "provider": "stub", "model_id": "s"}],
        "prompting": {
            "mhc_path": str(tmp_path / "missing.jsonl"),
            "predictions_out": str(tmp_path / "out.jsonl"),
        },
    }
    with pytest.raises(FileNotFoundError):
        classify_run(cfg, client_factory=lambda *_: _StubClient({"label": "hateful"}))

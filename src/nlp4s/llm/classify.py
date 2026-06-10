"""Run multi-LLM prompting over MHC and emit Prediction records (RQ3).

Pipeline:
  1. Read the processed MHC dump (``data/processed/mhc.jsonl``).
  2. Split deterministically into eval + pool (``llm.pool.split_mhc``); the
     pool is the few-shot demonstration source, eval is the test set.
  3. Subsample eval per (language, functionality) cell to fit the API budget.
  4. For each ``model`` x ``condition`` (and shot setting), build a cached
     LLM client, build prompts, call the model, parse responses, emit
     ``Prediction`` records to ``predictions_out``.

The few-shot pool is augmented with synthetic data when available
(``data/synthetic/implicit.jsonl``), which extends coverage for the 8 MHC
languages with no HASOC training data.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any

from nlp4s.io_utils import read_jsonl, write_jsonl
from nlp4s.llm.cache import CachedLLMClient, ResponseCache
from nlp4s.llm.client import LLMClient, build_client
from nlp4s.llm.fewshot import select as select_demos
from nlp4s.llm.pool import split_mhc, subsample_eval
from nlp4s.llm.prompts import build_prompt, parse_response
from nlp4s.llm.prompts import response_format as response_format_for
from nlp4s.schema import Example, Prediction

_DEFAULT_MHC_PATH = "data/processed/mhc.jsonl"
_DEFAULT_SYNTHETIC_PATH = "data/synthetic/implicit.jsonl"
_DEFAULT_CACHE_PATH = "outputs/llm/cache.sqlite"


def _load_pool(
    mhc_pool: list[Example],
    synthetic_path: str | None,
) -> list[Example]:
    """MHC held-out pool + (optional) synthetic data."""
    pool = list(mhc_pool)
    if synthetic_path:
        p = Path(synthetic_path)
        if p.exists():
            synth = read_jsonl(p)
            pool.extend(synth)
            print(f"[llm] pool: +{len(synth)} synthetic examples from {p}")
        else:
            warnings.warn(
                f"[llm] synthetic data {p} not found; pool is MHC-only.",
                stacklevel=2,
            )
    return pool


def _resolve_example_id(example: Example, idx: int) -> str:
    return example.id or f"mhc-{example.language}-{idx}"


def _example_seed(base_seed: int, example_id: str) -> int:
    """Derive a stable per-example seed from the example's identity.

    Used to seed random few-shot selection so each eval example draws the same
    demonstrations regardless of its position in the eval list — i.e. results
    are reproducible across the model x condition passes that re-iterate the
    same examples. ``hashlib`` (not the salted built-in ``hash``) keeps this
    stable across processes.
    """
    digest = hashlib.sha256(example_id.encode("utf-8")).hexdigest()
    return base_seed ^ int(digest[:8], 16)


def run(
    config: dict[str, Any],
    *,
    client_factory=None,
) -> str:
    """Prompting entrypoint per ``llm.yaml``; writes predictions JSONL.

    ``client_factory(provider, model_id) -> LLMClient`` is injected from tests
    to swap in a stub; in production it defaults to ``llm.client.build_client``.
    Returns the predictions output path.
    """
    prompting = config.get("prompting", {})
    models_cfg = config.get("models") or []
    if not models_cfg:
        raise ValueError("config.models must list at least one LLM to run")

    mhc_path = Path(prompting.get("mhc_path", _DEFAULT_MHC_PATH))
    if not mhc_path.exists():
        raise FileNotFoundError(
            f"MHC dump not found at {mhc_path}; run `nlp4s prep` first."
        )

    conditions: list[str] = list(prompting.get("conditions") or ["no_explanation"])
    shots = int(prompting.get("shots", 0))
    selection = str(prompting.get("fewshot_selection", "random"))
    temperature = float(prompting.get("temperature", 0.0))
    max_tokens = int(prompting.get("max_tokens", 256))
    # Structured-output mode for OpenAI-compatible backends (e.g. Ollama). Forces
    # valid JSON, near-eliminating parse failures / missing explanation fields.
    json_mode = bool(prompting.get("json_mode", False))
    # Per-condition overrides: the explanation condition emits a rationale and
    # needs more room than no_explanation's bare label.
    max_tokens_by_condition = prompting.get("max_tokens_by_condition") or {}
    seed = int(prompting.get("seed", 42))

    eval_subsample_per_cell = int(prompting.get("eval_subsample_per_cell", 25))
    pool_per_cell = int(prompting.get("pool_per_cell", 5))

    predictions_out = Path(prompting["predictions_out"])
    cache_path = Path(prompting.get("cache_path", _DEFAULT_CACHE_PATH))
    synthetic_path = prompting.get("synthetic_path", _DEFAULT_SYNTHETIC_PATH)

    mhc_all = read_jsonl(mhc_path)
    print(f"[llm] loaded {len(mhc_all)} MHC examples from {mhc_path}")

    eval_examples, mhc_pool = split_mhc(
        mhc_all, per_cell_pool_size=pool_per_cell, seed=seed
    )
    eval_examples = subsample_eval(
        eval_examples, per_cell=eval_subsample_per_cell, seed=seed
    )
    print(
        f"[llm] split: eval={len(eval_examples)} pool(MHC)={len(mhc_pool)} "
        f"(per_cell_pool_size={pool_per_cell}, eval_subsample_per_cell={eval_subsample_per_cell})"
    )

    pool = _load_pool(mhc_pool, synthetic_path) if shots > 0 else []
    if shots > 0:
        print(f"[llm] few-shot pool size: {len(pool)} (strategy={selection}, k={shots})")

    def _default_factory(provider: str, model_id: str) -> LLMClient:
        return build_client(provider, model_id, json_mode=json_mode)

    factory = client_factory or _default_factory
    cache = ResponseCache(cache_path)
    print(f"[llm] cache: {cache_path} (json_mode={json_mode})")

    predictions: list[Prediction] = []
    try:
        for model_cfg in models_cfg:
            name = model_cfg["name"]
            provider = model_cfg["provider"]
            model_id = model_cfg["model_id"]
            inner: LLMClient = factory(provider, model_id)
            # Namespace the cache key by mode so json_mode responses never collide
            # with previously-cached free-form ones for the same (model, prompt).
            cache_model_id = f"{model_id}|json" if json_mode else model_id
            client = CachedLLMClient(inner, cache, model_id=cache_model_id)
            print(f"[llm] running model: {name} ({provider}/{model_id})")

            for condition in conditions:
                cond_max_tokens = int(
                    max_tokens_by_condition.get(condition, max_tokens)
                )
                # Force a per-condition JSON schema (requires the explanation
                # field) on backends that support it; no-op for test stubs.
                if json_mode:
                    setattr(inner, "response_format", response_format_for(condition))
                preds = _run_one(
                    client=client,
                    eval_examples=eval_examples,
                    pool=pool,
                    model_name=name,
                    condition=condition,
                    shots=shots,
                    selection=selection,
                    temperature=temperature,
                    max_tokens=cond_max_tokens,
                    seed=seed,
                )
                predictions.extend(preds)
                print(
                    f"[llm]   condition={condition}: {len(preds)} predictions "
                    f"(cache hits={client.hits}, misses={client.misses})"
                )
    finally:
        cache.close()

    write_jsonl(predictions_out, predictions)
    print(f"[llm] wrote {len(predictions)} predictions to {predictions_out}")
    return str(predictions_out)


def _run_one(
    *,
    client: LLMClient,
    eval_examples: list[Example],
    pool: list[Example],
    model_name: str,
    condition: str,
    shots: int,
    selection: str,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> list[Prediction]:
    """One pass of (model x condition) over ``eval_examples``."""
    out: list[Prediction] = []
    for idx, example in enumerate(eval_examples):
        example_id = _resolve_example_id(example, idx)
        demos = (
            select_demos(
                selection,
                pool,
                query=example.text,
                k=shots,
                query_target=example.target,
                query_language=example.language,
                seed=_example_seed(seed, example_id),
            )
            if shots > 0 and pool
            else []
        )
        prompt = build_prompt(
            text=example.text,
            language=example.language,
            condition=condition,
            demonstrations=demos,
        )
        raw = client.complete(prompt, temperature=temperature, max_tokens=max_tokens)
        label, rationale = parse_response(raw, condition)
        out.append(
            Prediction(
                example_id=example_id,
                model=model_name,
                pred_label=label,
                condition=condition,
                rationale=rationale,
            )
        )
    return out

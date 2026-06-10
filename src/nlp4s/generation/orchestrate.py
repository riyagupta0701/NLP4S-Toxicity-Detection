"""Orchestrate the synthetic implicit-hate generation step.

Steps:
  1. Resolve target languages from the config, the coverage report written by
     ``nlp4s prep``, or the static ``languages_without_training_data()`` fallback.
  2. Build the LLM client via ``nlp4s.llm.client.build_client``.
  3. Load demonstration examples from the processed MHC dump (implicit hateful
     + non-hateful controls), or fall back to a small built-in English seed.
  4. For each target language: generate pairs, deduplicate, quality-filter, append.
  5. Write the combined synthetic corpus to ``generation.out_path``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from nlp4s.functionalities import IMPLICIT, languages_without_training_data
from nlp4s.generation.filters import deduplicate, quality_filter
from nlp4s.generation.generator import generate_for_language
from nlp4s.io_utils import read_jsonl, write_jsonl
from nlp4s.llm.client import LLMClient, build_client
from nlp4s.schema import Example

# Built-in English seed for demonstration prompts, used only when no MHC dump
# is available. Style-cue only — generation prompts state demos may differ in
# language from the requested output.
_BUILTIN_SEED: tuple[Example, ...] = (
    Example(text="We all know what those people are really like once they move in.",
            language="en", label="hateful",
            functionality="derog_impl_h", split="test", id="seed-en-1"),
    Example(text="They never contribute anything to the neighbourhood.",
            language="en", label="hateful",
            functionality="derog_impl_h", split="test", id="seed-en-2"),
    Example(text="My new neighbours brought over homemade cookies yesterday.",
            language="en", label="non-hateful",
            functionality="profanity_nh", split="test", id="seed-en-3"),
    Example(text="The neighbourhood association threw a wonderful block party.",
            language="en", label="non-hateful",
            functionality="profanity_nh", split="test", id="seed-en-4"),
)


def _load_demonstrations(path: str | None) -> list[Example]:
    if not path:
        return list(_BUILTIN_SEED)
    p = Path(path)
    if not p.exists():
        warnings.warn(
            f"[generate] demonstrations file {p} not found; using built-in English seed. "
            "Run `nlp4s prep` first to produce the processed MHC dump.",
            stacklevel=2,
        )
        return list(_BUILTIN_SEED)
    pool = read_jsonl(p)
    # Keep implicit hate + non-hateful (control). Pure explicit hate is off-style
    # for our generator (which targets implicit).
    filtered = [
        e for e in pool
        if (e.label == "hateful" and e.functionality in IMPLICIT)
        or (e.label == "non-hateful")
    ]
    return filtered or list(_BUILTIN_SEED)


def _resolve_target_languages(
    explicit: list[str], coverage_path: str | None
) -> list[str]:
    if explicit:
        return list(explicit)
    if coverage_path and Path(coverage_path).exists():
        cov = json.loads(Path(coverage_path).read_text(encoding="utf-8"))
        missing = cov.get("missing_languages") or []
        if missing:
            print(f"[generate] target_languages from {coverage_path}: {missing}")
            return list(missing)
    fallback = list(languages_without_training_data())
    warnings.warn(
        f"[generate] no explicit target_languages and no coverage report; "
        f"falling back to languages_without_training_data() = {fallback}",
        stacklevel=2,
    )
    return fallback


def run(cfg: dict[str, Any], *, client: LLMClient | None = None) -> list[Example]:
    """Execute the generation pipeline. Returns the final filtered examples."""
    gen_cfg = cfg.get("generation", {})
    if not gen_cfg:
        raise ValueError("config is missing the `generation:` block")

    target_languages = _resolve_target_languages(
        list(gen_cfg.get("target_languages") or []),
        gen_cfg.get("coverage_path"),
    )
    if not target_languages:
        print("[generate] nothing to do (no target languages).")
        write_jsonl(Path(gen_cfg["out_path"]), [])
        return []

    examples_per_language = int(gen_cfg.get("examples_per_language", 200))
    shots = int(gen_cfg.get("shots_per_prompt", 5))
    temperature = float(gen_cfg.get("temperature", 0.7))
    out_path = Path(gen_cfg["out_path"])
    seed = int(gen_cfg.get("seed", 42))

    demo_pool = _load_demonstrations(gen_cfg.get("demonstrations_path"))
    print(f"[generate] demonstration pool: {len(demo_pool)} examples")

    if client is None:
        provider = gen_cfg.get("provider")
        model_id = gen_cfg.get("model_id")
        if not provider or not model_id:
            raise ValueError(
                "generation.provider and generation.model_id are required when "
                "no client is injected"
            )
        client = build_client(provider, model_id)

    all_examples: list[Example] = []
    for lang_idx, language in enumerate(target_languages):
        # Per-call demo rotation happens inside generate_for_language; the seed
        # we hand it is offset per language for reproducible-yet-distinct draws.
        raw = generate_for_language(
            client,
            language,
            demo_pool,
            n=examples_per_language,
            shots=shots,
            temperature=temperature,
            seed=seed + lang_idx,
        )
        deduped = deduplicate(raw)
        filtered = quality_filter(deduped)
        print(
            f"[generate] {language}: raw={len(raw)} "
            f"deduped={len(deduped)} kept={len(filtered)}"
        )
        all_examples.extend(filtered)

    # Cross-language dedupe in case the LLM repeated itself.
    all_examples = deduplicate(all_examples)
    write_jsonl(out_path, all_examples)
    print(f"[generate] wrote {len(all_examples)} synthetic examples to {out_path}")
    return all_examples

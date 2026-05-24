"""ToxiGen-style synthetic generation of implicit hate / non-hate pairs.

Uses a multilingual LLM (Aya via the LLMClient) and demonstration-based
prompting, seeded from MHC functionality definitions, to produce implicit
examples in MHC languages that lack training data. ToxiGen is English-only, so
generation is the route to *implicit multilingual* coverage.

The generator depends only on the abstract ``nlp4s.llm.client.LLMClient``;
Role C owns the concrete backends. Tests therefore pass a stub client.

Each prompt asks the LLM for a batch of *matched pairs*: one implicit-hateful
sentence (functionality ``derog_impl_h``) and one structurally similar
non-hateful sentence on the same target group. Output is parsed as JSON; a
line-based fallback handles minor formatting drift.

Role A. See docs/assignment.md "Synthetic implicit-multilingual data generation".
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from nlp4s.functionalities import MHC_LANGUAGES
from nlp4s.llm.client import LLMClient
from nlp4s.schema import Example

# Human-readable language names for the prompt — LLMs respond better to names
# than ISO codes. Keys are ISO 639-1 codes from functionalities.MHC_LANGUAGES.
_LANG_NAMES: dict[str, str] = {
    "ar": "Arabic",
    "nl": "Dutch",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "it": "Italian",
    "zh": "Mandarin Chinese",
    "pl": "Polish",
    "pt": "Portuguese",
    "es": "Spanish",
    "en": "English",
}

# Hard caps so a malformed LLM response can't cause a runaway loop.
_MAX_PAIRS_PER_CALL = 5
_MAX_CALLS_PER_LANGUAGE = 200


def _language_name(code: str) -> str:
    if code not in _LANG_NAMES:
        raise ValueError(
            f"Unsupported target language {code!r}; expected one of {list(MHC_LANGUAGES)}"
        )
    return _LANG_NAMES[code]


def _format_demonstration(example: Example) -> str:
    label = "implicit-hateful" if example.label == "hateful" else "non-hateful"
    return f"- ({label}) {example.text}"


def build_prompt(language: str, demonstrations: list[Example], pairs: int) -> str:
    """Build a demonstration-conditioned prompt asking for ``pairs`` matched pairs.

    The schema-shape requested in the prompt matches ``_parse_pairs`` below.
    """
    lang_name = _language_name(language)
    demo_block = "\n".join(_format_demonstration(d) for d in demonstrations) or "(none)"
    return f"""You are helping build a research dataset for hate-speech detection.

Generate {pairs} MATCHED PAIRS of short social-media-style sentences in {lang_name}.

Each pair MUST contain:
  - "hateful": an IMPLICITLY hateful sentence about a real-world target group.
    Implicit hate conveys hostility via stereotyping, coded language, or
    insinuation — NOT via slurs, profanity, or overt dehumanisation.
  - "non_hateful": a structurally similar sentence on the same topic / target
    group that is NOT hateful (a neutral or positive statement).

Hard requirements:
  - Both sentences MUST be written in {lang_name}.
  - 5 to 30 words per sentence.
  - No slurs, no profanity, no explicit threats.
  - The two sentences in a pair should differ in stance, not in topic.

Reference examples (style cue only, may be in a different language):
{demo_block}

Return ONLY a JSON array of {pairs} objects with keys "hateful" and "non_hateful".
Do not include any other text, commentary, or markdown fences.
"""


_JSON_ARRAY_RE = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)
_KV_RE = re.compile(r'"?(hateful|non_hateful|non-hateful)"?\s*[:=]\s*"([^"\n]+)"', re.IGNORECASE)


def _parse_pairs(completion: str) -> list[tuple[str, str]]:
    """Parse LLM output into a list of (hateful_text, non_hateful_text) tuples.

    Tries JSON first; falls back to a permissive key/value scan if the model
    wrapped the JSON in prose or fences.
    """
    pairs: list[tuple[str, str]] = []
    text = completion.strip()

    # Strip common markdown fences.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # 1) Direct JSON.
    candidates: list[Any] = []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            candidates = parsed
    except json.JSONDecodeError:
        match = _JSON_ARRAY_RE.search(text)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    candidates = parsed
            except json.JSONDecodeError:
                candidates = []

    for obj in candidates:
        if not isinstance(obj, dict):
            continue
        h = obj.get("hateful") or obj.get("Hateful")
        n = obj.get("non_hateful") or obj.get("non-hateful") or obj.get("nonHateful")
        if isinstance(h, str) and isinstance(n, str) and h.strip() and n.strip():
            pairs.append((h.strip(), n.strip()))

    if pairs:
        return pairs

    # 2) Line-based fallback: walk key/value matches in order.
    matches = _KV_RE.findall(text)
    current: dict[str, str] = {}
    for key, value in matches:
        norm = "hateful" if key.lower() == "hateful" else "non_hateful"
        current[norm] = value.strip()
        if "hateful" in current and "non_hateful" in current:
            pairs.append((current["hateful"], current["non_hateful"]))
            current = {}
    return pairs


def _make_examples(pairs: list[tuple[str, str]], language: str) -> list[Example]:
    out: list[Example] = []
    for hateful_text, non_hateful_text in pairs:
        pair_id = uuid.uuid4().hex[:12]
        out.append(
            Example(
                text=hateful_text,
                language=language,
                label="hateful",
                functionality="derog_impl_h",
                split="synthetic",
                id=f"syn-{language}-{pair_id}-h",
            )
        )
        out.append(
            Example(
                text=non_hateful_text,
                language=language,
                label="non-hateful",
                # Matched non-hate control on the same topic, mirroring the
                # MHC ``profanity_nh`` control pattern for our pair structure.
                functionality="profanity_nh",
                split="synthetic",
                id=f"syn-{language}-{pair_id}-n",
            )
        )
    return out


def generate_for_language(
    client: LLMClient,
    language: str,
    demonstrations: list[Example],
    n: int,
    *,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> list[Example]:
    """Generate ``n`` synthetic examples in ``language`` via demonstration prompting.

    Examples are emitted as matched pairs (one implicit-hateful + one
    non-hateful control per pair), so the returned list length is even and
    approximately ``n`` (capped by what the LLM actually produces).
    """
    if n <= 0:
        return []
    out: list[Example] = []
    calls = 0
    while len(out) < n and calls < _MAX_CALLS_PER_LANGUAGE:
        remaining_examples = n - len(out)
        pairs_to_request = max(1, min(_MAX_PAIRS_PER_CALL, (remaining_examples + 1) // 2))
        prompt = build_prompt(language, demonstrations, pairs_to_request)
        completion = client.complete(prompt, temperature=temperature, max_tokens=max_tokens)
        calls += 1
        pairs = _parse_pairs(completion)
        if not pairs:
            # Empty response: don't keep spinning forever.
            break
        out.extend(_make_examples(pairs, language))
    return out[:n if n % 2 == 0 else n + 1]

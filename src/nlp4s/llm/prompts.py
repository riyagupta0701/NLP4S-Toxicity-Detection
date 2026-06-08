"""Prompt templates for the no-explanation vs explanation conditions (RQ3).

Design choices (settled with the team):
- Instructions are in English; demonstrations and target text are in the
  target language. Smaller open-weight models degrade with non-English
  instructions; Aya handles both.
- The model is asked to respond in strict JSON
  (``{"label": "...", "explanation": "..."}``) so parsing is robust across
  models. Free-prose drift is handled by extracting the first JSON object.
- Parse failures emit ``pred_label="non-hateful"`` (the conservative default,
  documented for Role D). The fallback is also flagged in the rationale.

Role C.
"""

from __future__ import annotations

import json
import re

from nlp4s.schema import LABELS, Example

CONDITIONS = ("no_explanation", "explanation")

# Human-readable language names so non-English instructions never appear in
# the system prompt itself; reused from generation/generator.py rather than
# imported to keep the LLM modules decoupled.
_LANG_NAMES: dict[str, str] = {
    "ar": "Arabic", "nl": "Dutch", "fr": "French", "de": "German",
    "hi": "Hindi", "it": "Italian", "zh": "Mandarin Chinese", "pl": "Polish",
    "pt": "Portuguese", "es": "Spanish", "en": "English",
}

PARSE_FAILURE_FALLBACK_LABEL = "non-hateful"
PARSE_FAILURE_RATIONALE = "[parse-failure: defaulted to non-hateful]"


def _language_name(code: str) -> str:
    return _LANG_NAMES.get(code, code)


def _format_demo(demo: Example) -> str:
    """Render one demo as a JSON object matching the requested response shape."""
    payload: dict[str, str] = {"text": demo.text, "label": demo.label}
    return json.dumps(payload, ensure_ascii=False)


def build_prompt(
    text: str,
    language: str,
    condition: str,
    demonstrations: list[Example] | None = None,
) -> str:
    """Build the classification prompt for one instance.

    Args:
        text: the instance to classify (in ``language``).
        language: ISO 639-1 code of the instance.
        condition: ``"no_explanation"`` (label only) or ``"explanation"``
            (label + rationale).
        demonstrations: optional few-shot examples to render before the target.

    The response format requested:
        no_explanation:  {"label": "hateful" | "non-hateful"}
        explanation:     {"label": "...", "explanation": "<one or two sentences>"}
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")

    lang_name = _language_name(language)

    schema_line = (
        '{"label": "hateful" | "non-hateful"}'
        if condition == "no_explanation"
        else '{"label": "hateful" | "non-hateful", "explanation": "<one or two sentences>"}'
    )

    instructions = [
        "You are a hate-speech classifier for short social-media-style texts.",
        f"The text below is written in {lang_name}.",
        'Decide whether the text is "hateful" or "non-hateful" toward any group.',
        "A text is hateful if it expresses hostility, derogation, dehumanisation, "
        "slurs, or implicit bias / stereotyping targeting a group or its members. "
        "Profanity alone is not hateful unless it targets a group.",
        f"Respond with JSON only, matching exactly this shape: {schema_line}",
    ]
    if condition == "explanation":
        instructions.append(
            "The explanation must briefly justify the label in one or two sentences."
        )
    instructions.append("Do not include any text outside the JSON object.")

    parts: list[str] = ["\n".join(instructions)]

    if demonstrations:
        demo_lines = [
            "Examples (input -> JSON answer):",
        ]
        for d in demonstrations:
            demo_lines.append(f"Text: {d.text}")
            demo_lines.append(f"Answer: {_format_demo(d)}")
        parts.append("\n".join(demo_lines))

    parts.append(f"Text: {text}\nAnswer:")
    return "\n\n".join(parts)


# Permissive: grab the first {...} block, even if the model wrapped it in
# code fences or chatty preamble. ``re.DOTALL`` so embedded newlines are fine.
_JSON_BLOCK_RE = re.compile(r"\{.*?\}", re.DOTALL)
_LABEL_NORMALISERS: dict[str, str] = {
    "hateful": "hateful",
    "hate": "hateful",
    "non-hateful": "non-hateful",
    "non hateful": "non-hateful",
    "nonhateful": "non-hateful",
    "not hateful": "non-hateful",
    "not_hateful": "non-hateful",
    "neutral": "non-hateful",
}


def _normalise_label(raw: str | None) -> str | None:
    if not raw:
        return None
    key = raw.strip().lower().strip(".\"'")
    return _LABEL_NORMALISERS.get(key)


def parse_response(raw: str, condition: str) -> tuple[str, str | None]:
    """Parse a model response into ``(pred_label, rationale)``.

    ``rationale`` is ``None`` for ``no_explanation``. Parse failures return
    ``(PARSE_FAILURE_FALLBACK_LABEL, PARSE_FAILURE_RATIONALE)`` so downstream
    eval can flag and count them via the rationale field rather than crashing
    a multi-hour run.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")

    label: str | None = None
    explanation: str | None = None

    for match in _JSON_BLOCK_RE.finditer(raw or ""):
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        label = _normalise_label(payload.get("label"))
        if label is None:
            continue
        if condition == "explanation":
            exp = payload.get("explanation")
            if isinstance(exp, str) and exp.strip():
                explanation = exp.strip()
        break

    if label is None:
        rationale = PARSE_FAILURE_RATIONALE if condition == "explanation" else None
        return PARSE_FAILURE_FALLBACK_LABEL, rationale

    if condition == "no_explanation":
        return label, None
    return label, explanation or PARSE_FAILURE_RATIONALE

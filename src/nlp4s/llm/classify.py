"""Run multi-LLM prompting over MHC and emit Prediction records (RQ3).

Role C. Orchestrates: for each configured model x condition (x few-shot
strategy), prompt the LLM on every MHC example and collect predictions +
rationales.
"""

from __future__ import annotations

from typing import Any


def run(config: dict[str, Any]) -> str:
    """Prompting entrypoint per ``llm.yaml``; writes predictions JSONL.

    Returns the predictions output path.

    TODO(Role C): for each model in config["models"], build a client
    (llm.client.build_client), then for each condition/shot setting build prompts
    (llm.prompts), select demonstrations (llm.fewshot), call the client, parse
    responses, and write Prediction records via io_utils.write_jsonl.
    """
    raise NotImplementedError("TODO(Role C): implement prompting orchestration")

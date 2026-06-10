"""Command-line entrypoints (shared contract).

Wires each workstream to a subcommand. Argument parsing is implemented.
Usage:
    nlp4s <command> --config <path>
    nlp4s --help
"""

from __future__ import annotations

import argparse

from nlp4s.config import load_yaml


def _cmd_prep(args: argparse.Namespace) -> None:
    """Load + normalise MHC/HASOC and assemble the training corpus."""
    from nlp4s.data import prep

    prep.run(load_yaml(args.config))


def _cmd_generate(args: argparse.Namespace) -> None:
    """Generate synthetic implicit-multilingual examples."""
    from nlp4s.generation import orchestrate

    orchestrate.run(load_yaml(args.config))


def _cmd_train(args: argparse.Namespace) -> None:
    """Fine-tune the XLM-RoBERTa baseline."""
    from nlp4s.encoder import train as encoder_train

    encoder_train.train(load_yaml(args.config))


def _cmd_infer(args: argparse.Namespace) -> None:
    """Run the encoder over MHC and write predictions."""
    from nlp4s.encoder import infer as encoder_infer

    encoder_infer.run(load_yaml(args.config))


def _cmd_llm(args: argparse.Namespace) -> None:
    """Run multi-LLM prompting (explanation vs no-explanation)."""
    from nlp4s.llm import classify

    classify.run(load_yaml(args.config))


def _cmd_eval(args: argparse.Namespace) -> None:
    """Score predictions and produce the results matrix + figures."""
    from nlp4s.eval import report

    report.run(load_yaml(args.config))


_COMMANDS = {
    "prep": (_cmd_prep, "Load datasets and assemble the training corpus"),
    "generate": (_cmd_generate, "Generate synthetic implicit-multilingual data"),
    "train": (_cmd_train, "Fine-tune the XLM-RoBERTa encoder baseline"),
    "infer": (_cmd_infer, "Run encoder inference over MHC"),
    "llm": (_cmd_llm, "Run multi-LLM prompting experiments"),
    "eval": (_cmd_eval, "Evaluate predictions and build the report"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nlp4s",
        description="Implicit vs Explicit Hate Speech Detection (multilingual) pipelines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (handler, help_text) in _COMMANDS.items():
        sub = subparsers.add_parser(name, help=help_text, description=help_text)
        sub.add_argument("--config", required=True, help="Path to the YAML config for this step.")
        sub.set_defaults(handler=handler)
    return parser


def main(argv: list[str] | None = None) -> None:
    # Load API keys etc. from .env if present, so users don't need to source it.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()

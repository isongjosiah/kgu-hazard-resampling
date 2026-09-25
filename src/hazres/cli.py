"""Command line entry point: ``hazres``."""

from __future__ import annotations

import argparse
import sys

from hazres import __version__, engine_version


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hazres",
        description="Does resampling predictors onto one grid change what a hazard map says?",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"hazres {__version__} (engine {engine_version()})",
    )
    parser.add_subparsers(dest="command", metavar="COMMAND")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Command line entry point: ``hazres``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hazres import __version__, engine_version
from hazres.data.registry import DEFAULT_DATA_ROOT, DEFAULT_REGISTRY


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
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    data = sub.add_parser("data", help="list, download and inspect hazard datasets")
    data.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    data.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    data_sub = data.add_subparsers(dest="data_command", metavar="ACTION", required=True)
    data_sub.add_parser("list", help="every dataset, its tier and whether its files are present")
    fetch = data_sub.add_parser("fetch", help="download a dataset's files into the data root")
    fetch.add_argument("key")
    fetch.add_argument("--force", action="store_true", help="download again even if present")
    inspect = data_sub.add_parser("inspect", help="load a dataset and report what was kept")
    inspect.add_argument("key")
    return parser


def _data(args: argparse.Namespace) -> int:
    from hazres.data.registry import load_inventory, load_registry, resolve

    reg = load_registry(args.registry)
    if args.data_command == "list":
        print(f"{'key':<24}{'tier':<6}{'hazard':<11}{'status':<12}{'absences':<18}files")
        for key, cfg in sorted(reg.items(), key=lambda kv: (kv[1].tier, kv[0])):
            required = {n: s for n, s in cfg.files.items() if s.required}
            present = sum(bool(resolve(s, args.data_root)) for s in required.values())
            files = f"{present}/{len(required)}" if required else "-"
            print(
                f"{key:<24}{cfg.tier:<6}{cfg.hazard:<11}{cfg.status:<12}"
                f"{cfg.real_absences.value:<18}{files}"
            )
        return 0

    if args.key not in reg:
        print(f"unknown dataset {args.key!r}; known: {', '.join(sorted(reg))}", file=sys.stderr)
        return 2

    if args.data_command == "fetch":
        from hazres.data.fetch import fetch

        for r in fetch(reg[args.key], args.data_root, force=args.force):
            line = f"{r.status:<11}{r.name:<18}{r.path}"
            if r.sha256:
                line += f"\n{'':<29}sha256 {r.sha256}"
            if r.note:
                line += f"\n{'':<29}{r.note}"
            print(line)
        return 0

    if args.data_command == "inspect":
        labels = load_inventory(args.key, registry=reg, data_root=args.data_root)
        print(labels.report.describe())
        return 0
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "data":
            return _data(args)
    except (FileNotFoundError, KeyError, ValueError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
    pred = data_sub.add_parser("predictors", help="list and inspect predictor layers")
    pred.add_argument("--predictors", type=Path, default=Path("configs/predictors.yaml"))
    pred_sub = pred.add_subparsers(dest="pred_command", metavar="ACTION", required=True)
    pred_sub.add_parser("list", help="every predictor layer, its native size and status")
    pinspect = pred_sub.add_parser("inspect", help="read one layer for an area and summarise it")
    pinspect.add_argument("key")
    pinspect.add_argument(
        "--bounds", nargs=4, type=float, required=True, metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
        help="area to read, in --bounds-crs (default: the analysis grid CRS)",
    )  # fmt: skip
    pinspect.add_argument("--bounds-crs", default=None)
    gfd = data_sub.add_parser(
        "export-gfd",
        help="export Global Flood Database events for a country from Earth Engine",
        description="Needs: uv sync --extra gee, and uv run earthengine authenticate once.",
    )
    gfd.add_argument(
        "--country", default="Nigeria", help="country name as in LSIB (default Nigeria)"
    )
    gfd.add_argument("--iso3", default="NGA", help="ISO3 code used by the flood database")
    gfd.add_argument("--scale", type=float, default=250.0, help="cell size in m (default 250)")
    gfd.add_argument("--crs", default="EPSG:32632", help="projected CRS (default UTM 32N)")
    gfd.add_argument("--method", choices=["download", "drive"], default="download")
    gfd.add_argument("--drive-folder", default="hazres_gfd")
    gfd.add_argument("--project", help="Google Cloud project registered for Earth Engine")
    gfd.add_argument("--force", action="store_true", help="export again even if the file exists")
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

    if args.data_command == "predictors":
        return _predictors(args)

    if args.data_command == "export-gfd":
        from hazres.data.gee import export_gfd

        out_dir = args.data_root / "global_flood_database" / args.country.lower().replace(" ", "_")
        results = export_gfd(
            out_dir,
            country_name=args.country,
            iso3=args.iso3,
            scale=args.scale,
            crs=args.crs,
            method=args.method,
            drive_folder=args.drive_folder,
            project=args.project,
            skip_existing=not args.force,
        )
        failed = 0
        for job, status in results:
            print(f"{job.filename:<20}{status}")
            failed += status.startswith("failed")
        print(f"{len(results)} events, {failed} failed; manifest: {out_dir / 'events.csv'}")
        return 1 if failed else 0

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


def _predictors(args: argparse.Namespace) -> int:
    from hazres.data.predictors import load_predictors, read_layer

    reg = load_predictors(args.predictors)
    if args.pred_command == "list":
        print(f"{'key':<22}{'group':<12}{'kind':<12}{'status':<9}native size")
        for key, c in reg.layers.items():
            print(f"{key:<22}{c.group:<12}{c.kind.value:<12}{c.status:<9}{c.native_res}")
        return 0

    if args.key not in reg.layers:
        print(f"unknown layer {args.key!r}; known: {', '.join(reg.layers)}", file=sys.stderr)
        return 2
    cfg = reg.layers[args.key]
    crs = args.bounds_crs or reg.analysis_grid.crs
    bounds = tuple(args.bounds)
    if cfg.access == "derived":
        import numpy as np

        from hazres.data.terrain import terrain_on_grid
        from hazres.grid.spec import GridSpec

        grid = GridSpec.covering(bounds, reg.analysis_grid.crs, reg.analysis_grid.res)
        values = terrain_on_grid(grid, reg, data_root=args.data_root)[args.key]
        ok = values[~np.isnan(values)]
        q = np.percentile(ok, [0, 50, 100]) if ok.size else [np.nan] * 3
        print(f"{args.key}: {values.shape[0]:,} x {values.shape[1]:,} cells of {grid.res:g} m")
        print(f"  {cfg.units}: min {q[0]:.4g}, median {q[1]:.4g}, max {q[2]:.4g}")
        return 0
    layer = read_layer(cfg, bounds, crs, data_root=args.data_root)
    if hasattr(layer, "describe"):
        print(layer.describe())
    else:
        n = len(layer.features)
        classes = layer.features[layer.class_column].value_counts().head(8).to_dict()
        print(f"{args.key}: {n:,} polygons; classes {classes}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "data":
            return _data(args)
    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        NotImplementedError,
        ImportError,
        RuntimeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

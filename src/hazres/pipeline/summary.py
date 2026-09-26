"""One table across experiments: model quality and verdicts, side by side."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarise(out_root: Path) -> pd.DataFrame:
    rows = []
    for comp_path in sorted(out_root.glob("*/*/comparison.csv")):
        model_dir = comp_path.parent
        exp, model = model_dir.parent.name, model_dir.name
        comp = pd.read_csv(comp_path)
        scores = pd.read_csv(model_dir / "scores.csv", index_col=0)
        methods = scores[scores.index.str.startswith("method:")]
        row = {
            "experiment": exp,
            "model": model,
            "auc": f"{methods.auc.min():.2f}-{methods.auc.max():.2f}",
            "calibration_slope": round(float(methods.calibration_slope.mean()), 2),
        }
        for scale, g in comp.groupby("scale"):
            sensitive = g.loc[g.verdict == "sensitive", "measure"].tolist()
            row[f"sensitive at {scale}"] = ", ".join(sensitive) if sensitive else "none"
        rows.append(row)
    return pd.DataFrame(rows)

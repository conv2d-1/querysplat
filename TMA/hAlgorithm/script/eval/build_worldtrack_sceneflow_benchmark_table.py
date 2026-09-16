#!/usr/bin/env python3
"""Build markdown benchmark table from WorldTrack scene-flow eval summary.json files.

Each model entry:
  {"name": "...", "subsets": {"adt_mini": path/to/summary.json, ...}}

Uses SF-APD/EPE for all queries and GT-dynamic subset (avg_sf_global_dyn / epe_sf_global_dyn).
Prefers flow-weighted micro averages when ``avg_sf_global_dyn_micro`` exists in summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SUBSET_ORDER = ["pstudio_mini", "po_mini", "ds_mini", "adt_mini"]
SUBSET_SHORT = {
    "pstudio_mini": "PStudio",
    "po_mini": "PO",
    "ds_mini": "DR",
    "adt_mini": "ADT",
}


def _load_summary(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _metric(summary: Dict[str, Any], apd_key: str, epe_key: str) -> tuple[float, float]:
    apd = summary.get(f"{apd_key}_micro", summary.get(apd_key, float("nan")))
    epe = summary.get(f"{epe_key}_micro", summary.get(epe_key, float("nan")))
    return float(apd), float(epe)


def _row_for_model(name: str, subset_paths: Dict[str, Path]) -> Dict[str, Any]:
    cells: Dict[str, tuple[float, float]] = {}
    apd_all: List[float] = []
    epe_all: List[float] = []
    apd_dyn: List[float] = []
    epe_dyn: List[float] = []

    for subset in SUBSET_ORDER:
        path = subset_paths.get(subset)
        if path is None or not path.is_file():
            cells[subset] = (float("nan"), float("nan"))
            continue
        s = _load_summary(path)
        apd, epe = _metric(s, "avg_sf_global", "epe_sf_global")
        apd_d, epe_d = _metric(s, "avg_sf_global_dyn", "epe_sf_global_dyn")
        cells[subset] = (apd, epe)
        if np.isfinite(apd):
            apd_all.append(apd)
        if np.isfinite(epe):
            epe_all.append(epe)
        if np.isfinite(apd_d):
            apd_dyn.append(apd_d)
        if np.isfinite(epe_d):
            epe_dyn.append(epe_d)

    avg_all = (
        float(np.mean(apd_all)) if apd_all else float("nan"),
        float(np.mean(epe_all)) if epe_all else float("nan"),
    )
    avg_dyn = (
        float(np.mean(apd_dyn)) if apd_dyn else float("nan"),
        float(np.mean(epe_dyn)) if epe_dyn else float("nan"),
    )
    return {"name": name, "cells": cells, "avg_all": avg_all, "avg_dyn": avg_dyn}


def _fmt_pct(apd: float) -> str:
    if not np.isfinite(apd):
        return "—"
    return f"{100.0 * apd:.2f}"


def _fmt_epe(epe: float) -> str:
    if not np.isfinite(epe):
        return "—"
    return f"{epe:.4f}"


def build_markdown(models: List[Dict[str, Any]], *, include_dynamic: bool) -> str:
    lines = [
        "# WorldTrack Scene Flow Benchmark (τ=0.1m)",
        "",
        "协议: TMA frame-0 query + pred-dynamic global scale。τ 列为 SF-APD×100（all queries）。",
    ]
    if include_dynamic:
        lines.append(
            "**Dyn-τ / Dyn-EPE**: GT dynamic points（`sum_t ||Δgt|| > 0.01m`）上的 scene-flow APD/EPE，"
            "复用 global scale。"
        )
    lines.append("**Avg** = 四子集宏平均（等权）。")
    lines.append("")

    if include_dynamic:
        header = (
            "| Model | PStudio | | PO | | DR | | ADT | | Avg(all) | | Avg(dyn) | |"
            "\n| | τ↑ | EPE↓ | τ↑ | EPE↓ | τ↑ | EPE↓ | τ↑ | EPE↓ | τ↑ | EPE↓ | Dyn-τ↑ | Dyn-EPE↓ |"
        )
    else:
        header = (
            "| Model | PStudio | | PO | | DR | | ADT | | Avg | |"
            "\n| | τ↑ | EPE↓ | τ↑ | EPE↓ | τ↑ | EPE↓ | τ↑ | EPE↓ | τ↑ | EPE↓ |"
        )
    lines.append(header)

    for model in models:
        row = [model["name"]]
        for subset in SUBSET_ORDER:
            apd, epe = model["cells"].get(subset, (float("nan"), float("nan")))
            row.extend([_fmt_pct(apd), _fmt_epe(epe)])
        apd_a, epe_a = model["avg_all"]
        row.extend([_fmt_pct(apd_a), _fmt_epe(epe_a)])
        if include_dynamic:
            apd_d, epe_d = model["avg_dyn"]
            row.extend([_fmt_pct(apd_d), _fmt_epe(epe_d)])
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build WorldTrack SF benchmark markdown table.")
    p.add_argument(
        "--config-json",
        required=True,
        help='JSON: {"models":[{"name":"...","subsets":{"adt_mini":"path/summary.json"}}]}',
    )
    p.add_argument("--output", required=True, help="Output .md path")
    p.add_argument("--no-dynamic", action="store_true", help="Omit dynamic-points columns")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    models_cfg = cfg.get("models", cfg)
    rows: List[Dict[str, Any]] = []
    for item in models_cfg:
        name = str(item["name"])
        subsets = {k: Path(v) for k, v in item.get("subsets", {}).items()}
        rows.append(_row_for_model(name, subsets))

    md = build_markdown(rows, include_dynamic=not bool(args.no_dynamic))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
make_phase1_figure.py

Single Phase-1 figure:
- Persistent-contact similarity (from *_window_telemetry.jsonl)
- regime boundaries (from *_regimes.json)
- irreversible commitments as tick marks (from *_commitments.json)

This figure is intentionally "mechanism-opaque": it shows results, not how the engine works.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry", type=Path, required=True, help="*_window_telemetry.jsonl")
    ap.add_argument("--regimes", type=Path, required=True, help="*_regimes.json")
    ap.add_argument("--commitments", type=Path, required=True, help="*_commitments.json")
    ap.add_argument("--out", type=Path, required=True, help="Output image path (png recommended)")
    ap.add_argument("--title", type=str, default="HP35 Phase-1: Regimes + Commitments (Geometry-only)")
    args = ap.parse_args()

    telem = read_jsonl(args.telemetry)
    reg = json.loads(args.regimes.read_text(encoding="utf-8"))["regimes"]
    com = json.loads(args.commitments.read_text(encoding="utf-8"))["commitments"]

    # X-axis in frame coordinates using window midpoints
    x = []
    y = []
    motif_sizes = []
    for r in telem:
        t0 = int(r["t_start"])
        t1 = int(r["t_end"])
        mid = (t0 + t1) / 2.0
        x.append(mid)
        y.append(float(r["jaccard_prev"]))
        motif_sizes.append(int(r["motif_size"]))

    if not x:
        raise SystemExit("No telemetry rows read.")

    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()

    # Main curve: window-to-window jaccard
    ax.plot(x, y, linewidth=1.0, label="Persistent-contact similarity vs previous window")

    # Regime boundaries (vertical lines) + labels
    # draw boundary at each regime start except the first
    for i, rj in enumerate(reg):
        t_start = int(rj["t_start"])
        t_end = int(rj["t_end"])
        label = rj.get("label", f"regime_{i}")

        if i > 0:
            ax.axvline(t_start, linewidth=1.0)

        # Put label near top
        mid = (t_start + t_end) / 2.0
        ax.text(mid, 1.02, label, ha="center", va="bottom", fontsize=9)

    # Commitment ticks along bottom
    # Put them slightly below the plotted curve region.
    ymin, ymax = ax.get_ylim()
    # Force a sane y range for jaccard visualization
    ax.set_ylim(0.0, 1.05)
    tick_y0 = 0.05
    tick_y1 = 0.15

    for c in com:
        t = int(c["t_commit"])
        ax.vlines(t, tick_y0, tick_y1, linewidth=1.0)

    # Axis labels and title
    ax.set_xlabel("Frame index (time)")
    ax.set_ylabel("Persistent-contact similarity (window-to-window)")
    ax.set_title(args.title)

    # Light annotation for commitments
    ax.text(x[0], 0.12, "Commitments (irreversible contacts) = tick marks", fontsize=9, va="bottom")

    ax.grid(True, linewidth=0.5)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"[make_phase1_figure] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

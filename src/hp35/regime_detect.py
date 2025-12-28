#!/usr/bin/env python3
"""
regime_detect.py (windowed)


Outputs:
  - telemetry per window (jsonl): persistence-set size, window-to-window jaccard
  - regimes (json): t_start/t_end boundaries in frame indices

Phase-1 friendly: no energies, no chemistry, no MSM.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple


Edge = Tuple[int, int]


def iter_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def edges_from_rec(rec: Dict) -> Set[Edge]:
    return {(int(a), int(b)) if a <= b else (int(b), int(a)) for a, b in rec.get("edges", [])}


@dataclass
class WindowStat:
    w_idx: int
    t_start: int
    t_end: int
    motif_size: int
    jaccard_prev: float


def jaccard(A: Set[Edge], B: Set[Edge]) -> float:
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Windowed regime detection from HP35 contact graphs JSONL.")
    p.add_argument("--in", dest="inp", type=Path, required=True, help="Input JSONL from extract_contacts.py")
    p.add_argument("--out-prefix", type=str, default="hp35_phase1", help="Prefix name for outputs in outputs/runs/")
    p.add_argument("--max-frames", type=int, default=0, help="Process only first N frames (0=all).")

    p.add_argument("--window-W", type=int, default=2000, help="Window size in frames.")
    p.add_argument("--p-thresh", type=float, default=0.80,
                   help="Edge persistence threshold within a window to be included in motif set.")

    p.add_argument("--boundary-jaccard", type=float, default=0.65,
                   help="If window-to-window motif jaccard drops below this, mark a boundary candidate.")
    p.add_argument("--sustain-windows", type=int, default=2,
                   help="How many consecutive boundary candidates needed to cut a regime.")
    p.add_argument("--min-regime-windows", type=int, default=3,
                   help="Minimum windows per regime (prevents micro-regimes).")
    return p


def main() -> int:
    args = build_argparser().parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "outputs" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    telem_path = out_dir / f"{args.out_prefix}_window_telemetry.jsonl"
    regimes_path = out_dir / f"{args.out_prefix}_regimes.json"

    W = max(1, args.window_W)
    p_thresh = args.p_thresh

    # Window accumulator: counts of edges in current window
    edge_counts: Dict[Edge, int] = {}
    frames_in_window = 0
    window_start_t: Optional[int] = None
    window_end_t: Optional[int] = None

    window_motifs: List[Set[Edge]] = []
    window_stats: List[WindowStat] = []

    n_frames = 0
    w_idx = 0

    def flush_window():
        nonlocal w_idx, edge_counts, frames_in_window, window_start_t, window_end_t
        if frames_in_window == 0 or window_start_t is None or window_end_t is None:
            return
        # Build persistent motif set
        motif: Set[Edge] = set()
        for e, c in edge_counts.items():
            if (c / frames_in_window) >= p_thresh:
                motif.add(e)
        # Compute telemetry
        if window_motifs:
            jac = jaccard(motif, window_motifs[-1])
        else:
            jac = 1.0
        window_motifs.append(motif)
        window_stats.append(
            WindowStat(
                w_idx=w_idx,
                t_start=window_start_t,
                t_end=window_end_t,
                motif_size=len(motif),
                jaccard_prev=jac,
            )
        )
        w_idx += 1
        # reset
        edge_counts = {}
        frames_in_window = 0
        window_start_t = None
        window_end_t = None

    # Stream frames -> windows
    for rec in iter_jsonl(args.inp):
        t = int(rec["t"])
        E = edges_from_rec(rec)

        if window_start_t is None:
            window_start_t = t
        window_end_t = t

        for e in E:
            edge_counts[e] = edge_counts.get(e, 0) + 1

        frames_in_window += 1
        n_frames += 1

        if frames_in_window >= W:
            flush_window()

        if args.max_frames and n_frames >= args.max_frames:
            break

    # Flush last partial window
    flush_window()

    # Write telemetry
    with telem_path.open("w", encoding="utf-8") as f:
        for ws in window_stats:
            f.write(json.dumps(ws.__dict__, separators=(",", ":")) + "\n")

    # Regime segmentation based on sustained jaccard drops
    regimes: List[Dict] = []
    if not window_stats:
        params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        regimes_path.write_text(
            json.dumps({"input": str(args.inp), "params": params, "regimes": regimes}, indent=2),
            encoding="utf-8",
        )

        print("[regime_detect] ERROR: no windows produced.")
        return 2

    start_w = 0
    boundary_streak = 0

    def close_regime(end_w: int):
        nonlocal start_w
        t_start = window_stats[start_w].t_start
        t_end = window_stats[end_w].t_end
        regimes.append(
            {
                "t_start": int(t_start),
                "t_end": int(t_end),
                "label": f"regime_{len(regimes)}",
                "windows": int(end_w - start_w + 1),
            }
        )
        start_w = end_w + 1

    for i in range(1, len(window_stats)):
        jac = window_stats[i].jaccard_prev
        is_boundary = jac < args.boundary_jaccard

        if is_boundary:
            boundary_streak += 1
        else:
            boundary_streak = 0

        # cut when sustained, but ensure min regime size
        if boundary_streak >= args.sustain_windows:
            end_w = i - args.sustain_windows  # end before the unstable stretch
            if end_w >= start_w and (end_w - start_w + 1) >= args.min_regime_windows:
                close_regime(end_w)
            boundary_streak = 0

    # close final
    if start_w < len(window_stats):
        close_regime(len(window_stats) - 1)

    params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    regimes_path.write_text(
        json.dumps({"input": str(args.inp), "params": params, "regimes": regimes}, indent=2),
        encoding="utf-8",
    )

    print("[regime_detect] OK (windowed)")
    print(f"  telemetry: {telem_path}")
    print(f"  regimes:   {regimes_path}")
    print(f"  frames:    {n_frames}")
    print(f"  windows:   {len(window_stats)} (W={W})")
    print(f"  regimes#:  {len(regimes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

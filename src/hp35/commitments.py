#!/usr/bin/env python3
"""
commitments.py

Find "irreversible commitments" in HP35 contact graphs:
contacts that appear and then persist for most of the remaining trajectory.

Input: JSONL from extract_contacts.py
Output:
  - commitments JSON (sorted by commit time)
  - per-contact stats JSON (optional)

Definition (Phase 1):
  A contact e is a commitment at time t if:
    - e is present at t
    - and in the suffix [t, T) its presence fraction >= suffix_p (e.g., 0.90)
    - and e was not already committed earlier (first commit time)

We compute this efficiently by tracking last-seen and suffix counts in one reverse pass.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

Edge = Tuple[int, int]


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def edges_from_rec(rec: dict) -> Set[Edge]:
    return {(int(a), int(b)) if a <= b else (int(b), int(a)) for a, b in rec.get("edges", [])}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract irreversible commitment contacts from HP35 contact JSONL.")
    p.add_argument("--in", dest="inp", type=Path, required=True, help="Input JSONL from extract_contacts.py")
    p.add_argument("--out-prefix", type=str, default="hp35_phase1", help="Prefix name for outputs in outputs/runs/")
    p.add_argument("--max-frames", type=int, default=0, help="Process only first N frames (0=all).")
    p.add_argument("--suffix-p", type=float, default=0.90,
                   help="Commitment threshold: presence fraction in suffix window [t..end).")
    p.add_argument("--min-suffix-frames", type=int, default=20000,
                   help="Don’t declare commitments too late; require at least this many frames remaining.")
    return p


def main() -> int:
    args = build_argparser().parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "outputs" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    commitments_path = out_dir / f"{args.out_prefix}_commitments.json"

    # Load edges per frame into memory for the slice we’re analyzing.
    # For 200k frames this is fine. For 1.5M, we’ll stream + do a two-pass approach later.
    frames: List[Set[Edge]] = []
    for rec in iter_jsonl(args.inp):
        frames.append(edges_from_rec(rec))
        if args.max_frames and len(frames) >= args.max_frames:
            break

    T = len(frames)
    if T == 0:
        print("[commitments] ERROR: no frames read.")
        return 2

    suffix_p = args.suffix_p
    min_suffix = args.min_suffix_frames

    # Reverse pass: for each edge, maintain suffix count seen so far
    suffix_count: Dict[Edge, int] = defaultdict(int)
    committed_at: Dict[Edge, int] = {}

    # We also maintain remaining frames count at each t during reverse traversal
    for t in range(T - 1, -1, -1):
        E = frames[t]
        for e in E:
            suffix_count[e] += 1

        remaining = T - t
        # Only consider commitments if there are enough frames remaining (avoid end artifacts)
        if remaining < min_suffix:
            continue

        # Any edge present now can be evaluated
        for e in E:
            if e in committed_at:
                continue
            frac = suffix_count[e] / remaining
            if frac >= suffix_p:
                committed_at[e] = t

    # Build sorted list
    commits = []
    for e, t in committed_at.items():
        commits.append({"t_commit": int(t), "edge": [e[0], e[1]]})

    commits.sort(key=lambda x: x["t_commit"])

    out = {
        "input": str(args.inp),
        "params": {
            "suffix_p": suffix_p,
            "min_suffix_frames": min_suffix,
            "frames": T,
        },
        "commitments": commits,
        "n_commitments": len(commits),
    }
    commitments_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("[commitments] OK")
    print(f"  commitments: {commitments_path}")
    print(f"  frames: {T}")
    print(f"  commitments#: {len(commits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

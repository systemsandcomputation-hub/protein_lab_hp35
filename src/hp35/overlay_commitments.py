#!/usr/bin/env python3
from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

def find_regime(regimes: List[Dict[str, Any]], t: int) -> str:
    for r in regimes:
        if r["t_start"] <= t <= r["t_end"]:
            return r["label"]
    return "UNKNOWN"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", type=Path, required=True)
    ap.add_argument("--commitments", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    reg = json.loads(args.regimes.read_text(encoding="utf-8"))["regimes"]
    com = json.loads(args.commitments.read_text(encoding="utf-8"))["commitments"]

    out_rows = []
    for c in com:
        t = int(c["t_commit"])
        label = find_regime(reg, t)
        out_rows.append({"t_commit": t, "edge": c["edge"], "regime": label})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": out_rows}, indent=2), encoding="utf-8")

    # print a compact view
    print("[overlay] OK")
    for r in out_rows[:15]:
        print(f'  t={r["t_commit"]:<7} edge={r["edge"]}  -> {r["regime"]}')
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

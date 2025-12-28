#!/usr/bin/env python3
"""
extract_contacts.py

Phase 1 extractor for HP35 / HP36 folding signals.

Reads:
  - hp35.mindists2.bz2 : time series of minimal contact distances (nm)
  - hp35.mindists2.ndx : (optional) index file providing per-column contact identities

Produces:
  - JSONL stream of per-frame contact graphs:
      {"t": <int>, "edges": [[i,j], [k,l], ...]}
    where edges are "contacts" defined by distance < threshold.

Notes / assumptions:
- We do NOT use energies, chemistry, or MSM.
- The .ndx format varies; we attempt to extract residue/contact ids from group names.
  If mapping fails, we fall back to stable integer ids (column indices).

Usage examples:
  python3 -m hp35.extract_contacts \
    --threshold-nm 0.8 \
    --out outputs/processed/hp35_mindists2_contacts_0p8.jsonl

  python3 src/hp35/extract_contacts.py \
    --data-root ~/protein_lab_hp35/data/HP35/HP35-DESRES \
    --threshold-nm 0.8 \
    --max-frames 2000
"""

from __future__ import annotations

import argparse
import bz2
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


# -------------------------
# Helpers: parsing .ndx
# -------------------------

@dataclass(frozen=True)
class ContactLabel:
    """A label for a contact column."""
    a: int
    b: int

    def as_edge(self) -> Tuple[int, int]:
        return (self.a, self.b) if self.a <= self.b else (self.b, self.a)


_NDX_GROUP_RE = re.compile(r"^\s*\[\s*(.*?)\s*\]\s*$")
_INT_RE = re.compile(r"(\d+)")


def parse_ndx_contact_labels(ndx_path: Path, expected_cols: Optional[int] = None) -> Optional[List[ContactLabel]]:
    """
    Parse hp35.mindists2.ndx which (in this dataset) is a simple whitespace-separated
    list of residue index pairs, one per line, in the same order as columns in hp35.mindists2.

    Example:
      3 7
      3 13
      ...

    Lines starting with '#' are ignored.
    Returns a list of ContactLabel with length == number of pairs parsed.
    """
    if not ndx_path.exists():
        return None

    pairs: List[ContactLabel] = []
    try:
        lines = ndx_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        print(f"[extract_contacts] WARN: failed reading ndx file {ndx_path}: {e}", file=sys.stderr)
        return None

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            a = int(parts[0])
            b = int(parts[1])
        except ValueError:
            continue
        pairs.append(ContactLabel(a=a, b=b))

    if not pairs:
        return None

    if expected_cols is not None and len(pairs) != expected_cols:
        print(
            f"[extract_contacts] WARN: ndx pairs ({len(pairs)}) != expected columns ({expected_cols}); "
            "skipping ndx mapping and using column indices instead.",
            file=sys.stderr,
        )
        return None

    return pairs


# -------------------------
# Helpers: reading mindists2.bz2
# -------------------------

def iter_distance_rows_bz2(path: Path) -> Iterator[List[float]]:
    """
    Stream-read a .bz2 text file containing rows of floats.
    Skips comment/meta lines starting with '#', ';', '@'.
    """
    with bz2.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s[0] in ("#", ";", "@"):
                continue
            # Some files might have trailing comments; split on whitespace only
            parts = s.split()
            try:
                row = [float(x) for x in parts]
            except ValueError:
                # If weird tokens exist, try to salvage numeric tokens
                nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
                if not nums:
                    continue
                row = [float(x) for x in nums]
            yield row


# -------------------------
# Contacts extraction
# -------------------------

def distances_to_edges(
    distances: List[float],
    threshold_nm: float,
    labels: Optional[List[ContactLabel]] = None,
    min_label: int = 1,
) -> List[List[int]]:
    """
    Convert a distance row into contact edges for distances < threshold_nm.

    If labels are provided and align with columns, edges become [i,j] from labels.
    Otherwise edges become [col_id, -1] (not ideal) or [col_id, col_id] fallback.

    We prefer stable [i,j] pairs; if labels missing, we use [col_index+min_label, 0].
    """
    edges: List[List[int]] = []
    for c, d in enumerate(distances):
        if d < threshold_nm:
            if labels is not None and c < len(labels):
                a, b = labels[c].as_edge()
                # If label extraction failed for this column, fall back to column id
                if a < 0 or b < 0:
                    edges.append([c + min_label, 0])
                else:
                    edges.append([a, b])
            else:
                edges.append([c + min_label, 0])
    return edges


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# -------------------------
# CLI
# -------------------------

def default_data_root() -> Path:
    # Default assumes script lives at <repo>/src/hp35/extract_contacts.py
    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # .../protein_lab_hp35
    return repo_root / "data" / "HP35" / "HP35-DESRES"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract per-frame contact graphs from hp35.mindists2(.bz2).")
    p.add_argument("--data-root", type=Path, default=default_data_root(),
                   help="Path to HP35-DESRES directory containing hp35.mindists2.bz2 and hp35.mindists2.ndx.")
    p.add_argument("--mindists", type=str, default="hp35.mindists2.bz2",
                   help="Filename of mindists2 bz2 file (relative to --data-root) or absolute path.")
    p.add_argument("--ndx", type=str, default="hp35.mindists2.ndx",
                   help="Filename of .ndx file (relative to --data-root) or absolute path. Optional.")
    p.add_argument("--threshold-nm", type=float, default=0.8,
                   help="Contact threshold in nm (distance < threshold => contact).")
    p.add_argument("--max-frames", type=int, default=0,
                   help="If >0, process only the first N frames.")
    p.add_argument("--skip-frames", type=int, default=0,
                   help="Skip the first N frames (after headers).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSONL path. Default: outputs/processed/hp35_mindists2_contacts_<thr>.jsonl")
    p.add_argument("--summary", type=Path, default=None,
                   help="Optional summary JSON path (counts, columns, frame stats).")
    p.add_argument("--no-ndx", action="store_true",
                   help="Ignore .ndx mapping even if present.")
    return p


def resolve_path(maybe_rel: str, root: Path) -> Path:
    p = Path(maybe_rel)
    return p if p.is_absolute() else (root / p)


def main() -> int:
    args = build_argparser().parse_args()

    data_root: Path = args.data_root
    mindists_path = resolve_path(args.mindists, data_root)
    ndx_path = resolve_path(args.ndx, data_root)

    if not mindists_path.exists():
        print(f"[extract_contacts] ERROR: mindists file not found: {mindists_path}", file=sys.stderr)
        return 2

    # Decide output path
    if args.out is None:
        here = Path(__file__).resolve()
        repo_root = here.parents[2]
        thr = str(args.threshold_nm).replace(".", "p")
        args.out = repo_root / "outputs" / "processed" / f"hp35_mindists2_contacts_{thr}.jsonl"

    ensure_parent(args.out)

    # Peek first data row to determine number of columns
    row_iter = iter_distance_rows_bz2(mindists_path)
    try:
        first_row = next(row_iter)
    except StopIteration:
        print(f"[extract_contacts] ERROR: no numeric data found in {mindists_path}", file=sys.stderr)
        return 3

    n_cols = len(first_row)
    if n_cols == 0:
        print(f"[extract_contacts] ERROR: first row has 0 columns in {mindists_path}", file=sys.stderr)
        return 4

    # Load labels from .ndx if possible
    labels: Optional[List[ContactLabel]] = None
    if not args.no_ndx and ndx_path.exists():
        labels = parse_ndx_contact_labels(ndx_path, expected_cols=n_cols)

    # Reconstruct iterator including the first row
    def all_rows() -> Iterator[List[float]]:
        yield first_row
        for r in row_iter:
            yield r

    # Process frames streaming -> JSONL
    frame_idx = 0
    written = 0
    total_edges = 0
    max_edges = 0
    min_edges = 10**18

    with args.out.open("w", encoding="utf-8") as out_f:
        for row in all_rows():
            if args.skip_frames and frame_idx < args.skip_frames:
                frame_idx += 1
                continue

            # Safety: skip malformed rows
            if len(row) != n_cols:
                # try to ignore, but warn occasionally
                if written < 3:
                    print(
                        f"[extract_contacts] WARN: row {frame_idx} has {len(row)} cols, expected {n_cols}; skipping",
                        file=sys.stderr,
                    )
                frame_idx += 1
                continue

            edges = distances_to_edges(row, threshold_nm=args.threshold_nm, labels=labels, min_label=1)
            n_edges = len(edges)
            total_edges += n_edges
            max_edges = max(max_edges, n_edges)
            min_edges = min(min_edges, n_edges)

            rec = {"t": frame_idx, "edges": edges}
            out_f.write(json.dumps(rec, separators=(",", ":")) + "\n")

            written += 1
            frame_idx += 1

            if args.max_frames and written >= args.max_frames:
                break

    if written == 0:
        print("[extract_contacts] ERROR: wrote 0 frames (check --skip-frames / file format).", file=sys.stderr)
        return 5

    avg_edges = total_edges / written
    if min_edges == 10**18:
        min_edges = 0

    # Optional summary
    if args.summary is not None:
        ensure_parent(args.summary)
        summary = {
            "mindists_path": str(mindists_path),
            "ndx_path": str(ndx_path) if ndx_path.exists() else None,
            "used_ndx_mapping": labels is not None,
            "threshold_nm": args.threshold_nm,
            "n_cols": n_cols,
            "frames_written": written,
            "avg_edges_per_frame": avg_edges,
            "min_edges_per_frame": min_edges,
            "max_edges_per_frame": max_edges,
            "out_jsonl": str(args.out),
        }
        args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[extract_contacts] OK")
    print(f"  input:  {mindists_path}")
    print(f"  output: {args.out}")
    print(f"  cols:   {n_cols}")
    print(f"  frames: {written}")
    print(f"  edges/frame: avg={avg_edges:.2f} min={min_edges} max={max_edges}")
    print(f"  ndx mapping: {'YES' if labels is not None else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

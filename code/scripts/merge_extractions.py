"""Merge extraction outputs (e.g. a main run + a retry of its failed files) into a NEW prefix; never overwrites the inputs.

  python code/claude/merge_extractions.py <out_prefix> <prefix_a> <prefix_b> [...]

Reads data/features/claude/<prefix>_features.npz + _index.csv (+ _seg_index.csv / _seg_arrays.npz when present), concatenates in the
given order, drops duplicate annotation rows (first occurrence wins) and writes <out_prefix>_* with the same layout.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FEAT, log_change  # noqa: E402

OUT = FEAT / "claude"
out_prefix, parts = sys.argv[1], sys.argv[2:]
assert out_prefix not in parts, "output prefix must differ from the inputs"

idx = pd.concat([pd.read_csv(OUT / f"{p}_index.csv") for p in parts], ignore_index=True)
dup = idx.ann_row.duplicated().values
idx[~dup].reset_index(drop=True).to_csv(OUT / f"{out_prefix}_index.csv", index=False)
Fs = [np.load(OUT / f"{p}_features.npz") for p in parts if (OUT / f"{p}_features.npz").exists()]
if Fs and len(Fs) == len(parts):  # file-level features only when EVERY part has them (a segment-only pass has none)
    keys =[k for k in Fs[0].files if all(k in f.files for f in Fs)]
    np.savez_compressed(OUT / f"{out_prefix}_features.npz", **{k: np.concatenate([f[k] for f in Fs])[~dup] for k in keys})

sparts = [p for p in parts if (OUT / f"{p}_seg_index.csv").exists()]
if sparts:
    sis = [pd.read_csv(OUT / f"{p}_seg_index.csv").assign(_src=p) for p in sparts]
    sa = [np.load(OUT / f"{p}_seg_arrays.npz") for p in sparts]
    si = pd.concat(sis, ignore_index=True)
    first_part = {}
    for p, s in zip(sparts, sis):  # a segment is kept from the first prefix that contains its annotation row
        for ar in s.ann_row.unique():
            first_part.setdefault(ar, p)
    keep = np.array([first_part[ar] == s for ar, s in zip(si.ann_row.values, si._src.values)])
    si.drop(columns="_src")[keep].reset_index(drop=True).to_csv(OUT / f"{out_prefix}_seg_index.csv", index=False)
    np.savez_compressed(OUT / f"{out_prefix}_seg_arrays.npz", **{k: np.concatenate([a[k] for a in sa])[keep] for k in ("mfcc19", "shape")})
log_change(f"merge_extractions: {out_prefix} <- {', '.join(parts)}; {int((~dup).sum())} annotation rows")
print(f"{out_prefix}: {int((~dup).sum())} annotation rows from {len(parts)} parts")

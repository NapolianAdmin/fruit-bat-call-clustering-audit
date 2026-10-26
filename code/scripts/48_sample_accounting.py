"""Step 7 (supplement) - exact accounting of the data partitions: which annotated files belong to the discovery sample, the two out-of-sample sets, and the reproduction-only extraction,
with their sizes, Isolation files, positive-coded files, bats, channels, months, and the overlap of bats / channels / dates between each held-out set and the discovery sample.
Partitions are defined by FileID membership of the extraction indices in data/features/claude (all2_index = discovery, repl_index, rest_index, hrowfull_index minus discovery).
Writes results/step7_sample_accounting.csv and results/step7_sample_overlap.csv.   Run: python code/claude/48_sample_accounting.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FEAT, RES  # noqa: E402

OF = FEAT / "claude"
ann = pd.read_csv(FEAT / "annotations_merged.csv.gz")
ann["ann_row"] = np.arange(len(ann))
idx = {k: pd.read_csv(OF / f"{k}_index.csv", usecols=["FileID", "ann_row"]).drop_duplicates("FileID") for k in ("all2", "repl", "rest", "hrowfull")}
disc = set(idx["all2"].FileID)
idx["hrowfull"] = idx["hrowfull"][~idx["hrowfull"].FileID.isin(disc)]
parts = {"discovery": idx["all2"], "replication": idx["repl"], "completion": idx["rest"], "reproduction only": idx["hrowfull"]}
fs = {k: set(v.FileID) for k, v in parts.items()}
assert not (fs["discovery"] & fs["replication"]) and not (fs["discovery"] & fs["completion"]) and not (fs["replication"] & fs["completion"])
assert len(set().union(*fs.values())) == ann.FileID.nunique()  # the four partitions together are every annotated file
rows, bats, chans, dates, frames = [], {}, {}, {}, []
for k, v in parts.items():
    d = ann[ann.ann_row.isin(v.ann_row)]  # the annotation row each sample actually used (one per file)
    frames.append(d)
    pos = d[d.Emitter > 0]
    bats[k], chans[k], dates[k] = set(pos.Emitter), set(d.channel), set(d.date.astype(str))
    rows.append(dict(partition=k, files=len(d), isolation_files=int((d.context_name == "Isolation").sum()), positive_coded_files=len(pos), positive_coded_bats=pos.Emitter.nunique(),
                     channels=d.channel.nunique(), recording_days=d.date.nunique()))
al = pd.concat(frames)
rows.append(dict(partition="all annotated files", files=len(al), isolation_files=int((al.context_name == "Isolation").sum()), positive_coded_files=int((al.Emitter > 0).sum()),
                 positive_coded_bats=al[al.Emitter > 0].Emitter.nunique(), channels=al.channel.nunique(), recording_days=al.date.nunique()))
A = pd.DataFrame(rows)
A.to_csv(RES / "step7_sample_accounting.csv", index=False)
ov = []
for k in ("replication", "completion"):
    ov.append(dict(held_out=k, bats_in_held_out=len(bats[k]), bats_also_in_discovery=len(bats[k] & bats["discovery"]), channels_in_held_out=len(chans[k]), channels_also_in_discovery=len(chans[k] & chans["discovery"]),
                   recording_days_in_held_out=len(dates[k]), recording_days_also_in_discovery=len(dates[k] & dates["discovery"])))
O = pd.DataFrame(ov)
O.to_csv(RES / "step7_sample_overlap.csv", index=False)
pd.set_option("display.width", 220)
print(A.to_string())
print(O.to_string())

"""Step 7a - a REPLICATION sample: annotated files that were never extracted in any earlier step.

The Isolation finding came out of an exploratory UMAP + HDBSCAN run on the 41,116-file discovery sample.  To test it out of sample, draw N (default 12,000) annotated files
(one annotation row per FileID, first row) uniformly at random from the pool of FileIDs that appear in NO `*_index.csv` under data/features/claude (baseline, main, main2, hrow, segold, ... AVES).
Uniform sampling keeps the corpus's natural context / channel mix (no stratification, no floors).
Writes data/features/claude/repl_manifest.csv (ann_row, FileID; format of the other manifests) and results/step7_replication_sample.csv (context / channel mix vs the discovery sample).
Run: python code/claude/32_replication_sample.py [N=12000] [prefix=repl]   (prefix 'rest' with a large N draws ALL annotated files not extracted so far)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FEAT, RES, SEED, log_change  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
PFX = sys.argv[2] if len(sys.argv) > 2 else "repl"
OUT = FEAT / "claude"
ann = pd.read_csv(FEAT / "annotations_merged.csv.gz")
ann["ann_row"] = np.arange(len(ann))
used = set()
for f in OUT.glob("*_index.csv"):
    try:
        used |= set(pd.read_csv(f, usecols=["FileID"]).FileID.tolist())
    except Exception as e:  # an index without FileID is not a file list
        print("skipped", f.name, repr(e)[:60])
files = ann.drop_duplicates("FileID")
pool = files[~files.FileID.isin(used)]
print(f"annotated files {len(files)}, already extracted {len(used)}, pool {len(pool)}", flush=True)
rg = np.random.default_rng(SEED + 777)
pick = pool.iloc[np.sort(rg.choice(len(pool), min(N, len(pool)), replace=False))]
pick[["ann_row", "FileID"]].to_csv(OUT / f"{PFX}_manifest.csv", index=False)
disc = files[files.FileID.isin(used)]
mix = pd.DataFrame({"replication_share": pick.context_name.value_counts(normalize=True), "discovery_share": disc.context_name.value_counts(normalize=True)}).fillna(0)
mix.to_csv(RES / ("step7_replication_sample.csv" if PFX == "repl" else f"step7_{PFX}_sample.csv"))
print(f"replication sample {len(pick)} files; Isolation files {(pick.context_name == 'Isolation').sum()} ({(pick.context_name == 'Isolation').mean():.3f}); positive-emitter share {(pick.Emitter > 0).mean():.3f}")
print(mix.round(3).to_string())
log_change(f"Step 7a replication sample: {len(pick)} never-extracted annotated files (pool {len(pool)}), manifest data/features/claude/{PFX}_manifest.csv")

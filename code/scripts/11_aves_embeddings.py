"""Step 5 - AVES (Animal Vocalization Encoder, Earth Species Project; HuBERT-style, torchaudio port) embeddings on a subset.

STATED LIMITATION: AVES takes 16 kHz audio. Bat calls here are recorded at 250 kHz and extend to ~100 kHz, so a plain downsample to
16 kHz keeps only the < 8 kHz part of each call. Three variants are therefore computed per file:
  aves_ds16_naive   annotation span, resampled 250 kHz -> 16 kHz (the brief's variant)
  aves_ds16_voiced  voiced samples of the span concatenated (10 ms gaps), resampled to 16 kHz
  aves_te_voiced    voiced samples concatenated and RELABELLED as 16 kHz ("time expansion", x15.6 slower, pitch lowered 15.6x):
                    keeps the ultrasonic content in AVES's audible range; NOT a model-intended input - an exploratory contrast
Embedding = mean over frames of the last transformer layer (768 dims).
Weights: https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio.torchaudio.pt (+ model_config.json), loaded with
torch.load(..., weights_only=True) so no pickled code is executed.
Subset: 1,500 files from sample_all_manifest.csv, >= 20 per channel, seed 42; audio read by HTTP range request, not kept.
Run: python code/claude/11_aves_embeddings.py [n_files=1500]
"""
import importlib.util
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FEAT, LOGS, META, RES, SEED, log_change  # noqa: E402

_spec = importlib.util.spec_from_file_location("ext", Path(__file__).with_name("02_extract_features.py"))
ext = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ext)

OUT = FEAT / "claude"
AV = META.parent / "aves"
BASE = "https://storage.googleapis.com/esp-public-files/ported_aves/"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500


def get(name):
    AV.mkdir(parents=True, exist_ok=True)
    p = AV / name
    if not p.exists():
        r = requests.get(BASE + name, timeout=300)
        r.raise_for_status()
        p.write_bytes(r.content)
        with open(LOGS / "claude_download_ledger.csv", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M')},esp-public-files {name},{len(r.content)},AVES weights\n")
    return p


import librosa  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from torchaudio.models import wav2vec2_model  # noqa: E402

MODEL = sys.argv[2] if len(sys.argv) > 2 else "aves-base-bio"  # or aves-base-all
OUTP = "aves2" + MODEL.split("-")[-1]  # e.g. aves2bio / aves2all
cfg = json.load(open(get(f"{MODEL}.torchaudio.model_config.json")))
model = wav2vec2_model(**cfg, aux_num_out=None)
model.load_state_dict(torch.load(get(f"{MODEL}.torchaudio.pt"), map_location="cpu", weights_only=True))
model.eval()
torch.set_num_threads(4)


def open_zip(z):
    for attempt in range(8):
        try:
            zf = zipfile.ZipFile(ext.RangeFile(z.url, int(z.size_bytes)))
            return zf, {i.filename: i for i in zf.infolist()}
        except Exception:
            time.sleep(2 ** min(attempt, 5))
    raise RuntimeError("could not open archive")


def embed(wave):
    with torch.no_grad():
        feats = model.extract_features(torch.from_numpy(wave.astype(np.float32))[None])[0]
    return feats[-1][0].mean(0).numpy()


S = pd.read_csv(OUT / "sample_all_manifest.csv")
rg = np.random.default_rng(SEED)
pick = set()
for ch, g in S.groupby("channel"):
    pick |= set(rg.choice(g.ann_row.values, min(20, len(g)), replace=False))
rest = S[~S.ann_row.isin(pick)]
pick |= set(rg.choice(rest.ann_row.values, N - len(pick), replace=False))
ann = pd.read_csv(FEAT / "annotations_merged.csv.gz")
ann["ann_row"] = np.arange(len(ann))
fi = pd.read_csv(FEAT / "fileinfo_parsed.csv.gz", usecols=["FileID", "file_name"])
rows = ann[ann.ann_row.isin(pick)].merge(fi, on="FileID")
segs = ext.load_segments(rows.FileID.unique())
lst = pd.read_csv(META.parent / "figshare_file_listing.csv").set_index("name")

E = {"aves_ds16_naive": [], "aves_ds16_voiced": [], "aves_te2_naive": [], "aves_te2_voiced": []}  # te2 = resample to 32 kHz, read as 16 kHz (2x slower, content 0-16 kHz -> 0-8 kHz)
IDX, t0 = [], time.time()
for folder, g in rows.groupby("folder"):
    z = lst.loc[f"{folder}.zip"]
    zf, names = open_zip(z)
    for _, r in g.iterrows():
        data = None
        for attempt in range(4):  # network resets happen: re-open the archive and retry
            try:
                data = zf.read(names[r.file_name])
                break
            except Exception:
                time.sleep(2 ** attempt)
                zf, names = open_zip(z)
        if data is None:
            print(f"skipped {r.file_name} after retries", flush=True)
            continue
        y, sr = sf.read(io.BytesIO(data), dtype="float32")
        s0, e0 = max(1, int(r.Start)), min(int(r.End), len(y))
        ys = y[s0 - 1:e0]
        sg = segs[int(r.FileID)]
        sg = sg[(sg[:, 1] >= s0) & (sg[:, 0] <= e0)]
        gap = np.zeros(int(0.010 * sr), np.float32)
        parts = [x for a, b in sg for x in (ys[max(a, s0) - s0:min(b, e0) - s0 + 1], gap)]
        yv = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        if len(yv) < 2000:
            continue
        E["aves_ds16_naive"].append(embed(librosa.resample(ys, orig_sr=sr, target_sr=16000)))
        E["aves_ds16_voiced"].append(embed(librosa.resample(yv, orig_sr=sr, target_sr=16000)))
        E["aves_te2_naive"].append(embed(librosa.resample(ys[: int(2.5 * sr)], orig_sr=sr, target_sr=32000)))  # <= 2.5 s source -> 5 s model input
        E["aves_te2_voiced"].append(embed(librosa.resample(yv[: int(1.2 * sr)], orig_sr=sr, target_sr=32000)))  # <= 1.2 s voiced source -> 2.4 s model input
        IDX.append(dict(ann_row=int(r.ann_row), FileID=int(r.FileID)))
        if len(IDX) % 50 == 0:
            print(f"{len(IDX)} files, {time.time()-t0:.0f}s", flush=True)
            np.savez_compressed(OUT / f"{OUTP}_features_partial.npz", **{k: np.stack(v) for k, v in E.items()})  # checkpoint
            pd.DataFrame(IDX).to_csv(OUT / f"{OUTP}_index_partial.csv", index=False)
idx = pd.DataFrame(IDX)
scal = pd.concat([pd.read_csv(OUT / f"{p}_index.csv") for p in ("baseline", "main") if (OUT / f"{p}_index.csv").exists()]).drop_duplicates("ann_row")
idx = idx.merge(scal.drop(columns=["FileID"]), on="ann_row", how="left")
idx["too_short"] = idx.too_short.fillna(0).astype(int)
idx.to_csv(OUT / f"{OUTP}_index.csv", index=False)
np.savez_compressed(OUT / f"{OUTP}_features.npz", **{k: np.stack(v) for k, v in E.items()})
pd.DataFrame([dict(model=MODEL, files=len(idx), wall_s=time.time() - t0, s_per_file=(time.time() - t0) / max(1, len(idx)), torch=torch.__version__)]).to_csv(RES / f"step5_{OUTP}_timing.csv", index=False)
log_change(f"Step 5 AVES embeddings: {len(idx)} files, {time.time()-t0:.0f}s; variants ds16_naive, ds16_voiced, te_voiced (state: 16 kHz downsample loses > 8 kHz)")
print(f"done {len(idx)} files in {time.time()-t0:.0f}s")

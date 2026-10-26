"""Step 2d - Dataset B (Elie et al. 2024; hearing vs deaf Egyptian fruit bats; ambient mic AND throat piezo per vocal event).

Dataset B has NO FileInfo voiced segments, so voiced frames come from a documented energy detector:
  frame energy = 10*log10(sum of mel-band power) per STFT frame (same STFT as dataset A: n_fft 2048, hop 512, mel 128, 2 kHz-fmax)
  floor        = median energy of the lowest 20% of frames
  voiced       = frames with energy > floor + THR dB, gaps shorter than 10 ms closed, segments shorter than 5 ms dropped
THR is chosen on Dataset A by frame-level F1 against the authors' FileInfo segments (300 random baseline files).

  python code/claude/02b_datasetB.py validate     -> results/stepB_detector_validation.csv, results/stepB_detector_choice.json
  python code/claude/02b_datasetB.py extract      -> downloads VocDataBase.zip (1.8 GB, md5-checked) to data/raw/datasetB/, extracts features
                                                     for all 29,688 WAVs (B_features.npz, B_index.csv), keeps the zip until you delete it.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import hashlib
import importlib.util
import io
import json
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
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
ZIP = META.parent / "datasetB" / "VocDataBase.zip"
HOP, NFFT, NMEL, FMIN = 512, 2048, 128, 2000
_ZF = None  # per-process cached ZipFile


def runs(mask):
    """(start, end_exclusive) of True runs."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def stft_params(sr):
    """Frame ~8-10 ms like dataset A: n_fft 2048 / hop 512 for sr >= 100 kHz (A: 250 kHz, B mic: 192 kHz), 512 / 128 for B's 50 kHz piezo."""
    return (2048, 512) if sr >= 100_000 else (512, 128)


def delta_safe(X, order=1):
    """librosa delta that survives very short signals (width shrinks to the frame count, mode 'nearest')."""
    import librosa
    T = X.shape[1]
    w = min(9, T if T % 2 == 1 else T - 1)
    return np.zeros_like(X) if w < 3 else librosa.feature.delta(X, order=order, width=w, mode="nearest")


def mfcc_safe(L, mask):
    import librosa
    M = librosa.feature.mfcc(S=L, n_mfcc=20)
    parts = [M, delta_safe(M), delta_safe(M, 2)]
    v = np.concatenate([np.concatenate([p[:, mask].mean(1), p[:, mask].std(1)]) for p in parts])
    v[~np.isfinite(v)] = 0.0
    return v.astype(np.float32)


def detect_voiced(Sp, sr, thr_db):
    e = 10 * np.log10(Sp.sum(0) + 1e-20)
    floor = np.median(np.sort(e)[: max(1, int(0.2 * len(e)))])
    v = e > floor + thr_db
    fr_ms = stft_params(sr)[1] / sr * 1000
    gap, mn = int(np.ceil(10 / fr_ms)), int(np.ceil(5 / fr_ms))
    for s, t in runs(~v):  # close short gaps (not at the edges)
        if t - s < gap and s > 0 and t < len(v):
            v[s:t] = True
    for s, t in runs(v):  # drop short segments
        if t - s < mn:
            v[s:t] = False
    return v


def mel_power(y, sr):
    import librosa
    nfft, hop = stft_params(sr)
    return librosa.feature.melspectrogram(y=y, sr=sr, n_fft=nfft, hop_length=hop, n_mels=NMEL, fmin=FMIN, fmax=min(125_000, sr / 2 - 1), power=2.0)


def process_file(args):
    global _ZF
    import librosa
    import soundfile as sf
    name, path, thr = args
    try:
        if _ZF is None:
            _ZF = zipfile.ZipFile(ZIP)
        y, sr = sf.read(io.BytesIO(_ZF.read(path)), dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        n = len(y)
        Sp = mel_power(y, sr)
        vm = detect_voiced(Sp, sr, thr)
        T = Sp.shape[1]
        vs = np.repeat(vm, stft_params(sr)[1])[:n]
        vs = np.r_[vs, np.zeros(n - len(vs), bool)]
        yc = y - y.mean()
        sc = dict(name=name, header_sr=int(sr), file_len_s=n / sr, n_frames=T, n_voiced_frames=int(vm.sum()), n_voiced_seg=len(runs(vm)),
                  voiced_dur_s=vs.sum() / sr, silence_fraction=1 - vs.mean(), too_short=int(vm.sum() < 3))
        if (~vs).sum() >= 1000 and vs.sum() > 0:
            pv, pu = np.mean(yc[vs] ** 2), np.mean(yc[~vs] ** 2)
            sc.update(snr_db=10 * np.log10(pv / max(pu, 1e-20)), noise_floor_dbfs=10 * np.log10(max(pu, 1e-20)))
        else:
            sc.update(snr_db=np.nan, noise_floor_dbfs=np.nan)
        sc["clip_frac"] = float((np.abs(y[vs]) >= ext.CLIP).mean()) if vs.any() else np.nan
        L = librosa.power_to_db(Sp)
        D1, D2 = delta_safe(L), delta_safe(L, 2)
        allm = np.ones(T, bool)
        ft = dict(naive_ref1=ext.band_stats(L, D1, D2, allm), mfcc_naive=mfcc_safe(L, allm))
        ft["voiced_ref1"] = ext.band_stats(L, D1, D2, vm) if vm.sum() >= 3 else np.full(1408, np.nan, np.float32)
        ft["mfcc_voiced"] = mfcc_safe(L, vm) if vm.sum() >= 3 else np.full(120, np.nan, np.float32)
        return sc, ft, None
    except Exception as e:
        return None, None, (name, repr(e)[:200])


def validate():
    import soundfile as sf
    ann = pd.read_csv(FEAT / "annotations_merged.csv.gz")
    ann["ann_row"] = np.arange(len(ann))
    fi = pd.read_csv(FEAT / "fileinfo_parsed.csv.gz", usecols=["FileID", "file_name"])
    base = pd.read_csv(FEAT / "claude" / "baseline_manifest.csv")
    rows = ann[ann.ann_row.isin(base.ann_row)].merge(fi, on="FileID").sample(300, random_state=SEED)
    segs = ext.load_segments(rows.FileID.unique())
    lst = pd.read_csv(META.parent / "figshare_file_listing.csv").set_index("name")
    res = []
    thrs = (10, 15, 20, 25, 30, 35, 40)
    t0 = time.time()
    for folder, g in rows.groupby("folder"):
        z = lst.loc[f"{folder}.zip"]
        zf = zipfile.ZipFile(ext.RangeFile(z.url, int(z.size_bytes)))
        names = {i.filename: i for i in zf.infolist()}
        for _, r in g.iterrows():
            y, sr = sf.read(io.BytesIO(zf.read(names[r.file_name])), dtype="float32")
            s0, e0 = int(r.Start), min(int(r.End), len(y))
            ys = y[s0 - 1:e0]
            Sp = mel_power(ys, sr)
            T = Sp.shape[1]
            sg = segs[r.FileID]
            sg = sg[(sg[:, 1] >= s0) & (sg[:, 0] <= e0)]
            vs = np.zeros(len(ys), bool)
            for a, b in sg:
                vs[max(a, s0) - s0:min(b, e0) - s0 + 1] = True
            ref = vs[np.minimum(np.arange(T) * HOP, len(ys) - 1)]
            for thr in thrs:
                det = detect_voiced(Sp, sr, thr)
                tp = (det & ref).sum()
                res.append(dict(FileID=int(r.FileID), thr_db=thr, f1=2 * tp / max(1, det.sum() + ref.sum()), iou=tp / max(1, (det | ref).sum()),
                                ref_voiced_frac=ref.mean(), det_voiced_frac=det.mean()))
    D = pd.DataFrame(res)
    D.to_csv(RES / "stepB_detector_validation.csv", index=False)
    S = D.groupby("thr_db").agg(f1_mean=("f1", "mean"), f1_median=("f1", "median"), iou_mean=("iou", "mean"),
                                ref_frac=("ref_voiced_frac", "mean"), det_frac=("det_voiced_frac", "mean")).reset_index()
    best = int(S.loc[S.f1_mean.idxmax(), "thr_db"])
    (RES / "stepB_detector_choice.json").write_text(json.dumps(dict(chosen_thr_db=best, n_files=300, table=S.round(4).to_dict("records"), seconds=time.time() - t0), indent=2))
    print(S.round(3).to_string(), "\nchosen THR =", best)


def download():
    lst = pd.read_csv(META.parent / "figshare_file_listing.csv").set_index("name").loc["VocDataBase.zip"]
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    if ZIP.exists() and ZIP.stat().st_size == int(lst.size_bytes):
        return
    with requests.get(lst.url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(ZIP, "wb") as f:
            for ch in r.iter_content(1 << 20):
                f.write(ch)
    h = hashlib.md5()
    with open(ZIP, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""):
            h.update(ch)
    ok = h.hexdigest() == lst.md5.lower()
    with open(LOGS / "claude_download_ledger.csv", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')},figshare VocDataBase.zip,{ZIP.stat().st_size},Dataset B (md5 ok={ok})\n")
    if not ok:
        raise RuntimeError("md5 mismatch for VocDataBase.zip")


def thrtest():
    """Share of Dataset-B files with >= 3 voiced frames, by detector threshold and sensor (600 random files)."""
    inv = pd.read_csv(RES / "stepB_file_inventory.csv").sample(600, random_state=SEED)
    rows = []
    for thr in (6, 8, 10, 12, 15, 20):
        for _, r in inv.iterrows():
            sc, ft, err = process_file((r["name"], r.path, thr))
            if sc:
                rows.append(dict(thr_db=thr, source=r.source, valid=int(sc["too_short"] == 0), voiced_frac=1 - sc["silence_fraction"], sr=sc["header_sr"]))
    T = pd.DataFrame(rows).groupby(["thr_db", "source"]).agg(share_ge3_voiced_frames=("valid", "mean"), mean_voiced_frac=("voiced_frac", "mean"), n=("valid", "size")).reset_index()
    T.to_csv(RES / "stepB_threshold_test.csv", index=False)
    print(T.round(3).to_string())


def extract():
    thr = int(sys.argv[2]) if len(sys.argv) > 2 else json.loads((RES / "stepB_detector_choice.json").read_text())["chosen_thr_db"]
    download()
    inv = pd.read_csv(RES / "stepB_file_inventory.csv", parse_dates=["date"])
    t0 = time.time()
    R, F = [], []
    jobs = [(nm, p, thr) for p, nm in zip(inv.path, inv.name)]
    with ProcessPoolExecutor(max_workers=6) as ex:
        for i, (sc, ft, err) in enumerate(ex.map(process_file, jobs, chunksize=16), 1):
            if err:
                F.append(err)
            else:
                R.append((sc, ft))
            if i % 2000 == 0:
                print(f"{i}/{len(inv)} files, {len(F)} failed, {time.time()-t0:.0f}s", flush=True)
    idx = pd.DataFrame([r[0] for r in R]).merge(inv, on="name", how="left")
    idx["event_id"] = idx.individual + "_" + idx.date.dt.strftime("%y%m%d") + "_" + idx.hhmm.astype(str).str.zfill(4) + "_" + idx.voc_num.astype(str)
    idx.to_csv(OUT / "B_index.csv", index=False)
    np.savez_compressed(OUT / "B_features.npz", **{k: np.stack([r[1][k] for r in R]) for k in R[0][1]})
    pd.DataFrame(F, columns=["name", "error"]).to_csv(OUT / "B_failed.csv", index=False)
    pd.DataFrame([dict(files=len(R), failed=len(F), wall_s=time.time() - t0, files_per_s=len(R) / (time.time() - t0), thr_db=thr)]).to_csv(RES / "step2_timing_B.csv", index=False)
    log_change(f"Dataset B extraction: {len(R)} files, {len(F)} failed, {time.time()-t0:.0f}s, detector THR={thr} dB")
    print(f"done: {len(R)} files, {len(F)} failed, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    {"validate": validate, "thrtest": thrtest, "extract": extract}[sys.argv[1]]()

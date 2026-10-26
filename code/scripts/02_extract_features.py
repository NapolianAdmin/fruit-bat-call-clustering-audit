"""Step 2 - feature extraction straight from the remote WAV archives (HTTP range reads; NO audio kept on disk).

For every annotation row in a manifest it computes, from the annotation span of the file:
  naive_*   all frames of the span (what the draft did, up to the unknown dB reference)   1,408 stats
  voiced_*  only frames whose centre sample lies inside a FileInfo voiced segment          1,408 stats
  mfcc_*    20 MFCC + delta + delta2, mean and std (120) over naive / voiced frames
  scalars   silence_fraction, snr_db, noise_floor_dbfs, clip_frac, clip_flag, n_voiced_frames, voiced_dur_s, header sr ...
dB variants (the draft does not say which): ref1 = power_to_db(S); refmax = power_to_db(S, ref=np.max); notop = power_to_db(S, top_db=None).
1,408 layout (per mel band, 128 each):
  [0:128] mean L, [128:256] std L, [256:384] mean D1, [384:512] std D1, [512:640] mean D2, [640:768] std D2,
  [768:896] skew L, [896:1024] kurtosis L, [1024:1152] min L, [1152:1280] max L, [1280:1408] range L   (non-finite -> 0)
Voice sample indices in FileInfo are treated as 1-based inclusive (checked against the WAV length: see 'voice_oob').

Run:  python code/claude/02_extract_features.py <manifest.csv> <out_prefix> [all|ref1]
      manifest.csv needs a column 'ann_row' (0-based row of Annotations.csv). Output: data/features/claude/<out_prefix>_*
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import csv
import io
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

OUT = FEAT / "claude"
SR, NFFT, HOP, NMEL, FMIN, FMAX = 250_000, 2048, 512, 128, 2000, 125_000
CLIP = 32767 / 32768
LISTING = META.parent / "figshare_file_listing.csv"


class RangeFile(io.RawIOBase):
    """Read-only seekable file over HTTP range requests; re-resolves the pre-signed URL when it expires."""

    def __init__(self, url, size):
        self.url, self.size, self.pos, self.n, self.bytes = url, size, 0, 0, 0
        self.ses = requests.Session()
        self._resolve()

    def _resolve(self):
        self.final = self.ses.head(self.url, allow_redirects=True, timeout=60).url

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        for attempt in range(6):
            try:
                r = self.ses.get(self.final, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"}, timeout=120)
                if r.status_code in (400, 401, 403):
                    self._resolve()
                    continue
                r.raise_for_status()
                self.n += 1
                self.bytes += len(r.content)
                self.pos += len(r.content)
                return r.content
            except requests.RequestException:
                time.sleep(1.5 ** attempt)
                self._resolve()
        raise IOError("range read failed after retries")

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def band_stats(L, D1, D2, mask):
    from scipy.stats import kurtosis, skew
    L, D1, D2 = L[:, mask], D1[:, mask], D2[:, mask]
    v = np.concatenate([L.mean(1), L.std(1), D1.mean(1), D1.std(1), D2.mean(1), D2.std(1),
                        skew(L, axis=1), kurtosis(L, axis=1), L.min(1), L.max(1), L.max(1) - L.min(1)])
    v[~np.isfinite(v)] = 0.0
    return v.astype(np.float32)


def mfcc_stats(Ldb, mask):
    import librosa
    M = librosa.feature.mfcc(S=Ldb, n_mfcc=20)
    parts = [M, librosa.feature.delta(M), librosa.feature.delta(M, order=2)]
    v = np.concatenate([np.concatenate([p[:, mask].mean(1), p[:, mask].std(1)]) for p in parts])
    v[~np.isfinite(v)] = 0.0
    return v.astype(np.float32)


def one_row(y, start, end, segs, variants, seg_mode=""):
    """y: whole file float32 in [-1,1]; start/end: 1-based inclusive annotation span; segs: (n,2) 1-based inclusive."""
    import librosa
    n_file = len(y)
    s0, e0 = max(1, start), min(end, n_file)
    ys = y[s0 - 1:e0].astype(np.float32)
    n = len(ys)
    sg = segs[(segs[:, 1] >= s0) & (segs[:, 0] <= e0)]
    sg = np.stack([np.maximum(sg[:, 0], s0), np.minimum(sg[:, 1], e0)], 1) - s0  # 0-based inclusive, span-relative
    vs = np.zeros(n, bool)
    for a, b in sg:
        vs[a:b + 1] = True
    out = dict(span_len_s=n / SR, file_len_s=n_file / SR, n_voiced_seg=len(sg), voiced_dur_s=vs.sum() / SR,
               silence_fraction=1 - vs.mean(), voice_oob=int((segs[:, 1] > n_file).any()))
    yc = ys - ys.mean()
    if (~vs).sum() >= 1000 and vs.sum() > 0:
        pv, pu = np.mean(yc[vs] ** 2), np.mean(yc[~vs] ** 2)
        out.update(snr_db=10 * np.log10(pv / max(pu, 1e-20)), noise_floor_dbfs=10 * np.log10(max(pu, 1e-20)))
    else:
        out.update(snr_db=np.nan, noise_floor_dbfs=np.nan)
    out["clip_frac"] = float((np.abs(ys[vs]) >= CLIP).mean()) if vs.any() else np.nan
    out["clip_flag"] = int(out["clip_frac"] > 0.001) if vs.any() else 0
    S = librosa.feature.melspectrogram(y=ys, sr=SR, n_fft=NFFT, hop_length=HOP, n_mels=NMEL, fmin=FMIN, fmax=FMAX, power=2.0)
    T = S.shape[1]
    centres = np.arange(T) * HOP  # librosa centre=True: frame t is centred on sample t*HOP
    vm = vs[np.minimum(centres, n - 1)]
    out["n_frames"], out["n_voiced_frames"] = T, int(vm.sum())
    allm = np.ones(T, bool)
    feats = {}
    if seg_mode != "segonly":
        for var in variants:
            L = librosa.power_to_db(S) if var == "ref1" else librosa.power_to_db(S, ref=np.max) if var == "refmax" else librosa.power_to_db(S, top_db=None)
            D1, D2 = librosa.feature.delta(L), librosa.feature.delta(L, order=2)  # deltas on the FULL span, then masked
            feats[f"naive_{var}"] = band_stats(L, D1, D2, allm)
            feats[f"voiced_{var}"] = band_stats(L, D1, D2, vm) if vm.sum() >= 3 else np.full(1408, np.nan, np.float32)
            if var == "ref1":
                feats["mfcc_naive"] = mfcc_stats(L, allm)
                feats["mfcc_voiced"] = mfcc_stats(L, vm) if vm.sum() >= 3 else np.full(120, np.nan, np.float32)
    out["too_short"] = int(vm.sum() < 3)
    if seg_mode in ("seg", "segonly"):
        noise_pow = float(S[:, ~vm].sum(0).mean()) if (~vm).any() else np.nan
        feats["__seg__"] = segment_records(S, librosa.power_to_db(S), sg, s0, noise_pow, n)
    return out, feats


_MELHZ = None


def segment_records(S, L, sg, s0, noise_pow, n):
    """Per-voiced-segment features (>= 3 frames). Returns list of (scalar dict, mfcc1-19 mean, 16x32 level-free shape).
    shape = log-mel pooled 128->32 bands, mean level removed, time-resampled to 16 bins (duration-normalised)."""
    import librosa
    global _MELHZ
    if _MELHZ is None:
        _MELHZ = librosa.mel_frequencies(n_mels=NMEL + 2, fmin=FMIN, fmax=FMAX)[1:-1]
    hz = _MELHZ
    M = librosa.feature.mfcc(S=L, n_mfcc=20)
    T = S.shape[1]
    out = []
    for k, (a, b) in enumerate(sg):
        f0, f1 = int(np.ceil(a / HOP)), int(min(T - 1, b // HOP))
        if f1 - f0 + 1 < 3:
            continue
        P, Lg = S[:, f0:f1 + 1], L[:, f0:f1 + 1]
        pm = P.mean(1)
        tot = pm.sum() + 1e-20
        cen = float((hz * pm).sum() / tot)
        bw = float(np.sqrt((((hz - cen) ** 2) * pm).sum() / tot))
        roll = float(hz[min(len(hz) - 1, np.searchsorted(np.cumsum(pm) / tot, 0.85))])
        sc = dict(seg_idx=k, seg_start=int(s0 + a), seg_end=int(s0 + b), dur_s=(b - a + 1) / SR, n_frames=f1 - f0 + 1,
                  snr_db=float(10 * np.log10(max(P.sum(0).mean(), 1e-20) / noise_pow)) if noise_pow == noise_pow and noise_pow > 0 else np.nan,
                  level_db=float(10 * np.log10(tot)), centroid_hz=cen, bandwidth_hz=bw, peak_hz=float(hz[int(np.argmax(pm))]), rolloff_hz=roll,
                  flatness=float(np.exp(np.mean(np.log(pm + 1e-20))) / (pm.mean() + 1e-20)))
        Lp = Lg.reshape(32, 4, -1).mean(1)
        Lp = Lp - Lp.mean()
        t_old, t_new = np.linspace(0, 1, Lp.shape[1]), np.linspace(0, 1, 16)
        shape = np.stack([np.interp(t_new, t_old, row) for row in Lp]).ravel().astype(np.float32)
        out.append((sc, M[1:20, f0:f1 + 1].mean(1).astype(np.float32), shape))
    return out


def work(task):
    import soundfile as sf
    url, size, folder, items, variants, seg_mode = task
    res, fails = [], []
    def open_archive():
        last = None
        for attempt in range(6):  # network resets happen: retry with back-off
            try:
                rf_ = RangeFile(url, size)
                zf_ = zipfile.ZipFile(rf_)
                return rf_, zf_, {i.filename: i for i in zf_.infolist()}
            except Exception as e:
                last = e
                time.sleep(2 ** attempt)
        raise last

    try:
        rf, zf, names = open_archive()
    except Exception as e:  # whole task failed
        return [], [(it["file_name"], f"archive open: {e}") for it in items], 0, 0.0, 0.0
    t_net = t_feat = 0.0
    for it in items:
        try:
            t1 = time.perf_counter()
            data = None
            for attempt in range(4):
                try:
                    data = zf.read(names[it["file_name"]])
                    break
                except Exception:
                    time.sleep(2 ** attempt)
                    rf, zf, names = open_archive()
            if data is None:
                raise IOError("read failed after retries")
            t2 = time.perf_counter()
            y, sr = sf.read(io.BytesIO(data), dtype="float32")
            if y.ndim > 1:
                y = y.mean(1)
            if sr != SR:
                import librosa
                y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            for r in it["rows"]:
                sc, ft = one_row(y, r["start"], r["end"], it["segs"], variants, seg_mode)
                sc.update(ann_row=r["ann_row"], FileID=it["FileID"], header_sr=int(sr))
                res.append((sc, ft))
            t_net += t2 - t1
            t_feat += time.perf_counter() - t2
        except Exception as e:
            fails.append((it["file_name"], repr(e)[:200]))
    return res, fails, rf.bytes, t_net, t_feat


def load_segments(file_ids):
    want, segs = set(file_ids), {}
    with open(META / "FileInfo.csv", newline="", encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            if r and int(r[0]) in want:
                v = [int(x) for x in r[6:] if x.strip() not in ("", "...")]
                segs[int(r[0])] = np.array(v[: len(v) // 2 * 2], dtype=np.int64).reshape(-1, 2)
    return segs


if __name__ == "__main__":
    manifest, prefix = Path(sys.argv[1]), sys.argv[2]
    variants = ("ref1", "refmax", "notop") if (len(sys.argv) > 3 and sys.argv[3] == "all") else ("ref1",)
    seg_mode = sys.argv[4] if len(sys.argv) > 4 else ""  # '' | 'seg' (file-level + per-segment) | 'segonly' (per-segment only, fast)
    OUT.mkdir(parents=True, exist_ok=True)
    ann = pd.read_csv(FEAT / "annotations_merged.csv.gz")
    ann["ann_row"] = np.arange(len(ann))
    rows = ann[ann.ann_row.isin(pd.read_csv(manifest).ann_row)].copy()
    segs = load_segments(rows.FileID.unique())
    lst = pd.read_csv(LISTING).set_index("name")
    fi = pd.read_csv(FEAT / "fileinfo_parsed.csv.gz", usecols=["FileID", "file_name"])
    rows = rows.merge(fi, on="FileID", how="left") if "file_name" not in rows.columns else rows
    tasks, CH = [], 60
    for folder, g in rows.groupby("folder"):
        zrow = lst.loc[f"{folder}.zip"]
        items = []
        for (fid, fname), gg in g.groupby(["FileID", "file_name"]):
            items.append(dict(FileID=int(fid), file_name=fname, segs=segs[int(fid)],
                              rows=[dict(ann_row=int(a), start=int(s), end=int(e)) for a, s, e in zip(gg.ann_row, gg.Start, gg.End)]))
        for i in range(0, len(items), CH):
            tasks.append((zrow.url, int(zrow.size_bytes), folder, items[i:i + CH], variants, seg_mode))
    print(f"{len(rows)} annotation rows, {sum(len(t[3]) for t in tasks)} files, {len(tasks)} tasks, variants {variants}", flush=True)
    t0 = time.perf_counter()
    R, F, by, tn, tf, done = [], [], 0, 0.0, 0.0, 0
    with ProcessPoolExecutor(max_workers=6) as ex:
        for res, fails, b, a, c in ex.map(work, tasks, chunksize=1):
            R += res; F += fails; by += b; tn += a; tf += c; done += 1
            if done % 5 == 0:
                print(f"{done}/{len(tasks)} tasks, {len(R)} rows, {len(F)} failed, {by/1e9:.2f} GB read, {time.perf_counter()-t0:.0f}s", flush=True)
    wall = time.perf_counter() - t0
    R.sort(key=lambda r: r[0]["ann_row"])
    idx = pd.DataFrame([r[0] for r in R])
    idx.to_csv(OUT / f"{prefix}_index.csv", index=False)
    segs_all = []
    for sc, ft in R:  # per-segment records travel inside the feature dict under '__seg__'
        for s, m19, sh in ft.pop("__seg__", []):
            s.update(ann_row=sc["ann_row"], FileID=sc["FileID"])
            segs_all.append((s, m19, sh))
    if segs_all:
        pd.DataFrame([s for s, _, _ in segs_all]).to_csv(OUT / f"{prefix}_seg_index.csv", index=False)
        np.savez_compressed(OUT / f"{prefix}_seg_arrays.npz", mfcc19=np.stack([m for _, m, _ in segs_all]), shape=np.stack([sh for _, _, sh in segs_all]))
        print(f"{len(segs_all)} voiced segments from {len(R)} annotation rows", flush=True)
    keys = list(R[0][1].keys())
    if keys:
        np.savez_compressed(OUT / f"{prefix}_features.npz", **{k: np.stack([r[1][k] for r in R]) for k in keys})
    pd.DataFrame(F, columns=["file_name", "error"]).to_csv(OUT / f"{prefix}_failed.csv", index=False)
    pd.DataFrame([dict(prefix=prefix, rows=len(R), files=idx.FileID.nunique(), failed_files=len(F), wall_s=wall, net_wait_s_sum=tn,
                       feature_s_sum=tf, gb_read=by / 1e9, workers=6, rows_per_s=len(R) / wall)]).to_csv(RES / f"step2_timing_{prefix}.csv", index=False)
    with open(LOGS / "claude_download_ledger.csv", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')},figshare range reads ({prefix}),{by},feature extraction\n")
    log_change(f"Step 2 extraction '{prefix}': {len(R)} rows, {len(F)} failed files, {by/1e9:.2f} GB read, {wall:.0f}s wall")
    print(f"done: {len(R)} rows, {len(F)} failed files, {by/1e9:.2f} GB, {wall:.0f}s")

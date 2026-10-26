"""Small, dependency-light clustering library for the audit (numpy + scikit-learn only).

Fuzzy c-means follows the draft: fuzzifier m=2, tolerance 0.005 on the Frobenius change of U,
max 1000 iterations, random-U initialisation, best of n_init restarts by final J_m.
Xie-Beni follows the draft's corrected form: sum(u^m * d^2) / (n * min_{p!=q} ||v_p - v_q||^2).
"""
import time

import numpy as np
from sklearn.metrics import silhouette_score


def _d2(X, V, xx=None):
    xx = (X * X).sum(1) if xx is None else xx
    return np.maximum(xx[None, :] - 2.0 * V @ X.T + (V * V).sum(1)[:, None], 1e-12)


def fcm(X, c, m=2.0, tol=0.005, maxiter=1000, seed=0):
    """One fuzzy c-means run. Returns dict(U (c,n), V (c,d), J, n_iter)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    U = rng.random((c, n))
    U /= U.sum(0, keepdims=True)
    xx = (X * X).sum(1)
    for it in range(1, maxiter + 1):
        Um = U ** m
        V = (Um @ X) / Um.sum(1, keepdims=True)
        D2 = _d2(X, V, xx)
        Un = D2 ** (-1.0 / (m - 1.0))
        Un /= Un.sum(0, keepdims=True)
        done = np.linalg.norm(Un - U) < tol
        U = Un
        if done:
            break
    Um = U ** m
    V = (Um @ X) / Um.sum(1, keepdims=True)
    D2 = _d2(X, V, xx)
    return dict(U=U, V=V, J=float((Um * D2).sum()), n_iter=it)


def fcm_best(X, c, seed=0, n_init=20, **kw):
    best = None
    for k in range(n_init):
        r = fcm(X, c, seed=seed * 1000 + k, **kw)
        if best is None or r["J"] < best["J"]:
            best = r
    return best


def fuzzy_indices(X, U, V, m=2.0):
    """FPC (higher = crisper) and Xie-Beni (lower = better)."""
    n = X.shape[0]
    fpc = float((U * U).sum() / n)
    D2 = _d2(np.asarray(X, dtype=np.float64), V)
    num = float(((U ** m) * D2).sum())
    cd = ((V[:, None, :] - V[None, :, :]) ** 2).sum(-1)
    cd[np.diag_indices_from(cd)] = np.inf
    return fpc, num / (n * cd.min())


def silhouette_fixed(X, labels, n_sub=3000, seed=42):
    """Silhouette on a fixed random subsample (the draft used 3,000 rows when n > 3,000)."""
    n = len(labels)
    if len(np.unique(labels)) < 2:
        return np.nan
    if n > n_sub:
        idx = np.random.default_rng(seed).choice(n, n_sub, replace=False)
        X, labels = X[idx], labels[idx]
        if len(np.unique(labels)) < 2:
            return np.nan
    return float(silhouette_score(X, labels))


def sweep(X, cs=range(2, 7), n_init=20, seed=0, m=2.0):
    """c-sweep returning one dict per c, with timing."""
    out = []
    for c in cs:
        t = time.perf_counter()
        r = fcm_best(X, c, seed=seed, n_init=n_init, m=m)
        fpc, xb = fuzzy_indices(X, r["U"], r["V"], m)
        lab = r["U"].argmax(0)
        out.append(dict(c=c, FPC=fpc, XieBeni=xb, Silhouette=silhouette_fixed(X, lab), J=r["J"],
                        sizes=np.bincount(lab, minlength=c).tolist(), n_iter=r["n_iter"],
                        seconds=time.perf_counter() - t))
    return out


# ----------------------------------------------------------------------------------------------------------------
# shared helpers for the analysis scripts (03a/03c/04/06): dataset loader and bat-level resampling
# ----------------------------------------------------------------------------------------------------------------
def load_set(tag):
    """Return (X, meta): X = N x d PCA scores (float64), meta = one row per unique FILE with the columns
    FileID, Emitter, context_name, channel, month, date, folder, treatment_id, bat (=|emitter| as str), stratum
    (channel|month), em_class, days_since_start and (when available) silence_fraction, snr_db, noise_floor_dbfs,
    clip_frac, voiced_dur_s, n_voiced_seg, span_len_s.  tag 'old' = the draft's saved 50-PC matrix (unique files);
    any other tag reads data/features/claude/set_<tag>_X.npy / set_<tag>_meta.csv written by 05_build_sets.py."""
    import sys
    from pathlib import Path

    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import FEAT, OLD
    if tag == "old":  # draft matrix, CORRECT reading: filename number = row index of Annotations.csv (reproduces the matrix exactly)
        Xp = np.load(OLD / "X_pca_train.npy").astype(np.float64)
        fn = np.load(OLD / "filenames_train.npy", allow_pickle=True)
        nrow = np.array([int(s.split(".")[0]) for s in fn])
        ann = pd.read_csv(FEAT / "annotations_merged.csv.gz", parse_dates=["rec_time"])
        Um = fcm_best(Xp, 2, seed=42, n_init=20)["U"]  # the draft's partition (fitted on all 3,897 rows)
        U = Um.argmax(0)
        if (U == 0).sum() > (U == 1).sum():
            U, Um = 1 - U, Um[::-1]
        first = ~pd.Series(nrow).duplicated().values  # unique annotation rows
        meta = ann.iloc[nrow[first]].reset_index(drop=True)
        meta["ann_row"] = nrow[first]
        meta["cluster_draft"] = U[first]
        meta["u1_draft"] = Um[1][first]
        meta["silence_fraction"] = 1 - meta.voiced_fraction_of_annot
        meta["span_len_s"] = meta.annot_len_s
        meta["n_voiced_seg"] = meta.n_voice_seg_in_annot
        meta["bat"] = meta.Emitter.abs().astype(str)
        meta["stratum"] = meta.channel.astype(str) + "|" + meta.month
        meta["em_class"] = np.where(meta.Emitter == 0, "zero", np.where(meta.Emitter < 0, "negative", "positive"))
        meta["days_since_start"] = (meta.rec_time - pd.Timestamp("2012-06-01")).dt.total_seconds() / 86400
        meta["channel"] = meta.channel.astype(str)
        meta["treatment_id"] = meta.treatment_id.astype(str)
        return Xp[first], meta
    if tag != "old_fileid":
        X = np.load(FEAT / "claude" / f"set_{tag}_X.npy").astype(np.float64)
        meta = pd.read_csv(FEAT / "claude" / f"set_{tag}_meta.csv", parse_dates=["rec_time"])
        meta["bat"] = meta.bat.astype(str)
        return X, meta
    Xp = np.load(OLD / "X_pca_train.npy").astype(np.float64)
    fn = np.load(OLD / "filenames_train.npy", allow_pickle=True)
    em = np.load(OLD / "emitter_train.npy", allow_pickle=True).astype(int)
    cx = np.load(OLD / "context_train.npy", allow_pickle=True)
    rows = pd.DataFrame(dict(FileID=[int(s.split(".")[0]) for s in fn], Emitter=em, context_name=cx))
    ann = pd.read_csv(FEAT / "annotations_merged.csv.gz", parse_dates=["rec_time"]).drop_duplicates(["FileID", "Emitter", "context_name"])
    m = rows.merge(ann[["FileID", "Emitter", "context_name", "channel", "month", "date", "folder", "treatment_id", "rec_time",
                        "voiced_fraction_of_annot", "annot_len_s", "n_voice_seg_in_annot"]],
                   on=["FileID", "Emitter", "context_name"], how="left")
    first = ~m.FileID.duplicated().values
    meta = m[first].reset_index(drop=True)
    meta["silence_fraction"] = 1 - meta.voiced_fraction_of_annot
    meta["span_len_s"] = meta.annot_len_s
    meta["n_voiced_seg"] = meta.n_voice_seg_in_annot
    meta["bat"] = meta.Emitter.abs().astype(str)
    meta["stratum"] = meta.channel.astype(str) + "|" + meta.month
    meta["em_class"] = np.where(meta.Emitter == 0, "zero", np.where(meta.Emitter < 0, "negative", "positive"))
    meta["days_since_start"] = (meta.rec_time - pd.Timestamp("2012-06-01")).dt.total_seconds() / 86400
    meta["channel"] = meta.channel.astype(str)
    meta["treatment_id"] = meta.treatment_id.astype(str)
    return Xp[first], meta


def auc_fast(y, x):
    """AUC of covariate x for the binary label y (rank formula; ties handled by average ranks)."""
    from scipy.stats import rankdata
    y = np.asarray(y)
    r = rankdata(x)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_groups(df, fun, group_col="bat", n_boot=500, seed=0, base_seed=42):
    """95% percentile CI of fun(resampled df), resampling whole groups (bats), never rows."""
    g = df[group_col].values
    ug = np.unique(g)
    idx = {k: np.where(g == k)[0] for k in ug}
    rg = np.random.default_rng(base_seed + seed)
    vals = []
    for _ in range(n_boot):
        ii = np.concatenate([idx[k] for k in rg.choice(ug, len(ug), replace=True)])
        try:
            vals.append(fun(df.iloc[ii]))
        except Exception:
            pass
    return np.nanpercentile(vals, [2.5, 97.5])

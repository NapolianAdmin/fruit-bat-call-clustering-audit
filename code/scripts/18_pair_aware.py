"""Step 4b - pair-aware use of NEGATIVE emitter codes (Metadata.pdf: 'the pair of interacting bats was recognised but their role is in doubt').

For a negative-coded file the caller is one of two known bats: |Emitter| and |Addressee|. Train an identifier on POSITIVE-coded files
(bats with >= 30 files), then on negative-coded files ask how much probability it puts on the two participants:
   p_pair      mean summed probability of the two participants (restricted to classes the model knows)
   top1_in_pair  share of files whose top-1 predicted bat is one of the two participants
Null: pair assignments shuffled among negative files within the same channel x month stratum (keeps which bats are plausible on a channel).
Two training schemes: (a) all positive files; (b) positive files from OTHER channels only (blocked: tests identity beyond channel).
Run: python code/claude/18_pair_aware.py <tag>     e.g. all_voiced_ref1
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RES, SEED, log_change  # noqa: E402
from clusterlib import load_set  # noqa: E402

warnings.filterwarnings("ignore")
TAG = sys.argv[1]
N_PERM = 100
X, u = load_set(TAG)
pos = (u.Emitter > 0).values
vc = u[pos].Emitter.value_counts()
bats = vc[vc >= 30].index.values
trainable = pos & u.Emitter.isin(bats).values
neg = ((u.Emitter < 0) & (u.Addressee < 0)).values  # both members of the pair recognised
nf = u[neg].reset_index(drop=True)
Xn = X[neg]
pairs = list(zip(nf.Emitter.abs().values, nf.Addressee.abs().values))
print(f"{TAG}: {int(trainable.sum())} trainable positive files ({len(bats)} bats), {int(neg.sum())} pair-coded files", flush=True)


def score(P, classes, pair_list):
    cidx = {c: i for i, c in enumerate(classes)}
    pp, top1 = [], []
    for row, (a, b) in zip(P, pair_list):
        ids = [cidx[c] for c in (a, b) if c in cidx]
        if not ids:
            continue
        pp.append(row[ids].sum())
        top1.append(int(np.argmax(row)) in ids)
    return float(np.mean(pp)), float(np.mean(top1)), len(pp)


def run(scheme, P, classes, mask_ok):
    pl = [p for p, k in zip(pairs, mask_ok) if k]
    Pk = P[mask_ok]
    obs_pp, obs_t1, kn = score(Pk, classes, pl)
    strata = (nf.channel.astype(str) + "|" + nf.month).values[mask_ok]
    rg = np.random.default_rng(SEED)
    idx_s = [np.where(strata == s)[0] for s in np.unique(strata)]
    npp, nt1 = [], []
    for _ in range(N_PERM):
        perm = np.arange(len(pl))
        for ii in idx_s:
            perm[ii] = rg.permutation(ii)
        a1, b1, _ = score(Pk, classes, [pl[k] for k in perm])
        npp.append(a1)
        nt1.append(b1)
    ge1, ge2 = int((np.array(npp) >= obs_pp).sum()), int((np.array(nt1) >= obs_t1).sum())
    return dict(scheme=scheme, tag=TAG, pair_files=int(mask_ok.sum()), files_with_a_known_member=kn, p_pair=obs_pp, p_pair_null_mean=float(np.mean(npp)), p_pair_null_q975=float(np.quantile(npp, .975)),
                top1_in_pair=obs_t1, top1_null_mean=float(np.mean(nt1)), top1_null_q975=float(np.quantile(nt1, .975)),
                p_perm=(f"<{1/(N_PERM+1):.3f} (0/{N_PERM})" if max(ge1, ge2) == 0 else f"{(1+ge1)/(N_PERM+1):.3f} / {(1+ge2)/(N_PERM+1):.3f}"))


out = []
mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5)).fit(X[trainable], u.Emitter.values[trainable])
out.append(run("(a) trained on all positive files", mdl.predict_proba(Xn), mdl.classes_, np.ones(len(nf), bool)))
# (b) blocked: for each channel c, train on positives from other channels, score the pair-coded files recorded on c
classes_b = np.sort(bats)
Pb = np.zeros((len(nf), len(classes_b)))
okb = np.zeros(len(nf), bool)
for c in nf.channel.astype(str).unique():
    trn = trainable & (u.channel.astype(str).values != c)
    if trn.sum() < 200 or len(np.unique(u.Emitter.values[trn])) < 5:
        continue
    m2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5)).fit(X[trn], u.Emitter.values[trn])
    sel = nf.channel.astype(str).values == c
    P2 = np.zeros((int(sel.sum()), len(classes_b)))
    pr = m2.predict_proba(Xn[sel])
    for j, cl in enumerate(m2.classes_):
        P2[:, int(np.where(classes_b == cl)[0][0])] = pr[:, j]
    Pb[sel], okb[sel] = P2, True
out.append(run("(b) trained on positive files from OTHER channels only", Pb, classes_b, okb))
R = pd.DataFrame(out)
R.to_csv(RES / f"step4b_pair_aware_{TAG}.csv", index=False)
log_change(f"Step 4b pair-aware identification on '{TAG}': results/step4b_pair_aware_{TAG}.csv")
pd.set_option("display.width", 250)
print(R.round(4).to_string())

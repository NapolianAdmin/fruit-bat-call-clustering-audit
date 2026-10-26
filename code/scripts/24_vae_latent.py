"""Step 6b - a LEARNED latent space for syllables (variational autoencoder, as in Goffinet et al. 2021 / Sainburg et al. 2020), trained WITHOUT the test bats.

Input: the duration- and level-free 512-value syllable 'shape' (16 time bins x 32 mel bands) of the same 30,000 randomly drawn syllables as the `segall_*` sets.
Bats (|emitter|) are split into two balanced halves; the VAE is trained on the syllables of half A only and the latent means of the syllables of half B (bats the model
never saw) form the new sets, which are directly comparable with a linear baseline built on the same rows:
    segall_vae8_b<beta>   8-D VAE latent means (MLP 512-256-128-8, Gaussian likelihood, KL weight beta in {0.1, 1}), standardised
    segall_pca8           first 8 principal components of the shape (PCA fitted on half A)
Reported: held-out reconstruction R^2 of each (VAE vs PCA8), written to results/step6b_vae_reconstruction.csv.
All downstream analyses reuse the standard scripts on these tags (08_methods_and_representations.py, 03c_clusterability_old_matrix.py).
Run: python code/claude/24_vae_latent.py [epochs=60]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FEAT, RES, SEED, log_change  # noqa: E402

OUT = FEAT / "claude"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
torch.set_num_threads(2)

# --- same 30,000 syllables as 16_build_segment_sets.py segall segall 30000
si = pd.read_csv(OUT / "segall_seg_index.csv")
A = np.load(OUT / "segall_seg_arrays.npz")
shape_all, mfcc_all = A["shape"], A["mfcc19"]
ok = np.isfinite(shape_all).all(1) & np.isfinite(mfcc_all).all(1) & np.isfinite(si.snr_db.values) & (si.dur_s.values > 0)
pick = np.sort(np.random.default_rng(SEED).choice(np.where(ok)[0], min(30000, int(ok.sum())), replace=False))
S = shape_all[pick].astype(np.float32)
meta = pd.read_csv(OUT / "set_segall_shape_meta.csv")
assert len(meta) == len(S) and np.array_equal(meta.ann_row.values, si.ann_row.values[pick]), "row alignment with segall_shape failed"

# --- balanced bat halves
bat = meta.bat.values
ub = np.unique(bat)
rg = np.random.default_rng(SEED)
for _ in range(200):
    half = rg.permutation(ub)[: len(ub) // 2]
    trn = np.isin(bat, half)
    if 0.4 <= trn.mean() <= 0.6:
        break
tst = ~trn
print(f"train rows {trn.sum()} ({len(np.unique(bat[trn]))} bats); held-out rows {tst.sum()} ({len(np.unique(bat[tst]))} bats)", flush=True)
mu, sd = S[trn].mean(0), S[trn].std(0) + 1e-6
Z = (S - mu) / sd
Ztr, Zte = torch.tensor(Z[trn]), torch.tensor(Z[tst])


class VAE(nn.Module):
    def __init__(self, d=512, h1=256, h2=128, z=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU())
        self.mu, self.lv = nn.Linear(h2, z), nn.Linear(h2, z)
        self.dec = nn.Sequential(nn.Linear(z, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU(), nn.Linear(h1, d))

    def forward(self, x):
        h = self.enc(x)
        m, lv = self.mu(h), self.lv(h)
        zz = m + torch.randn_like(m) * torch.exp(0.5 * lv)
        return self.dec(zz), m, lv


def train(beta, seed):
    torch.manual_seed(seed)
    net = VAE()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(Ztr)
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 256):
            x = Ztr[perm[i:i + 256]]
            rec, m, lv = net(x)
            loss = ((rec - x) ** 2).sum(1).mean() + beta * (-0.5 * (1 + lv - m ** 2 - lv.exp()).sum(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(x)
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"  beta={beta} epoch {ep} train loss {tot / n:.2f}", flush=True)
    net.eval()
    with torch.no_grad():
        m_te = net(Zte)[1].numpy()
        rec_te = net.dec(torch.tensor(m_te)).numpy()
    return m_te, rec_te


def r2(Zt, rec):
    return float(1 - ((Zt - rec) ** 2).sum() / ((Zt - Zt.mean(0)) ** 2).sum())


rows = []
meta_te = meta[tst].reset_index(drop=True)
for beta in (0.1, 1.0):
    lat, rec = train(beta, SEED)
    rows.append(dict(model=f"vae8_b{beta}", held_out_R2=r2(Z[tst], rec), active_latent_dims=int((lat.std(0) > 0.1 * lat.std(0).max()).sum())))
    X = (lat - lat.mean(0)) / (lat.std(0) + 1e-9)
    tag = f"segall_vae8_b{beta}"
    np.save(OUT / f"set_{tag}_X.npy", X.astype(np.float32))
    meta_te.to_csv(OUT / f"set_{tag}_meta.csv", index=False)
    print(rows[-1], flush=True)

# linear baseline on the same rows: 8 PCs fitted on the training half
U, s, Vt = np.linalg.svd(Z[trn] - Z[trn].mean(0), full_matrices=False)
P = (Z[tst] - Z[trn].mean(0)) @ Vt[:8].T
rec = P @ Vt[:8] + Z[trn].mean(0)
rows.append(dict(model="pca8", held_out_R2=r2(Z[tst], rec), active_latent_dims=8))
X = (P - P.mean(0)) / (P.std(0) + 1e-9)
np.save(OUT / "set_segall_pca8_X.npy", X.astype(np.float32))
meta_te.to_csv(OUT / "set_segall_pca8_meta.csv", index=False)
print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(RES / "step6b_vae_reconstruction.csv", index=False)
log_change(f"Step 6b VAE latent (8-D, beta 0.1 and 1, {EPOCHS} epochs) trained on half the bats; held-out rows {int(tst.sum())}; sets segall_vae8_b0.1, segall_vae8_b1.0, segall_pca8")

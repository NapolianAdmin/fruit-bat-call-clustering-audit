"""Step 4 (preliminary, DRAFT features) - mixed-effects statistics and blocked emitter identification.

NOTE: the draft matrix is whole-annotation-span features (87% unvoiced). This script is the PRELIMINARY version of
step 4; the same code is to be rerun on the voiced-segment features once Codex has produced them.

A. Mixed models of cluster membership on unique files (cluster from FCM c=2 on the saved 50 PCs, seed 42):
   * continuous: logit of FCM membership in cluster 1 ~ context + crossed random intercepts for bat (|emitter|) and
     recording stratum (channel x month); ML fits, likelihood-ratio test of context, variance components, ICC
   * binary: Bayesian-variational mixed logistic model (statsmodels BinomialBayesMixedGLM) with the same terms
   * repeatability by bat: ICC of the null model, with and without the stratum term; 95% CI resampling bats
   Run on (i) all unique files with bat = |emitter| (0 = unknown group, negative = pair-labelled) and
          (ii) positive-emitter files only (the only rows where the bat identity is real).
B. Emitter identification (positive codes, bats with >= 30 unique files): multinomial logistic regression on the 50 PCs,
   random stratified 5-fold vs blocked by day / channel / month / archive folder. Test rows whose bat is absent from the
   training fold are dropped and the retained share is reported. CI: bootstrap over test blocks (days).
Run: python code/claude/04_stats_old_matrix.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEAT, OLD, RES, SEED, log_change  # noqa: E402
from clusterlib import fcm_best, load_set  # noqa: E402

warnings.filterwarnings("ignore")
import os  # noqa: E402
N_BOOT_ICC, N_BOOT_ACC = int(os.environ.get("N_BOOT_ICC", 150)), 500  # lower N_BOOT_ICC for the 16k-file sets
ONLY_B = "--only-emitter" in sys.argv  # skip the (slow) mixed models; part A results are already on disk

TAG = next((a for a in sys.argv[1:] if not a.startswith("--")), "old")  # 'old' = draft matrix; else e.g. baseline_voiced_ref1
SUF = "" if TAG == "old" else f"_{TAG}"
Xu, d = load_set(TAG)  # unique files; FCM is fitted on these rows (the draft fitted on 3,897 rows incl. duplicates)
fit = fcm_best(Xu, 2, seed=SEED, n_init=20)
big = int(np.argmax(np.bincount(fit["U"].argmax(0))))  # orient: cluster "1" = the larger cluster (as in 03a)
d["u1"] = fit["U"][big]
if (fit["U"] ** 2).sum() / len(Xu) < 0.51:  # FCM collapsed (FPC ~ 0.5): memberships carry no information -> forced k-means split, labelled as such
    from sklearn.cluster import KMeans
    km = KMeans(2, n_init=10, random_state=SEED).fit(Xu)
    d2 = ((Xu[:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1)
    big = int(np.argmax(np.bincount(km.labels_)))
    u_ = (1 / np.maximum(d2, 1e-12))
    d["u1"] = (u_ / u_.sum(1, keepdims=True))[:, big]
    print("NOTE: FCM collapsed on this set; cluster/membership = forced k-means k=2 split (centroid-distance memberships)", flush=True)
if "u1_draft" in d.columns:  # the draft matrix: the draft's own fuzzy memberships
    d["u1"] = d.u1_draft
d["cluster"] = (d.u1 > 0.5).astype(int)
uc = np.clip(d.u1, 1e-4, 1 - 1e-4)
d["lg"] = np.log(uc / (1 - uc))
d["one"] = 1
d["context_name"] = d.context_name.astype(str)
REF = "General"
FIXED = f"C(context_name, Treatment('{REF}'))"
VC = {"bat": "0+C(bat)", "stratum": "0+C(stratum)"}


def mixed_fit(data, fixed, reml=False):
    md = smf.mixedlm(f"lg ~ {fixed}", data, groups=data["one"], vc_formula=VC)
    return md.fit(reml=reml, method="lbfgs", maxiter=300)


def vcomps(res):
    return dict(zip(res.model.exog_vc.names, res.vcomp)), float(res.scale)


def icc_from(res):
    vc, sc = vcomps(res)
    tot = sum(vc.values()) + sc
    return vc.get("bat", 0) / tot, vc.get("stratum", 0) / tot


out_lrt, out_fe, out_icc, out_bin = [], [], [], []
for tag, data in [] if ONLY_B else [("all unique files (bat=|emitter|; 0=unknown, neg=pair-labelled)", d),
                                    ("positive-emitter files only (real bat identity)", d[d.Emitter > 0].reset_index(drop=True))]:
    keep = data.context_name.value_counts()
    data = data[data.context_name.isin(keep[keep >= 5].index)].reset_index(drop=True)  # contexts with >= 5 files
    m0 = mixed_fit(data, "1")
    m1 = mixed_fit(data, FIXED)
    dfc = len(m1.fe_params) - len(m0.fe_params)
    lr = 2 * (m1.llf - m0.llf)
    p = chi2.sf(lr, dfc)
    ic_b, ic_s = icc_from(m0)
    vc0, sc0 = vcomps(m0)
    vc1, sc1 = vcomps(m1)
    out_lrt.append(dict(subset=tag, n_files=len(data), n_bats=data.bat.nunique(), n_strata=data.stratum.nunique(), n_contexts=data.context_name.nunique(),
                        LR_chi2=lr, df=dfc, p=("<1e-16" if p < 1e-16 else p), var_bat_null=vc0["bat"], var_stratum_null=vc0["stratum"], var_resid_null=sc0,
                        var_bat_with_context=vc1["bat"], var_stratum_with_context=vc1["stratum"], ICC_bat_null=ic_b, ICC_stratum_null=ic_s))
    ci = m1.conf_int()
    for k in m1.fe_params.index:
        if k != "Intercept":
            out_fe.append(dict(subset=tag, term=k.replace("C(context_name, Treatment('General'))", "context"), logit_diff_vs_General=m1.fe_params[k],
                               ci95_lo=ci.loc[k, 0], ci95_hi=ci.loc[k, 1]))
    rg = np.random.default_rng(SEED + 5)
    bats = data.bat.unique()
    ib, is_ = [], []
    for _ in range(N_BOOT_ICC):
        pick = rg.choice(bats, len(bats), replace=True)
        bd = pd.concat([data[data.bat == b].assign(bat=f"{b}_{j}") for j, b in enumerate(pick)], ignore_index=True)
        try:
            a, b_ = icc_from(mixed_fit(bd, "1", reml=True))
            ib.append(a)
            is_.append(b_)
        except Exception:
            pass
    out_icc.append(dict(subset=tag, ICC_bat_adjusted_for_stratum=ic_b, ci95_lo=np.percentile(ib, 2.5), ci95_hi=np.percentile(ib, 97.5),
                        ICC_stratum=ic_s, ci95_lo_stratum=np.percentile(is_, 2.5), ci95_hi_stratum=np.percentile(is_, 97.5), n_boot_ok=len(ib)))
    g = BinomialBayesMixedGLM.from_formula(f"cluster ~ {FIXED}", VC, data)
    r = g.fit_vb()
    sd = np.exp(r.vcp_mean)
    for k, mu, s in zip(g.exog_names, r.fe_mean, r.fe_sd):
        if k != "Intercept":
            out_bin.append(dict(subset=tag, term=k.replace("C(context_name, Treatment('General'))", "context"), logodds_post_mean=mu, lo=mu - 1.96 * s, hi=mu + 1.96 * s))
    out_bin.append(dict(subset=tag, term="random-effect SD: bat (lo col) / stratum (hi col are NOT CIs; posterior means of exp(log sd))",
                        logodds_post_mean=np.nan, lo=float(sd[0]), hi=float(sd[1])))
    print(tag, "LR chi2 %.1f df %d p %s | ICC bat %.3f stratum %.3f" % (lr, dfc, p, ic_b, ic_s), flush=True)

if not ONLY_B:
    pd.DataFrame(out_lrt).to_csv(RES / f"step4_mixed_context_lrt{SUF}.csv", index=False)
    pd.DataFrame(out_fe).to_csv(RES / f"step4_mixed_context_effects{SUF}.csv", index=False)
    pd.DataFrame(out_icc).to_csv(RES / f"step4_repeatability_by_bat{SUF}.csv", index=False)
    pd.DataFrame(out_bin).to_csv(RES / f"step4_binary_glmm_vb{SUF}.csv", index=False)

# ------------------------------------------------------------------ B. emitter identification: random vs blocked
pos = d[d.Emitter > 0].copy()
vc_ = pos.Emitter.value_counts()
pos = pos[pos.Emitter.isin(vc_[vc_ >= 30].index)]
Xe = Xu[pos.index.values]
ye = pos.Emitter.values
classes = np.unique(ye)
chance_major = pd.Series(ye).value_counts(normalize=True).max()
print(f"emitter task: {len(pos)} files, {len(classes)} bats, majority-class rate {chance_major:.3f}, uniform chance {1/len(classes):.3f}", flush=True)


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


schemes = {"random 5-fold (files)": None, "blocked by day": "date", "blocked by channel": "channel", "blocked by month": "month", "blocked by archive folder": "folder"}
res_e = []
for name, key in schemes.items():
    if key is None:
        splits = StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xe, ye)
        blocks = pos.date.astype(str).values
    else:
        blocks = pos[key].astype(str).values
        splits = GroupKFold(min(5, len(np.unique(blocks)))).split(Xe, ye, blocks)
    pred = np.full(len(pos), -1)
    kept = np.zeros(len(pos), bool)
    for tr, te in splits:
        mdl = clf().fit(Xe[tr], ye[tr])
        ok = np.isin(ye[te], np.unique(ye[tr]))
        if not ok.any():  # no test bat was seen in training (possible with blocked splits)
            continue
        pred[te[ok]] = mdl.predict(Xe[te[ok]])
        kept[te[ok]] = True
    corr = (pred == ye) & kept
    acc = corr[kept].mean()
    bal = balanced_accuracy_score(ye[kept], pred[kept])
    ub = np.unique(blocks[kept])
    ib = {b: np.where((blocks == b) & kept)[0] for b in ub}
    rg = np.random.default_rng(SEED + 11)
    bs = [corr[np.concatenate([ib[b] for b in rg.choice(ub, len(ub), replace=True)])].mean() for _ in range(N_BOOT_ACC)]
    res_e.append(dict(scheme=name, n_files=len(pos), n_bats=len(classes), test_rows_kept=int(kept.sum()), share_kept=kept.mean(), accuracy=acc,
                      ci95_lo=np.percentile(bs, 2.5), ci95_hi=np.percentile(bs, 97.5), balanced_accuracy=bal,
                      majority_chance=chance_major, uniform_chance=1 / len(classes), n_test_blocks=len(ub)))
    print(f"{name:28s} acc {acc:.3f} [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]  bal {bal:.3f}  kept {kept.mean():.2f}", flush=True)
E = pd.DataFrame(res_e)
E["accuracy_drop_vs_random"] = E.accuracy.iloc[0] - E.accuracy
E.to_csv(RES / f"step4_emitter_id_random_vs_blocked{SUF}.csv", index=False)
log_change(f"Step 4 on set '{TAG}': mixed models, bat repeatability, random-vs-blocked emitter ID; results/step4_*{SUF}.csv")
print(E.round(3).to_string())

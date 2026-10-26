"""Supplementary figure: data partitions and the discovery -> validation sequence of the Isolation-group analysis.
All counts are read from results/step7_sample_accounting.csv, step7_sample_overlap.csv, step7_replication_summary.csv and step7_rest_summary.csv (no number is typed).
The dates are the time stamps of the entries in the chronological analysis log that fixed the criteria before any held-out file was analysed.
Writes paper/figures/fig_flow_v4.pdf.   Run: python code/claude/49_flow_diagram.py
"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RES  # noqa: E402

acc = pd.read_csv(RES / "step7_sample_accounting.csv").set_index("partition")
ovl = pd.read_csv(RES / "step7_sample_overlap.csv").set_index("held_out")
rep = pd.read_csv(RES / "step7_replication_summary.csv").iloc[0]
rest = pd.read_csv(RES / "step7_rest_summary.csv").iloc[0]
plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
import textwrap

H = 4.5  # inches; one data unit = H*72/100 pt vertically, 6.69*72/100 pt horizontally
fig = plt.figure(figsize=(6.69, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-0.8, 100.8)
ax.set_ylim(0, 100)
ax.axis("off")
LH = 8.6 / (H * 72 / 100)  # line height in data units


def box(x, ytop, w, paras, fc="#f2f4f7", ec="#5b6770", bold_first=True):
    """paras: list of strings; each is wrapped to the box width; returns the bottom y."""
    cpl = int(w * 4.82 / 4.45)  # characters per line at 7 pt
    lines = []
    for i, pr in enumerate(paras):
        for ln in textwrap.wrap(pr, cpl) or [""]:
            lines.append((ln, i == 0 and bold_first))
    h = len(lines) * LH + 2.0
    ax.add_patch(FancyBboxPatch((x, ytop - h), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec=ec, lw=0.7))
    for i, (ln, bd) in enumerate(lines):
        ax.text(x + w / 2, ytop - 1.0 - (i + 0.5) * LH, ln, ha="center", va="center", fontsize=7, fontweight="bold" if bd else "normal")
    return ytop - h


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#5b6770", shrinkA=0, shrinkB=0))


a_ = acc.loc["all annotated files"]
yb = box(20, 99, 60, [f"Annotated corpus: {int(a_.files):,} files ({int(a_.isolation_files):,} Isolation-context)", "205,252 unannotated recordings are not used"], fc="#e8eef6")
cols = [("discovery", "Discovery sample", 0, 28, "#fdf1e3"), ("replication", "Replication sample", 30, 22, "#e6f3e6"), ("completion", "Completion sample", 54, 26, "#e6f3e6"), ("reproduction only", "Reproduction only", 81.5, 18.5, "#eeeeee")]
y2 = yb - 5
bots = []
for k, t, x, w, c in cols:
    r = acc.loc[k]
    bots.append(box(x, y2, w, [t, f"{int(r.files):,} files", f"{int(r.isolation_files):,} Isolation; {int(r.positive_coded_bats)} bats"], fc=c))
    arrow(50, yb - 0.3, x + w / 2, y2 + 0.6)
y3 = min(bots) - 4
yl = box(0, y3, 28, ["Used for (discovery sample only)", "all development choices and settings (50 principal components, logistic regression C = 0.5, UMAP and HDBSCAN settings)", "exploratory discovery of the cluster (UMAP-HDBSCAN on 12,000 voiced files)",
                     "supervised tests blocked by bat, channel, treatment and age window"], fc="#fdf1e3")
arrow(14, min(bots) - 0.3, 14, y3 + 0.6)
yc = box(30, y3, 70, ["Criteria C1-C3 fixed before any held-out file was analysed", "written to the chronological analysis log on 8 Oct 2026 at 09:48 (replication sample) and 15:15 (completion sample): C1 AUC >= 0.90; C2 a cluster of 3-12% of files with >= 70% Isolation-context; C3 null datasets fragment more than the real data"],
        fc="#fffbe6", ec="#b59b00")
arrow(41, min(bots[1:3]) - 0.3, 41, y3 + 0.6)
arrow(67, min(bots[1:3]) - 0.3, 67, y3 + 0.6)
y4 = yc - 4
ye = box(30, y4, 70, ["Evaluation of the held-out files", "scaler, PCA and Isolation model fitted on discovery files only; UMAP-HDBSCAN fitted on the new files alone (no labels used to fit)",
                      f"replication: AUC {rep.C1_AUC:.3f}; cluster {rep.C2_cluster_share * 100:.1f}% of files, {rep.C2_isolation_share_within * 100:.0f}% Isolation-context",
                      f"completion: AUC {rest.C1_AUC:.3f}; cluster {rest.C2_cluster_share * 100:.1f}% of files, {rest.C2_isolation_share_within * 100:.0f}% Isolation-context",
                      "no feature, model or threshold was tuned on these files"], fc="#e6f3e6")
arrow(65, yc - 0.3, 65, y4 + 0.6)
o1, o2 = ovl.loc["replication"], ovl.loc["completion"]
y5 = min(yl, ye) - 4
box(0, y5, 99, ["Unit separated between stages: the recording (file); bats, channels and recording days are not separated",
                f"All {int(o1.bats_in_held_out)} positive-coded bats of each held-out sample and all {int(o1.channels_in_held_out)} channels also occur in the discovery sample (recording days shared: {int(o1.recording_days_also_in_discovery)} of {int(o1.recording_days_in_held_out)} and {int(o2.recording_days_also_in_discovery)} of {int(o2.recording_days_in_held_out)}). "
                "The held-out files therefore test over-fitting to the discovery recordings (same corpus, same colony), not generalisation to new individuals or colonies; generalisation across bats, channels and treatments is tested inside the discovery sample."], fc="#fbeaea", ec="#a05050")
out = Path(__file__).resolve().parents[2] / "paper" / "figures" / "fig_flow_v4.pdf"
fig.savefig(out)
print("saved", out)

"""Shared paths, seeds and small helpers for the bat_audit pipeline."""
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "raw" / "prat2017_meta"
FEAT = ROOT / "data" / "features"
RES = ROOT / "results"
FIG = ROOT / "figures"
LOGS = ROOT / "logs"
OLD = Path(r"D:\Downloads\validation_suite_results-20260626T063403Z-3-001\validation_suite_results")

# Table 4 of Metadata.pdf (Prat et al. 2017)
CONTEXT_NAMES = {0: "Unknown", 1: "Separation", 2: "Biting", 3: "Feeding", 4: "Fighting",
                 5: "Grooming", 6: "Isolation", 7: "Kissing", 8: "Landing",
                 9: "Mating protest", 10: "Threat-like", 11: "General", 12: "Sleeping"}


def rng(offset=0):
    return np.random.default_rng(SEED + offset)


def cramers_v(a, b, bias_correct=True):
    """Cramer's V between two label vectors (Bergsma bias correction by default).

    Returns (V, chi2, p, n, r, k)."""
    from scipy.stats import chi2_contingency
    tab = pd.crosstab(pd.Series(np.asarray(a)), pd.Series(np.asarray(b)))
    tab = tab.loc[tab.sum(axis=1) > 0, tab.sum(axis=0) > 0]
    n = tab.values.sum()
    r, k = tab.shape
    if r < 2 or k < 2:
        return np.nan, np.nan, np.nan, n, r, k
    chi2, p, _, _ = chi2_contingency(tab.values, correction=False)
    phi2 = chi2 / n
    if bias_correct:
        phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
        rc = r - (r - 1) ** 2 / (n - 1)
        kc = k - (k - 1) ** 2 / (n - 1)
        denom = min(kc - 1, rc - 1)
        v = np.sqrt(phi2c / denom) if denom > 0 else np.nan
    else:
        v = np.sqrt(phi2 / min(r - 1, k - 1))
    return v, chi2, p, n, r, k


def log_change(text):
    """Append a line to the analysis/changes log (every choice, every number that changed)."""
    LOGS.mkdir(exist_ok=True)
    with open(LOGS / "analysis_log.md", "a", encoding="utf-8") as f:
        f.write(f"- [{datetime.now():%Y-%m-%d %H:%M}] {text}\n")


def record_versions(path=None):
    import importlib
    path = path or LOGS / "package_versions.txt"
    lines = [f"python {sys.version}", f"platform {platform.platform()}", f"seed {SEED}"]
    for m in ["numpy", "pandas", "scipy", "sklearn", "statsmodels", "matplotlib", "librosa",
              "soundfile", "torch", "torchaudio", "pypdf", "requests"]:
        try:
            lines.append(f"{m} {importlib.import_module(m).__version__}")
        except Exception:
            lines.append(f"{m} NOT INSTALLED")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

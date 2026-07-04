# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Luis Guillermo Papiri Melendez

"""
spectral_level_stats.py
========================

Rigorous random-matrix-theory (RMT) level-statistics diagnostic for neural
network hidden-layer activations.

This module exists because the first version of this analysis
("Wigner Moat Distribution") was wrong: it claimed GUE statistics from a
covariance matrix that was never mean-centered, had 7 eigenvalues (nowhere
near enough for spacing statistics), used no spectral unfolding, and had no
theoretical justification for GUE over GOE. The plot it produced showed
eigenvalue pile-up near zero -- the opposite of level repulsion.

This rebuild fixes all four problems:
  1. Strict mean-centering before covariance.
  2. Polynomial spectral unfolding (removes the mean level density trend
     so what's left is pure fluctuation statistics).
  3. Closed-form KS tests against Poisson, GOE, and GUE CDFs, PLUS Monte
     Carlo p-value calibration against finite-N synthetic ensembles
     (closed-form asymptotics are for N -> infinity; finite covariance
     spectra are not that).
  4. The unfolding-free ratio statistic <r> (Atas et al. 2013), which is
     far more robust than KS-on-unfolded-spacings when the raw spectrum is
     a Marchenko-Pastur bulk with a few outlier "spike" eigenvalues --
     exactly the shape of a trained network's activation covariance
     spectrum. Polynomial unfolding distorts around those spikes; <r>
     doesn't need unfolding at all.

Reference values (Atas et al., PRL 110, 084101 (2013)):
    <r>_Poisson = 0.386     <r>_GOE = 0.536     <r>_GUE = 0.603

Self-test at the bottom confirms the pipeline correctly identifies all
three ensembles from synthetic ground truth before you trust it on real
data.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from dataclasses import dataclass, field

R_MEAN = {"Poisson": 0.386, "GOE": 0.536, "GUE": 0.603}


# ---------------------------------------------------------------------------
# Covariance construction
# ---------------------------------------------------------------------------

def covariance_from_activations(A: np.ndarray) -> np.ndarray:
    """
    A: (n_samples, n_features) activation matrix.
    Returns the strictly mean-centered sample covariance matrix.
    This centering step was missing in the original flawed pipeline.
    """
    A = np.asarray(A, dtype=np.float64)
    A_centered = A - A.mean(axis=0, keepdims=True)
    n = A_centered.shape[0]
    return (A_centered.T @ A_centered) / max(n - 1, 1)


def clean_covariance_eigs(C: np.ndarray, edge_trim: float = 0.075) -> np.ndarray:
    """
    Eigenvalues of a symmetric covariance matrix, sorted ascending, with the
    extreme `edge_trim` fraction removed from each end.

    Trained-network covariance spectra are a Marchenko-Pastur bulk plus a
    handful of huge outlier "spike" eigenvalues from dominant features.
    Those spikes are real but they are not part of the bulk fluctuation
    statistics we're testing -- keeping them in poisons both the unfolding
    fit and the spacing statistics. Trim harder (0.10-0.15) if you still
    see obvious outliers dominating the unfolded spectrum.
    """
    ev = np.linalg.eigvalsh(C)
    ev = np.sort(ev)
    n = len(ev)
    k = int(np.floor(n * edge_trim))
    if k > 0:
        ev = ev[k: n - k]
    return ev


# ---------------------------------------------------------------------------
# Spectral unfolding
# ---------------------------------------------------------------------------

def unfold_spectrum(eigenvalues: np.ndarray, poly_degree: int = 6) -> np.ndarray:
    """
    Polynomial unfolding: fit a smooth polynomial to the cumulative
    staircase N(E) (the number of eigenvalues below E), then use that fit
    to map raw eigenvalues onto a new scale with unit mean spacing.

    This removes the system-specific mean level density so that what
    remains reflects only the fluctuation statistics (the thing RMT
    universality actually predicts).

    NOTE: This is the step that is fragile against Marchenko-Pastur bulk +
    outlier spectra. Prefer `spacing_ratios` (unfolding-free) when you
    suspect outliers are present -- see ratio_test.py for the
    unfolding-free two-sample test that resolves ambiguity when this
    function's output disagrees with <r>.
    """
    ev = np.sort(eigenvalues)
    n = len(ev)
    staircase = np.arange(1, n + 1)

    coeffs = np.polyfit(ev, staircase, deg=poly_degree)
    smooth_staircase = np.polyval(coeffs, ev)

    unfolded = np.diff(smooth_staircase)
    unfolded = unfolded[unfolded > 0]  # guard against non-monotonic fit artifacts
    return unfolded / unfolded.mean() if len(unfolded) else unfolded


def spacing_ratios(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Unfolding-free ratio statistic (Atas et al. 2013):
        r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})
    where s_n are RAW (not unfolded) consecutive spacings. Requires no
    unfolding at all, which makes it robust to the exact spectral shape --
    the single most important practical property for real activation
    covariance spectra.
    """
    ev = np.sort(eigenvalues)
    s = np.diff(ev)
    s = s[s > 0]
    if len(s) < 2:
        return np.array([])
    r = np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])
    return r


# ---------------------------------------------------------------------------
# Closed-form reference distributions
# ---------------------------------------------------------------------------

def cdf_poisson(s):
    return 1.0 - np.exp(-s)


def cdf_goe(s):
    # Wigner surmise, beta=1: p(s) = (pi/2) s exp(-pi s^2 / 4)
    return 1.0 - np.exp(-np.pi * s ** 2 / 4.0)


def cdf_gue(s):
    # Wigner surmise, beta=2: p(s) = (32/pi^2) s^2 exp(-4 s^2 / pi)
    # CDF via erf: F(s) = erf(2s/sqrt(pi)) - (8s/pi) exp(-4s^2/pi)
    from scipy.special import erf
    return erf(2.0 * s / np.sqrt(np.pi)) - (8.0 * s / np.pi) * np.exp(-4.0 * s ** 2 / np.pi)


_CDFS = {"Poisson": cdf_poisson, "GOE": cdf_goe, "GUE": cdf_gue}


def _ks_against_all(unfolded_spacings: np.ndarray) -> dict:
    out = {}
    for name, cdf in _CDFS.items():
        D, p = stats.kstest(unfolded_spacings, cdf)
        out[name] = {"D": float(D), "p": float(p)}
    return out


# ---------------------------------------------------------------------------
# Brody distribution fit (interpolates Poisson <-> GOE/GUE crossover)
# ---------------------------------------------------------------------------

def brody_pdf(s, q):
    b = (np.math.gamma((q + 2) / (q + 1))) ** (q + 1)
    return (q + 1) * b * s ** q * np.exp(-b * s ** (q + 1))


def fit_brody_q(unfolded_spacings: np.ndarray) -> float:
    """Maximum-likelihood-ish fit of the Brody parameter q via histogram LSQ.
    q=0 -> Poisson, q=1 -> GOE-like, q=2 -> GUE-like (heuristic mapping)."""
    hist, edges = np.histogram(unfolded_spacings, bins=40, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    try:
        popt, _ = curve_fit(brody_pdf, centers, hist, p0=[1.0], bounds=(0, 3))
        return float(popt[0])
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Monte Carlo p-value calibration
# ---------------------------------------------------------------------------

def monte_carlo_pvalue(observed_D: float, ensemble: str, n_matrix: int, n_mc: int = 200,
                        rng: np.random.Generator | None = None) -> float:
    """
    Closed-form KS asymptotics assume N -> infinity. Real covariance
    matrices don't have infinite dimension, so we calibrate the p-value by
    generating `n_mc` synthetic matrices of the SAME size from the target
    ensemble, computing their KS statistic against the closed-form CDF, and
    reporting the fraction that exceed the observed D. This is the
    statistically honest way to get a p-value at finite N.
    """
    rng = rng or np.random.default_rng()
    cdf = _CDFS[ensemble]
    D_samples = np.empty(n_mc)
    for i in range(n_mc):
        ev = _sample_ensemble_eigs(ensemble, n_matrix, rng)
        unfolded = unfold_spectrum(ev)
        if len(unfolded) < 2:
            D_samples[i] = np.nan
            continue
        D_samples[i], _ = stats.kstest(unfolded, cdf)
    D_samples = D_samples[~np.isnan(D_samples)]
    if len(D_samples) == 0:
        return float("nan")
    return float(np.mean(D_samples >= observed_D))


def _sample_ensemble_eigs(ensemble: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Ground-truth eigenvalue samples for self-testing / MC calibration."""
    if ensemble == "Poisson":
        return np.sort(rng.exponential(scale=1.0, size=n).cumsum())
    if ensemble == "GOE":
        M = rng.standard_normal((n, n))
        S = (M + M.T) / np.sqrt(2 * n)
        return np.linalg.eigvalsh(S)
    if ensemble == "GUE":
        M = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
        H = (M + M.conj().T) / np.sqrt(2 * n)
        return np.linalg.eigvalsh(H)
    raise ValueError(f"unknown ensemble {ensemble}")


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

@dataclass
class LevelStatsReport:
    n_eigs: int
    ks: dict
    r_mean: float
    r_closest: str
    brody_q: float
    verdict: str
    pooled_r: float | None = None
    pooled_r_closest: str | None = None
    extras: dict = field(default_factory=dict)


def report_from_covariance(C: np.ndarray, monte_carlo: bool = False, n_mc: int = 200,
                            edge_trim: float = 0.075) -> dict:
    ev = clean_covariance_eigs(C, edge_trim=edge_trim)
    unfolded = unfold_spectrum(ev)
    ks = _ks_against_all(unfolded) if len(unfolded) >= 2 else {k: {"D": float("nan"), "p": float("nan")} for k in _CDFS}

    if monte_carlo and len(unfolded) >= 2:
        for name in ks:
            ks[name]["p_mc"] = monte_carlo_pvalue(ks[name]["D"], name, len(ev), n_mc=n_mc)

    r = spacing_ratios(ev)
    r_mean = float(r.mean()) if len(r) else float("nan")
    r_closest = min(R_MEAN, key=lambda k: abs(R_MEAN[k] - r_mean)) if r_mean == r_mean else "undefined"
    brody_q = fit_brody_q(unfolded) if len(unfolded) >= 10 else float("nan")

    # Verdict priority: <r> is the primary discriminator (unfolding-free,
    # robust to MP-bulk-plus-spikes). KS-on-unfolded-spacings is reported
    # for completeness but should not override <r> when the two disagree --
    # see ratio_test.py and the README for why.
    verdict = r_closest

    return {
        "n_eigs": len(ev),
        "ks": ks,
        "r_mean": r_mean,
        "r_closest": r_closest,
        "brody_q": brody_q,
        "verdict": verdict,
    }


def summary(rep: dict) -> str:
    lines = [f"n_eigs={rep['n_eigs']}"]
    for name, vals in rep["ks"].items():
        line = f"  KS vs {name:8s}: D={vals['D']:.4f}  p={vals['p']:.4f}"
        if "p_mc" in vals:
            line += f"  p_mc={vals['p_mc']:.4f}"
        lines.append(line)
    lines.append(f"  <r>={rep['r_mean']:.4f}  -> nearest {rep['r_closest']}  (Brody q={rep['brody_q']:.3f})")
    lines.append(f"  VERDICT (by <r>, unfolding-free): {rep['verdict']}")
    return "\n".join(lines)


def plot_level_statistics(rep: dict, save_path: str, title: str = "") -> None:
    """Optional plotting helper -- requires matplotlib, imported lazily so
    this module has no hard plotting dependency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(rep["ks"].keys())
    Ds = [rep["ks"][k]["D"] for k in labels]
    ax.bar(labels, Ds, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("KS statistic D (lower = better fit)")
    ax.set_title(title or f"Level statistics -- verdict: {rep['verdict']}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Self-test: pipeline must recover ground truth on synthetic ensembles
# ---------------------------------------------------------------------------

def _self_test(n_matrix: int = 200, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    print("=== self-test: pipeline vs synthetic ground truth ===")
    for ensemble in ("Poisson", "GOE", "GUE"):
        ev = _sample_ensemble_eigs(ensemble, n_matrix, rng)
        r = spacing_ratios(ev)
        r_mean = r.mean()
        closest = min(R_MEAN, key=lambda k: abs(R_MEAN[k] - r_mean))
        status = "PASS" if closest == ensemble else "FAIL"
        print(f"  {ensemble:8s} ground truth -> <r>={r_mean:.4f}  identified as {closest:8s}  [{status}]")


if __name__ == "__main__":
    _self_test()

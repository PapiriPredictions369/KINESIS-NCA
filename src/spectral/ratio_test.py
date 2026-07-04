# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Luis Guillermo Papiri Melendez

"""
ratio_test.py
=============

Why this file exists: on real trained-network activation covariance
spectra (Marchenko-Pastur bulk + outlier spikes), polynomial unfolding
distorts the spacing distribution enough that KS-vs-unfolded-spacings can
favor Poisson even when <r> is nowhere near the Poisson value (0.386).
This happened on the actual KINESIS-NCA/baseline MNIST runs: KS picked Poisson
(smallest D) while <r> ~= 0.52, which is impossible under Poisson.

The fix: don't unfold at all. Build a large synthetic reference sample of
the ratio statistic r for each ensemble at matched sample size, and run a
standard two-sample KS test between the empirical r distribution and each
reference. This sidesteps unfolding entirely and correctly resolves
GOE-vs-Poisson ambiguity where the unfolded-spacing approach fails.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from spectral_level_stats import _sample_ensemble_eigs, spacing_ratios, R_MEAN


def reference_ratio_sample(ensemble: str, n_matrix: int, n_draws: int = 300,
                            rng: np.random.Generator | None = None) -> np.ndarray:
    """Pool ratio statistics across `n_draws` independent synthetic matrices
    of size `n_matrix` drawn from `ensemble`, giving a large reference
    sample of r-values at matched finite-N statistics."""
    rng = rng or np.random.default_rng()
    pooled = []
    for _ in range(n_draws):
        ev = _sample_ensemble_eigs(ensemble, n_matrix, rng)
        pooled.append(spacing_ratios(ev))
    return np.concatenate(pooled) if pooled else np.array([])


def two_sample_ratio_test(empirical_r: np.ndarray, n_matrix: int, n_draws: int = 300,
                           rng: np.random.Generator | None = None) -> dict:
    """
    Two-sample KS test of the empirical ratio distribution against a large
    synthetic reference for each of Poisson / GOE / GUE, all built from
    matrices of the same size as the data (matched finite-size effects).

    Returns a dict keyed by ensemble name with {"D": ..., "p": ...}, plus
    "closest_by_mean" (nearest reference <r>) and "closest_by_ks" (smallest
    two-sample D) so you can see immediately whether they agree.
    """
    rng = rng or np.random.default_rng()
    results = {}
    for ensemble in ("Poisson", "GOE", "GUE"):
        ref = reference_ratio_sample(ensemble, n_matrix, n_draws=n_draws, rng=rng)
        D, p = stats.ks_2samp(empirical_r, ref)
        results[ensemble] = {"D": float(D), "p": float(p)}

    r_mean = float(empirical_r.mean())
    closest_by_mean = min(R_MEAN, key=lambda k: abs(R_MEAN[k] - r_mean))
    closest_by_ks = min(results, key=lambda k: results[k]["D"])

    return {
        "per_ensemble": results,
        "empirical_r_mean": r_mean,
        "closest_by_mean": closest_by_mean,
        "closest_by_ks": closest_by_ks,
        "agree": closest_by_mean == closest_by_ks,
    }


def summarize(result: dict) -> str:
    lines = [f"empirical <r> = {result['empirical_r_mean']:.4f}"]
    for name, vals in result["per_ensemble"].items():
        lines.append(f"  two-sample KS vs {name:8s}: D={vals['D']:.4f}  p={vals['p']:.4f}")
    lines.append(f"  closest by mean <r>: {result['closest_by_mean']}")
    lines.append(f"  closest by two-sample KS: {result['closest_by_ks']}")
    lines.append(f"  agreement: {'YES' if result['agree'] else 'NO -- inspect before reporting a verdict'}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Demonstration: construct a case where naive unfolded-KS fails
    # (GOE ground truth contaminated with a few huge outlier eigenvalues,
    # mimicking a trained network's MP-bulk-plus-spikes spectrum) and show
    # that the unfolding-free ratio test still recovers GOE correctly.
    rng = np.random.default_rng(0)
    n = 200
    ev = _sample_ensemble_eigs("GOE", n, rng)
    ev_contaminated = np.concatenate([ev, ev.max() * np.array([5.0, 8.0, 12.0])])

    r_emp = spacing_ratios(ev_contaminated)
    result = two_sample_ratio_test(r_emp, n_matrix=n, n_draws=200, rng=rng)
    print("=== outlier-contaminated GOE spectrum ===")
    print(summarize(result))
    print(f"\nGround truth was GOE. Verdict by <r>: {result['closest_by_mean']} "
          f"({'correct' if result['closest_by_mean'] == 'GOE' else 'INCORRECT'})")

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Luis Guillermo Papiri Melendez

"""
multiseed_r.py
===============

A single run of <r> is not a result -- it's one draw from a distribution
with unknown variance. This harness trains KINESIS-NCA and the baseline across
multiple seeds, computes <r> for each, and reports Delta<r> = <r>_KINESIS -
<r>_baseline with a mean and standard deviation across seeds.

Usage:
    from multiseed_r import multiseed_r
    stats = multiseed_r(build_kinesis_fn, build_baseline_fn, train_fn, analyze_fn,
                         seeds=[0, 1, 2, 3, 4])

Interpretation:
    If |mean(Delta<r>)| is within ~1-2 sigma of the seed-to-seed std, the
    honest conclusion is that the two architectures are statistically
    indistinguishable on this metric -- report that, don't round up to
    "confirmed."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class MultiSeedResult:
    seeds: list
    r_kinesis: list
    r_baseline: list
    delta_r: list

    @property
    def delta_mean(self) -> float:
        return float(np.mean(self.delta_r))

    @property
    def delta_std(self) -> float:
        return float(np.std(self.delta_r, ddof=1)) if len(self.delta_r) > 1 else float("nan")

    def is_significant(self, n_sigma: float = 2.0) -> bool:
        if len(self.delta_r) < 2 or self.delta_std != self.delta_std:  # NaN check
            return False
        return abs(self.delta_mean) > n_sigma * self.delta_std / np.sqrt(len(self.delta_r))

    def summary(self) -> str:
        lines = [f"Delta<r> across {len(self.seeds)} seeds:"]
        for s, rn, rb, d in zip(self.seeds, self.r_kinesis, self.r_baseline, self.delta_r):
            lines.append(f"  seed={s}: r_kinesis={rn:.4f}  r_baseline={rb:.4f}  delta={d:+.4f}")
        lines.append(f"  mean(Delta<r>) = {self.delta_mean:+.4f}  std = {self.delta_std:.4f}")
        verdict = ("statistically significant difference"
                   if self.is_significant() else
                   "NOT distinguishable from noise -- report as a null result")
        lines.append(f"  verdict: {verdict}")
        return "\n".join(lines)


def multiseed_r(
    build_kinesis_fn: Callable[[int], object],
    build_baseline_fn: Callable[[int], object],
    train_fn: Callable[[object, int], None],
    analyze_fn: Callable[[object], float],
    seeds: list[int],
) -> MultiSeedResult:
    """
    build_*_fn(seed) -> model
    train_fn(model, seed) -> trains model in place
    analyze_fn(model) -> returns pooled <r> for that model's hidden layer

    Kept generic/injectable rather than hardcoded to KINESIS-NCA's exact training
    loop so it can be reused for the sample-efficiency and spectral-geometry
    follow-up experiments without modification.
    """
    r_kinesis, r_baseline = [], []
    for seed in seeds:
        kinesis = build_kinesis_fn(seed)
        base = build_baseline_fn(seed)
        train_fn(kinesis, seed)
        train_fn(base, seed)
        r_kinesis.append(analyze_fn(kinesis))
        r_baseline.append(analyze_fn(base))

    delta = [n - b for n, b in zip(r_kinesis, r_baseline)]
    return MultiSeedResult(seeds=list(seeds), r_kinesis=r_kinesis, r_baseline=r_baseline, delta_r=delta)

# KINESIS-NCA — Neuromorphic Cognitive Architecture

**A neuromorphic architecture built from three biological subsystems (microtubule cellular automata, astrocytic homeostasis, locus coeruleus gain control), paired with a falsifiable random-matrix-theory prediction about the biological substrate it's modeled on.**

[![CI](https://github.com/REPLACE_ME/kinesis-nca/actions/workflows/ci.yml/badge.svg)](https://github.com/REPLACE_ME/kinesis-nca/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/code-Apache%202.0-blue.svg)](LICENSE)
[![Papers: CC BY 4.0](https://img.shields.io/badge/papers-CC%20BY%204.0-lightgrey.svg)](LICENSE)

**Author:** Luis Guillermo Papiri Melendez, Independent Researcher

> **Naming note:** this architecture was originally written up under the name "N-BIO" (see `papers/N-BIO_arXiv_preprint.pdf`, unchanged). The repository and codebase are branded **KINESIS-NCA** instead, solely to avoid a naming collision with an unrelated, pre-existing project also called N-BIO. The paper's title, content, and results are not altered — only the software/repo name differs from the paper name. See `NOTICE` for the full statement.

---

## What's in this repo

Two connected research threads, both aimed at the same question — does biological detail below the neuron actually buy you anything computationally, and can that claim be tested?

| | Claim | Status |
|---|---|---|
| **KINESIS-NCA (V4)** (`src/kinesis/`) | A neuromorphic MLP variant with dendritic gating outperforms a parameter-comparable classical MLP under input corruption. | Reported result, code included, independently reproducible via `experiments/train_robustness.py`. |
| **Spectral diagnostics** (`src/spectral/`) | KINESIS-NCA's hidden-layer representations show random-matrix-theory level repulsion (GOE-class, anti-Poisson). Whether dendritic gating *increases* that repulsion relative to baseline is currently a **null result** — see below. | Rigorously tested, null result reported honestly, multi-seed validation pending. |
| **Tubulin-GUE Conjecture** (`papers/Tubulin_GUE_Conjecture_Papiri_2026.pdf`) | Excitonic state-transition intervals in microtubule tryptophan networks follow GUE nearest-neighbor spacing statistics (⟨r⟩ = 0.603), shifting toward Poisson (⟨r⟩ = 0.386) under anesthesia. | Untested, falsifiable, fully specified experimental protocol. Confirmed literature gap (2003–2026). |

The link between them: KINESIS-NCA's microtubule layer is a *computational* model inspired by the same biological substrate the Tubulin-GUE conjecture makes a *wet-lab* prediction about. If the conjecture is validated, it gives a precise statistical target (GUE spacing statistics, not Poisson or GOE) that a biologically faithful sub-neuronal computing layer should reproduce.

---

## Why this matters for life sciences

This isn't "AI inspired by biology" in the loose branding sense. Specifically:

- **Astrocytes** make up roughly half of human brain cells and are absent from virtually every standard ANN. KINESIS-NCA's `AstroUnit` implements calcium-wave homeostatic gating with the same temporal-inertia dynamics as real perisynaptic astrocyte processes, and — without being programmed to do so — exhibits the same suppression-then-recovery trajectory observed biologically (see Table 3 in the paper, reproduced in `experiments/train_robustness.py`).
- **Microtubules** are modeled as a 13-protofilament cellular automaton, matching the real structural protofilament count in tubulin lattices, with a state-transition rule inspired by Hameroff–Penrose sub-neuronal information processing.
- **The Tubulin-GUE Conjecture** is a genuinely novel, falsifiable prediction: no published work (2003–2026) has applied Wigner-surmise / RMT spacing statistics to biological event-timing intervals at the molecular scale. It comes with sharp numerical falsification criteria (§7 of the paper) and a fully specified experimental protocol using time-correlated single-photon counting on tryptophan fluorescence — this is a real wet-lab test someone could run, not a metaphor.

---

## Results, reported honestly

### 1. Noise robustness (real result)

KINESIS-NCA (V4) vs. a parameter-comparable classical MLP (`218,058` vs. `448,372` params, same depth, same optimizer/epochs), evaluated under zero-mean Gaussian noise on MNIST:

| σ (noise) | Baseline | KINESIS-NCA (V4) | Advantage |
|---|---|---|---|
| 0.0 (clean) | 97.07% | 97.71% | +0.64% |
| 1.0 | 93.59% | 94.44% | +0.85% |
| 2.0 (severe) | 78.16% | 80.08% | **+1.92%** |
| 3.0 (extreme) | 59.78% | 60.65% | +0.87% |

Reproduce with `python experiments/train_robustness.py`. An ablation (`Table 5` in the paper) shows this depends on correct astrocyte calibration: γ=1.5 (V3) *degrades* robustness below baseline via over-suppression — a documented failure mode, not swept under the rug.

### 2. Spectral level statistics (null result, reported as one)

Do KINESIS-NCA's hidden representations show *more* random-matrix-theory level repulsion than a classical baseline's? The honest answer right now is **no measurable difference**:

```
KINESIS-NCA (V4) : <r> = 0.5245 -> GOE-class
Baseline         : <r> = 0.5188 -> GOE-class
Delta<r> = +0.0057   (within observed run-to-run noise, ~0.01-0.02)
```

Both networks show genuine level repulsion relative to Poisson (spacing ratio ⟨r⟩ ≈ 0.52, far from the Poisson value of 0.386) — that part is real and reproducible. But the *difference* attributable to dendritic gating is not yet distinguishable from seed-to-seed noise. `src/spectral/multiseed_r.py` exists specifically to make that call properly once run across multiple seeds. This repo reports the null result rather than rounding it up, because that's the standard the rest of the pipeline is held to.

**Why ⟨r⟩ and not the more common KS-on-unfolded-spacings test?** Trained-network covariance spectra are a Marchenko–Pastur bulk plus a few outlier eigenvalues. Polynomial unfolding distorts around those outliers badly enough that KS-vs-unfolded-spacings can favor Poisson even when ⟨r⟩ is nowhere near the Poisson value — this actually happened during development (see `src/spectral/ratio_test.py`, which demonstrates the failure mode on synthetic outlier-contaminated data and the unfolding-free fix). The pipeline reports both statistics and flags disagreement automatically.

### 3. Tubulin-GUE Conjecture (untested prediction)

| Statistic | GUE (physiological) | Poisson (anesthetic) |
|---|---|---|
| Mean spacing ratio ⟨r⟩ | 0.603 ± 0.02 | 0.386 ± 0.02 |
| Brody parameter *q* | ~2.0 | ~0.0 |
| Spacing variance | 0.180 | 1.000 |

Full falsification criteria, minimum sample sizes (≥500 switching events for initial discrimination, ≥2,000 for crossover characterization), and the complete experimental protocol are in the paper. This is included in the repo as the connective tissue between the computational architecture and an actual biological research question.

---

## Architecture

```
Input (784)
   │
   ▼
┌─────────────────────────────┐
│ MicrotubuleLayer (CBD)      │  13-protofilament cellular automaton,
│  784 → 256                  │  lattice-stability weight gating,
└──────────────┬──────────────┘  LC attention modulation
               ▼
┌─────────────────────────────┐
│ AstroUnit (RCH)             │  calcium-wave homeostatic gate,
│  calcium memory, τ=0.95     │  tonic-inhibition floor (gate_min=0.5)
└──────────────┬──────────────┘
               ▼
             ReLU
               │
               ▼
┌─────────────────────────────┐
│ MicrotubuleLayer (CBD)      │
│  256 → 64                   │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ AstroUnit (RCH)             │
└──────────────┬──────────────┘
               ▼
             ReLU
               │
               ▼
        Linear(64 → 10)
```

A global scalar Locus Coeruleus signal (default 0.7, "alert/attentive") modulates both `MicrotubuleLayer`s simultaneously.

---

## Repo layout

```
src/
  kinesis/
    model.py             KinesisV4: MicrotubuleLayer, AstroUnit, full network
    baseline.py          Parameter-comparable classical MLP
  spectral/
    spectral_level_stats.py   Rigorous RMT diagnostic (mean-centering, unfolding,
                               KS vs Poisson/GOE/GUE, ratio statistic, MC calibration,
                               self-test against synthetic ground truth)
    ratio_test.py              Unfolding-free two-sample test (resolves the
                               GOE-vs-Poisson ambiguity that unfolded-KS gets wrong
                               on outlier-contaminated spectra)
    multiseed_r.py             Multi-seed harness for Delta<r> confidence intervals
experiments/
  train_robustness.py    Reproduces the noise-robustness benchmark (Table 4)
  run_level_analysis.py  Trains both models, runs the full spectral pipeline,
                         reports Delta<r> with automatic disagreement flagging
papers/
  N-BIO_arXiv_preprint.pdf           (original paper; see naming note above)
  Tubulin_GUE_Conjecture_Papiri_2026.pdf
```

## Quickstart

```bash
pip install -r requirements.txt

# Self-test: pipeline correctly identifies Poisson/GOE/GUE from synthetic ground truth
python src/spectral/spectral_level_stats.py

# Architecture sanity check: forward pass + exact parameter-count match to the paper
python src/kinesis/model.py

# Full noise-robustness benchmark (downloads MNIST on first run)
python experiments/train_robustness.py --epochs 5

# Full spectral level-statistics comparison
python experiments/run_level_analysis.py --epochs 5 --pooled-chunks 8
```

CI (`.github/workflows/ci.yml`) runs the self-test and architecture check on every push.

---

## What's *not* claimed here

- KINESIS-NCA's biological mechanisms are **algorithmic analogs**, not simulations of actual ion channel or cytoskeletal biophysics. The correspondence to real astrocyte/microtubule dynamics is structural (same qualitative dynamics, same order-of-magnitude protofilament count), not a claim of biophysical equivalence.
- The Δ⟨r⟩ spectral-geometry result between KINESIS-NCA and baseline is a **null result pending multi-seed confirmation** — it is not evidence that dendritic gating changes representational geometry, and this repo doesn't present it as such.
- The Tubulin-GUE Conjecture is **untested**. It is included because it is falsifiable, precisely specified, and occupies a genuine gap in the literature — not because it has been confirmed.
- Orch-OR / quantum-consciousness framing is **not** part of this repo's empirical claims. Where it appears in the source papers as motivating context (Hameroff–Penrose), it's cited as the historical origin of the microtubule-computation hypothesis, not as a validated mechanism.
- **KINESIS-NCA is a software rename only.** No claim is made that this repository is affiliated with, derived from, or supersedes any other project named N-BIO or KINESIS elsewhere. The rename exists purely to avoid namespace collision.

## Reproducing / extending

Planned next steps (see `src/spectral/multiseed_r.py` and the paper's discussion section):
1. Multi-seed Δ⟨r⟩ confidence intervals across ≥5 seeds.
2. Spectral geometry metrics (participation ratio, effective rank, stable rank) — computable from the same covariance matrices already produced by `run_level_analysis.py`.
3. Sample-efficiency curves (KINESIS-NCA vs. baseline on varying MNIST fractions) as the sharper falsifiable test of whether dendritic gating buys genuine low-data advantage, as opposed to noise robustness alone.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

Code: Apache License 2.0 (explicit patent grant + defensive termination — relevant given the DG-FeFET hardware cross-referencing in the architecture paper; see [`NOTICE`](NOTICE) for the IP-boundary statement). Papers: CC BY 4.0. See [`LICENSE`](LICENSE).

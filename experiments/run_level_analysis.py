# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Luis Guillermo Papiri Melendez

"""
run_level_analysis.py
======================

Trains KINESIS-NCA (V4) and the classical baseline on MNIST, collects hidden-layer
activations via forward hooks from matched-width layers, and runs the full
spectral level-statistics pipeline on both -- reporting whichever
architecture actually shows more level repulsion, honestly, including a
null result if that's what the data says.

Analysis point: first hidden layer, width 256, post-nonlinearity.
    KINESIS-NCA (V4)  -> d1 output (after AstroUnit gate + ReLU)
    Baseline  -> net[1] output (after first ReLU)

Run:
    python experiments/run_level_analysis.py --epochs 5 --pooled-chunks 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "spectral"))
from kinesis import KinesisV4, BaselineMLP, apply_microtubule_rescue  # noqa: E402
import spectral_level_stats as sls  # noqa: E402
import ratio_test as rt  # noqa: E402


def load_data(batch_size: int = 64, data_root: str = "./data"):
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(data_root, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(data_root, train=False, download=True, transform=tfm)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=1000, shuffle=False),
    )


def train(model, loader, device, epochs, is_kinesis):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, correct, n = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            if is_kinesis:
                grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
                apply_microtubule_rescue(model.d1, grad_norm)
                apply_microtubule_rescue(model.d2, grad_norm)
            opt.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        print(f"  epoch {epoch}: loss={total_loss / n:.4f} acc={100 * correct / n:.2f}%")


@torch.no_grad()
def collect_activations(model, layer, loader, device, max_samples: int = 5000) -> np.ndarray:
    acts = []
    hook_handle = None

    def hook(_module, _inp, out):
        acts.append(out.detach().cpu().numpy())

    hook_handle = layer.register_forward_hook(hook)
    model.eval()
    seen = 0
    for x, _ in loader:
        x = x.to(device)
        model(x)
        seen += x.size(0)
        if seen >= max_samples:
            break
    hook_handle.remove()
    return np.concatenate(acts, axis=0)[:max_samples]


def analyze(model, layer, name, loader, device, max_samples, pooled_chunks, n_mc):
    A = collect_activations(model, layer, loader, device, max_samples=max_samples)
    print(f"  [{name}] collected activations: {A.shape}")

    C = sls.covariance_from_activations(A)
    rep = sls.report_from_covariance(C, monte_carlo=True, n_mc=n_mc)
    print(f"\n  --- {name}: single covariance (MC-calibrated) ---")
    print(sls.summary(rep))

    S, R = [], []
    for ch in np.array_split(A, pooled_chunks):
        if ch.shape[0] < ch.shape[1] * 2:
            continue
        ev = sls.clean_covariance_eigs(sls.covariance_from_activations(ch))
        S.append(sls.unfold_spectrum(ev))
        R.append(sls.spacing_ratios(ev))

    if S:
        s_pool = np.concatenate(S)
        r_pool = np.concatenate(R)
        ks = sls._ks_against_all(s_pool)
        r_mean = r_pool.mean()
        r_close = min(sls.R_MEAN, key=lambda k: abs(sls.R_MEAN[k] - r_mean))
        print(f"\n  --- {name}: pooled ({pooled_chunks} chunks, n={s_pool.size} spacings) ---")
        for nm in ("Poisson", "GOE", "GUE"):
            print(f"    KS vs {nm:8s}: D={ks[nm]['D']:.4f}  p={ks[nm]['p']:.4f}")
        print(f"    <r>={r_mean:.4f}  -> nearest {r_close}")

        # Unfolding-free two-sample resolution -- run whenever KS-verdict and
        # <r>-verdict disagree, since that disagreement is a known artifact
        # of polynomial unfolding on MP-bulk-plus-spikes spectra.
        ks_closest = min(ks, key=lambda k: ks[k]["D"])
        if ks_closest != r_close:
            print(f"    [!] KS-unfolded verdict ({ks_closest}) disagrees with <r> verdict ({r_close}).")
            print(f"        Running unfolding-free two-sample ratio test to resolve...")
            resolved = rt.two_sample_ratio_test(r_pool, n_matrix=int(np.mean([len(ch) for ch in np.array_split(A, pooled_chunks)])))
            print("    " + rt.summarize(resolved).replace("\n", "\n    "))

        rep["pooled_r"] = float(r_mean)
        rep["pooled_r_closest"] = r_close
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=5000)
    ap.add_argument("--pooled-chunks", type=int, default=8)
    ap.add_argument("--n-mc", type=int, default=200)
    ap.add_argument("--data-root", type=str, default="./data")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = load_data(data_root=args.data_root)

    print("=== training KINESIS-NCA (V4) ===")
    kinesis = KinesisV4().to(device)
    train(kinesis, train_loader, device, args.epochs, is_kinesis=True)

    print("\n=== training classical baseline ===")
    base = BaselineMLP().to(device)
    train(base, train_loader, device, args.epochs, is_kinesis=False)

    rep_n = analyze(kinesis, kinesis.d1, "KINESIS-NCA (V4) d1", test_loader, device,
                     args.max_samples, args.pooled_chunks, args.n_mc)
    rep_b = analyze(base, base.net[1], "Baseline relu1", test_loader, device,
                     args.max_samples, args.pooled_chunks, args.n_mc)

    print("\n" + "=" * 70)
    print("  SIDE BY SIDE")
    print("=" * 70)
    rn = rep_n.get("pooled_r", rep_n["r_mean"])
    rb = rep_b.get("pooled_r", rep_b["r_mean"])
    print(f"  KINESIS-NCA: <r>={rn:.4f} -> {rep_n.get('pooled_r_closest', rep_n['r_closest'])}")
    print(f"  Baseline   : <r>={rb:.4f} -> {rep_b.get('pooled_r_closest', rep_b['r_closest'])}")
    print(f"  Delta<r> (KINESIS-NCA - Baseline) = {rn - rb:+.4f}")
    print("  Run this across multiple seeds (see src/spectral/multiseed_r.py) before")
    print("  asserting the difference is real -- single-run |Delta<r>| ~ 0.01-0.02")
    print("  is within observed run-to-run noise on this pipeline.")


if __name__ == "__main__":
    main()

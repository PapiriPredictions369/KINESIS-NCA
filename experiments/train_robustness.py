# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Luis Guillermo Papiri Melendez

"""
train_robustness.py
====================

Reproduces the noise-robustness benchmark methodology from the N-BIO paper
(Section 4, Table 4): train N-BIO V4 and the classical baseline on MNIST,
then evaluate both under zero-mean Gaussian noise at sigma in
{0.0, 0.5, 1.0, 1.5, 2.0, 3.0} applied independently to each test image.

Run:
    python experiments/train_robustness.py --epochs 5 --seed 0

This is provided so the paper's numbers are independently reproducible,
not asserted. If your run doesn't match Table 4 closely, that's useful
information -- report the discrepancy, don't paper over it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kinesis import KinesisV4, BaselineMLP, count_parameters, apply_microtubule_rescue  # noqa: E402

NOISE_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def load_data(batch_size: int = 64, data_root: str = "./data"):
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(data_root, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(data_root, train=False, download=True, transform=tfm)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=256, shuffle=False),
    )


def train(model: nn.Module, loader: DataLoader, device: str, epochs: int = 5, is_kinesis: bool = False):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, correct, n = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if is_kinesis:
                logits = model(x)
            else:
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

        msg = f"  epoch {epoch}: loss={total_loss / n:.4f} acc={100 * correct / n:.2f}%"
        if is_kinesis:
            msg += f"  gates={model.gate_state()}"
        print(msg)


@torch.no_grad()
def evaluate_with_noise(model: nn.Module, loader: DataLoader, device: str, sigma: float) -> float:
    model.eval()
    correct, n = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if sigma > 0:
            x = x + torch.randn_like(x) * sigma
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return 100 * correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", type=str, default="./data")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_loader, test_loader = load_data(data_root=args.data_root)

    print("\n=== training KINESIS-NCA (V4) ===")
    kinesis = KinesisV4(lc_signal=0.7)
    print(f"parameters: {count_parameters(kinesis):,}")
    train(kinesis, train_loader, device, epochs=args.epochs, is_kinesis=True)

    print("\n=== training classical baseline ===")
    base = BaselineMLP()
    print(f"parameters: {count_parameters(base):,}")
    train(base, train_loader, device, epochs=args.epochs, is_kinesis=False)

    print("\n" + "=" * 70)
    print("  ROBUSTNESS BENCHMARK (reproducing Table 4 methodology)")
    print("=" * 70)
    print(f"  {'sigma':>6} | {'baseline %':>10} | {'KINESIS %':>10} | {'advantage':>10}")
    for sigma in NOISE_LEVELS:
        acc_base = evaluate_with_noise(base.to(device), test_loader, device, sigma)
        acc_kinesis = evaluate_with_noise(kinesis.to(device), test_loader, device, sigma)
        print(f"  {sigma:>6.1f} | {acc_base:>10.2f} | {acc_kinesis:>10.2f} | {acc_kinesis - acc_base:>+9.2f}%")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Luis Guillermo Papiri Melendez

"""
KINESIS-NCA: Neuromorphic Cognitive Architecture
====================================================

Implements the three coupled biological subsystems described in
Papiri Melendez (2026), "N-BIO: A Neuromorphic Bio-Integrated Architecture
Unifying Microtubule Cellular Automata, Astrocytic Homeostatic Gating,
and Locus Coeruleus Modulation for Robust Neuromorphic Computing"
(see papers/N-BIO_arXiv_preprint.pdf for the original paper -- the
codebase is branded KINESIS-NCA rather than N-BIO to avoid a naming
collision with an unrelated, pre-existing project of the same name).

  1. MicrotubuleLayer (CBD) -- 13-protofilament cellular automaton +
     lattice-stability weight gating, modulated by a global LC signal.
  2. AstroUnit (RCH)        -- calcium-wave homeostatic gate with
     temporal inertia (tau) and a tonic-inhibition floor (gate_min).
  3. Locus Coeruleus (LC)   -- scalar attention/arousal signal shared
     across all MicrotubuleLayers.

Hyperparameters below match Appendix A of the paper exactly (V4,
calibrated). V3's gamma=1.5 is kept as ASTRO_GAMMA_V3 for the ablation
script since it is a documented failure mode (over-suppression), not a
config anyone should train with by default.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Calibrated V4 hyperparameters (Appendix A, original N-BIO arXiv preprint --
# see papers/N-BIO_arXiv_preprint.pdf)
# ---------------------------------------------------------------------------
N_PROTOFILAMENTS = 13        # canonical microtubule protofilament count
LC_FACTOR = 0.5               # norepinephrine sensitivity coefficient
ASTRO_ALPHA = 0.1             # astrocyte sensitivity to baseline activity
ASTRO_BETA = 0.2              # astrocyte sensitivity to prediction error
ASTRO_GAMMA_V4 = 0.5          # calibrated suppression slope
ASTRO_GAMMA_V3 = 1.5          # documented failure mode (over-suppression)
ASTRO_THRESHOLD = 0.8         # homeostatic setpoint
CALCIUM_TAU = 0.95            # calcium-wave temporal inertia
GATE_MIN = 0.5                # tonic inhibition floor
LATTICE_CLAMP_MAX = 2.0       # GTP-cap-analog stability ceiling
PROTO_INIT_STD = 0.1          # low-amplitude initial tubulin activations
WEIGHT_INIT_STD = 0.01


class MicrotubuleLayer(nn.Module):
    """
    Sub-neuronal computation layer (CBD).

    Forward pass implements, in order (matching Appendix B pseudocode
    exactly -- proto_filters are per-INPUT vectors, not per-output
    tensors, so each protofilament yields one scalar pattern per sample
    that gates the whole output row, not a separate value per output
    unit):

      CBD-1  dynamic_stability = lattice_stability * (1 + lc_factor * attn)
      CBD-2  effective_weights = weights * sigmoid(dynamic_stability)
      CBD-3  sigma_i = sigmoid(x @ proto_filter_i),   i in {1..13}   [x: (B,in), proto_filter_i: (in,) -> (B,)]
      CBD-4  C(i, t+1) = clamp(|C(i-1,t) - C(i,t)| + C(i+1,t), 0, 1)
      CBD-5  pattern = sum_i softmax(proto_combine)_i * C(i)         [-> (B, 1)]
      CBD-6  output = (x @ effective_weights) * pattern              [(B,out) * (B,1) broadcast]

    Rescue (CBD-7/8) is applied externally by the training loop, since it
    depends on the loss gradient magnitude and must run after backward().
    See `apply_microtubule_rescue` below.
    """

    def __init__(self, in_features: int, out_features: int,
                 n_proto: int = N_PROTOFILAMENTS, lc_factor: float = LC_FACTOR):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_proto = n_proto
        self.lc_factor = lc_factor

        self.weights = nn.Parameter(torch.randn(in_features, out_features) * WEIGHT_INIT_STD)
        # lattice_stability starts near 0 -> sigmoid(0) = 0.5 effective gate
        self.lattice_stability = nn.Parameter(torch.zeros(in_features, out_features))

        # 13 protofilament projection vectors, each in_features -> scalar.
        self.proto_filters = nn.Parameter(torch.randn(n_proto, in_features) * PROTO_INIT_STD)
        self.proto_combine = nn.Parameter(torch.zeros(n_proto))  # softmax'd combination weights

    def forward(self, x: torch.Tensor, attention_signal: float | torch.Tensor = 0.5) -> torch.Tensor:
        # CBD-1, CBD-2: lattice stability gated by LC attention signal
        dyn_stab = self.lattice_stability * (1.0 + self.lc_factor * attention_signal)
        eff_w = self.weights * torch.sigmoid(dyn_stab)

        # CBD-3: 13 parallel protofilament activations, shape (B, n_proto)
        acts = torch.sigmoid(x @ self.proto_filters.T)  # (B, in) @ (in, n_proto) -> (B, n_proto)

        # CBD-4: XOR/OR cellular-automaton propagation along the protofilament axis.
        # Synchronous update: each new column only reads the previous timestep's
        # neighbor columns, matching the paper's rule exactly.
        acts_next = acts.clone()
        for i in range(1, self.n_proto - 1):
            xor = (acts[:, i - 1] - acts[:, i]).abs()
            acts_next[:, i] = torch.clamp(xor + acts[:, i + 1], 0.0, 1.0)
        acts = acts_next

        # CBD-5: softmax-weighted combination across protofilaments -> one scalar pattern per sample
        combine_weights = F.softmax(self.proto_combine, dim=0)  # (n_proto,)
        pattern = (acts * combine_weights).sum(dim=1, keepdim=True)  # (B, 1)

        # CBD-6: effective linear transform, multiplicatively gated by the CA pattern (broadcasts over out_features)
        return (x @ eff_w) * pattern


@torch.no_grad()
def apply_microtubule_rescue(layer: MicrotubuleLayer, loss_grad_norm: float) -> None:
    """
    CBD-7/8: GTP-cap-analog rescue. Call once per optimizer step, after
    backward(), passing the norm of the loss gradient (or any scalar proxy
    for instability you want lattice_stability to react to).
    """
    layer.lattice_stability.add_(layer.lc_factor * abs(loss_grad_norm))
    layer.lattice_stability.clamp_(0.0, LATTICE_CLAMP_MAX)


class AstroUnit(nn.Module):
    """
    Glial homeostatic gate (RCH).

    RCH-1  stress = alpha * mean(|x|) + beta * mean(|error|)
    RCH-2  calcium(t) = tau * calcium(t-1) + (1 - tau) * stress(t)
    RCH-3  gate = max(gate_min, 1 - gamma * (calcium - threshold))  if calcium > threshold
    RCH-4  gate = 1.0                                                otherwise
    RCH-5  output = x * gate

    `calcium` is a persistent (non-learnable) buffer -- this is a state
    variable, not a trained weight, matching Table 2 of the paper
    ("0 learnable parameters" for every AstroUnit).
    """

    def __init__(self, alpha: float = ASTRO_ALPHA, beta: float = ASTRO_BETA,
                 gamma: float = ASTRO_GAMMA_V4, threshold: float = ASTRO_THRESHOLD,
                 tau: float = CALCIUM_TAU, gate_min: float = GATE_MIN):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.threshold, self.tau, self.gate_min = threshold, tau, gate_min
        self.register_buffer("calcium", torch.tensor(0.0))
        self.last_gate: float = 1.0  # exposed for logging/homeostasis tracking

    def forward(self, x: torch.Tensor, error: torch.Tensor | None = None):
        stress = self.alpha * x.abs().mean()
        if error is not None:
            stress = stress + self.beta * error.abs().mean()

        self.calcium = self.tau * self.calcium + (1.0 - self.tau) * stress.detach()

        if self.calcium.item() > self.threshold:
            gate = max(self.gate_min, 1.0 - self.gamma * (self.calcium.item() - self.threshold))
        else:
            gate = 1.0

        self.last_gate = gate
        return x * gate, gate


class KinesisV4(nn.Module):
    """
    Full KINESIS-NCA (V4 generation) network for 28x28 MNIST-shaped input.

    Architecture (Table 2 of the original N-BIO paper):
      784 -> MicrotubuleLayer(256) -> AstroUnit -> ReLU
          -> MicrotubuleLayer(64)  -> AstroUnit -> ReLU
          -> Linear(10)

    LC attention signal defaults to 0.7 ("alert/attentive" per Table 1).
    """

    def __init__(self, lc_signal: float = 0.7):
        super().__init__()
        self.lc_signal = lc_signal
        self.d1 = MicrotubuleLayer(784, 256)
        self.astro1 = AstroUnit()
        self.d2 = MicrotubuleLayer(256, 64)
        self.astro2 = AstroUnit()
        self.out = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor, error: torch.Tensor | None = None) -> torch.Tensor:
        x = x.view(x.size(0), -1)

        h1 = self.d1(x, attention_signal=self.lc_signal)
        h1, _ = self.astro1(h1, error=error)
        h1 = F.relu(h1)

        h2 = self.d2(h1, attention_signal=self.lc_signal)
        h2, _ = self.astro2(h2, error=error)
        h2 = F.relu(h2)

        return self.out(h2)

    def gate_state(self) -> dict:
        """Snapshot of both astrocyte gates -- for reproducing Table 3."""
        return {"gate1": self.astro1.last_gate, "gate2": self.astro2.last_gate}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = KinesisV4()
    x = torch.randn(8, 1, 28, 28)
    y = m(x)
    print(f"KINESIS-NCA (V4) output shape: {tuple(y.shape)}")
    print(f"KINESIS-NCA (V4) trainable parameters: {count_parameters(m):,}")
    print(f"Gate state after forward: {m.gate_state()}")

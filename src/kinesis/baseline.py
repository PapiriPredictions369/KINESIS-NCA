# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Luis Guillermo Papiri Melendez

"""
Classical baseline: standard McCulloch-Pitts MLP, matched in depth to
KINESIS-NCA (784 -> 256 -> 64 -> 10, ReLU activations, no biological
mechanisms).

This is the network every KINESIS-NCA comparison in the paper and in
this repo is run against. Parameter count differs from KinesisV4
(218,058 vs 448,372) because the protofilament tensors add parameters
without adding depth; see the README for why this comparison is still
meaningful (same depth, same optimizer, same training budget) and
where a stricter param-matched control would need a wider baseline.
"""

from __future__ import annotations

import torch.nn as nn


class BaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

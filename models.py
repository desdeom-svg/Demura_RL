# -*- coding: utf-8 -*-
"""Core RL model definitions shared by simulation and real-world training."""

from __future__ import annotations

import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TrainConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResBlock(nn.Module):
    """Simple residual block for dense spatial prediction."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="replicate"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.conv(x))


class Actor(nn.Module):
    """Predict prior correction gain maps."""

    def __init__(self):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, padding_mode="replicate"),
            nn.ReLU(),
        )
        self.res = nn.Sequential(ResBlock(32), ResBlock(32), ResBlock(32))
        self.exit = nn.Conv2d(32, 1, 3, padding=1, padding_mode="replicate")
        self.output_smoother = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        nn.init.uniform_(self.exit.weight, -1e-4, 1e-4)
        nn.init.constant_(self.exit.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.entry(x)
        feat = self.res(feat)
        network_out = self.exit(feat)
        return self.output_smoother(network_out)


class Critic(nn.Module):
    """Estimate scalar Q-values from state-action pairs."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, padding_mode="replicate"),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ReplayBuffer:
    """CPU-backed replay buffer."""

    def __init__(self, capacity: int = 5000):
        self.buf = deque(maxlen=capacity)

    def push(self, s: torch.Tensor, a: torch.Tensor, r: float, ns: torch.Tensor) -> None:
        self.buf.append((s.cpu(), a.cpu(), r, ns.cpu()))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns = zip(*batch)
        return (
            torch.cat(s).to(device),
            torch.cat(a).to(device),
            torch.tensor(r).float().unsqueeze(1).to(device),
            torch.cat(ns).to(device),
        )

    def clear(self) -> None:
        self.buf.clear()

    def __len__(self) -> int:
        return len(self.buf)

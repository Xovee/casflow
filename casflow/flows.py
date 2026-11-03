"""Invertible planar flows with their exact log-Jacobian determinants."""

import torch
from torch import nn
from torch.nn import functional as F


class PlanarFlow(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dimension) * 0.05)
        self.u = nn.Parameter(torch.randn(dimension) * 0.05)
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, z):
        inner = self.w @ self.u
        # w.T @ u_hat > -1 guarantees invertibility for tanh.
        u_hat = self.u + (F.softplus(inner) - 1 - inner) * self.w / self.w.square().sum()
        activation = torch.tanh(z @ self.w + self.b)
        log_det = torch.log((1 + (1 - activation.square()) * (self.w @ u_hat)).clamp_min(1e-8))
        return z + activation.unsqueeze(-1) * u_hat, log_det


class PlanarFlows(nn.Module):
    def __init__(self, dimension, count=8):
        super().__init__()
        self.layers = nn.ModuleList(PlanarFlow(dimension) for _ in range(count))

    def forward(self, z):
        log_det = z.new_zeros(z.shape[0])
        for layer in self.layers:
            z, increment = layer(z)
            log_det = log_det + increment
        return z, log_det


def flow_kl(mean, log_variance, initial, transformed, log_det):
    """KL(q_K || N(0,I)), using the analytic base KL to reduce variance."""
    base_kl = 0.5 * (mean.square() + log_variance.exp() - log_variance - 1).sum(-1)
    prior_change = 0.5 * (transformed.square() - initial.square()).sum(-1)
    return (base_kl + prior_change - log_det) / mean.shape[-1]

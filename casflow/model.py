"""Node VAE, cascade VAE, planar flows, and a temporal prediction branch."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .flows import PlanarFlows, flow_kl


def sample(mean, log_variance, generator):
    noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
    return mean + (0.5 * log_variance).exp() * noise


def sequence_mean(values, mask):
    return ((values * mask).sum(1) / mask.sum(1)).mean()


class CasFlow(nn.Module):
    def __init__(self, input_dim=80, latent_dim=64, hidden_dim=128, n_flows=8):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2 * latent_dim)
        )
        self.node_decoder = nn.Linear(latent_dim, input_dim)
        self.cascade_encoder = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.cascade_posterior = nn.Linear(hidden_dim, 2 * latent_dim)
        self.flows = PlanarFlows(latent_dim, n_flows)
        self.cascade_decoder = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.cascade_reconstruction = nn.Linear(hidden_dim, latent_dim)
        self.temporal_first = nn.GRU(input_dim, 2 * hidden_dim, batch_first=True, bidirectional=True)
        self.temporal_second = nn.GRU(4 * hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + 2 * hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus()
        )

    def forward(self, x, lengths, *, generator=None, regularize=False, samples=1):
        mask = torch.arange(x.shape[1], device=x.device)[None] < lengths.to(x.device)[:, None]
        x = self.input_norm(x) * mask.unsqueeze(-1)

        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        temporal, _ = self.temporal_first(packed_x)
        _, temporal_hidden = self.temporal_second(temporal)
        temporal = torch.cat((temporal_hidden[-2], temporal_hidden[-1]), -1)

        node_mean, node_log_variance = self.node_encoder(x).chunk(2, -1)
        node_mean = node_mean.repeat(samples, 1, 1)
        node_log_variance = node_log_variance.repeat(samples, 1, 1)
        node_z = sample(node_mean, node_log_variance, generator)
        sampled_lengths = lengths.repeat(samples)
        packed_nodes = pack_padded_sequence(node_z, sampled_lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.cascade_encoder(packed_nodes)
        mean, log_variance = self.cascade_posterior(hidden[-1]).chunk(2, -1)
        initial = sample(mean, log_variance, generator)
        transformed, log_det = self.flows(initial)

        prediction = self.predictor(torch.cat((transformed, temporal.repeat(samples, 1)), -1))
        # Average posterior predictions in the space optimized by the loss.
        prediction = prediction.log1p().reshape(samples, len(x)).mean(0).expm1()

        terms = {}
        if regularize:
            node_reconstruction = self.node_decoder(node_z)
            repeated = transformed[:, None].expand(-1, x.shape[1], -1)
            packed_repeated = pack_padded_sequence(
                repeated, sampled_lengths, batch_first=True, enforce_sorted=False
            )
            decoded, _ = self.cascade_decoder(packed_repeated)
            decoded, _ = pad_packed_sequence(decoded, batch_first=True, total_length=x.shape[1])
            cascade_reconstruction = self.cascade_reconstruction(decoded)
            sampled_mask = mask.repeat(samples, 1)
            terms = {
                "node_reconstruction": sequence_mean(
                    (node_reconstruction - x.repeat(samples, 1, 1)).square().mean(-1), sampled_mask
                ),
                "node_kl": sequence_mean(
                    0.5 * (node_mean.square() + node_log_variance.exp() - node_log_variance - 1).mean(-1),
                    sampled_mask,
                ),
                "cascade_reconstruction": sequence_mean(
                    (cascade_reconstruction - node_z).square().mean(-1), sampled_mask
                ),
                "flow_kl": flow_kl(mean, log_variance, initial, transformed, log_det).mean(),
            }
        return prediction, terms

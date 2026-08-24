"""Checkpoint-compatible conditional VAE used for composition generation.

PyTorch remains optional at package-import time.  The generation tool reports
a typed missing-prerequisite result when the ML extra is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - depends on the installed extras
    torch = None
    nn = None


@dataclass(frozen=True)
class VAEConfig:
    composition_dim: int = 89
    condition_dim: int = 14
    hidden_dim: int = 256
    latent_dim: int = 32


if torch is not None and nn is not None:

    class ConditionalVAE(nn.Module):
        """Compact CVAE architecture used by the JARVIS composition model."""

        def __init__(self, config: VAEConfig = VAEConfig()) -> None:
            super().__init__()
            self.config = config
            encoder_in = config.composition_dim + config.condition_dim
            self.encoder = nn.Sequential(
                nn.Linear(encoder_in, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.SiLU(),
            )
            self.mu = nn.Linear(config.hidden_dim, config.latent_dim)
            self.logvar = nn.Linear(config.hidden_dim, config.latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(
                    config.latent_dim + config.condition_dim,
                    config.hidden_dim,
                ),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.composition_dim),
                nn.Softplus(),
            )

        def decode(self, latent, condition):
            fractions = self.decoder(torch.cat([latent, condition], dim=-1))
            return fractions / fractions.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)

        def encode(self, composition, condition):
            hidden = self.encoder(
                torch.cat([composition, condition], dim=-1)
            )
            return self.mu(hidden), self.logvar(hidden)

        @staticmethod
        def reparameterize(mu, logvar):
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std

        def forward(self, composition, condition):
            mu, logvar = self.encode(composition, condition)
            reconstruction = self.decode(
                self.reparameterize(mu, logvar), condition
            )
            return reconstruction, mu, logvar

        def sample(self, condition, count: int = 8):
            expanded = condition.expand(count, -1)
            latent = torch.randn(
                count,
                self.config.latent_dim,
                device=expanded.device,
            )
            return self.decode(latent, expanded)

        @staticmethod
        def loss(reconstruction, target, mu, logvar, beta: float = 0.05):
            reconstruction_loss = -(
                target * reconstruction.clamp_min(1e-8).log()
            ).sum(-1).mean()
            kl = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp()
            )
            return reconstruction_loss + beta * kl

else:

    class ConditionalVAE:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError(
                "PyTorch is required to load the conditional VAE checkpoint"
            )

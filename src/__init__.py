"""
HiLAB: A Hybrid Inverse-Design Framework

ViT-VAE based latent space + Bayesian Optimization for nanophotonic inverse design.
Paper: https://doi.org/10.1002/smtd.202500975
"""

__version__ = "1.0.0"

from .model import ViTVAE, VAELoss, make_loaders_from_arrays_flexible

__all__ = [
    "ViTVAE",
    "VAELoss",
    "make_loaders_from_arrays_flexible",
]

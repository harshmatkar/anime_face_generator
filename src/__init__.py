"""
src package
-----------
Anime-face DCGAN: Generator, Discriminator, and the GANTrainer training loop.
"""

from .model import Generator, Discriminator
from .train import GANTrainer

__all__ = ["Generator", "Discriminator", "GANTrainer"]
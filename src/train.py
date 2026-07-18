"""
train.py
--------
Custom training loop for the anime-face DCGAN, wrapped in a `GANTrainer`
class. Uses tf.GradientTape directly since GANs need two separate
optimizer updates per step (Generator + Discriminator), which doesn't
fit Keras's plain model.fit().
"""

import os
import time

import tensorflow as tf

from .model import Generator, Discriminator
from .dataset import get_dataset
from .utils import ensure_dirs, save_sample_images


class GANTrainer:

    def __init__(
        self,
        data_dir: str,
        image_size: int = 64,
        latent_dim: int = 100,
        batch_size: int = 128,
        feature_maps: int = 64,
        channels: int = 3,
        lr: float = 2e-3,
        beta_1: float = 0.5,
        output_dir: str = "outputs",
    ):
        self.latent_dim = latent_dim
        self.output_dir = output_dir
        self.samples_dir = os.path.join(output_dir, "samples")
        self.ckpt_dir = os.path.join(output_dir, "checkpoints")
        ensure_dirs(self.output_dir, self.samples_dir, self.ckpt_dir)

        # --- models ---
        self.generator = Generator(latent_dim=latent_dim, feature_maps=feature_maps, channels=channels)
        self.discriminator = Discriminator(feature_maps=feature_maps, channels=channels)

        # --- loss + optimizers ---
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1)

        # fixed noise so we can watch the *same* seeds improve over epochs
        self.seed = tf.random.normal([16, latent_dim]) # 16 fixed noises of 100 length so we can see progress

        # --- checkpointing ---
        self.checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            g_optimizer=self.g_optimizer,
            d_optimizer=self.d_optimizer,
        )
        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, self.ckpt_dir, max_to_keep=3)

        # --- data ---
        self.dataset = get_dataset(data_dir, image_size=image_size, batch_size=batch_size)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def generator_loss(self, fake_output):
        return self.loss_fn(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim]) # num_of_samples * Each individual noise vector

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_images = self.generator(noise, training=True)

            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)

            g_loss = self.generator_loss(fake_output)
            d_loss = self.discriminator_loss(real_output, fake_output)

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        return g_loss, d_loss

    def train(self, epochs: int, save_every: int = 5):
        for epoch in range(1, epochs + 1):
            start = time.time()
            g_loss_total, d_loss_total, num_batches = 0.0, 0.0, 0

            for real_images in self.dataset:
                g_loss, d_loss = self.train_step(real_images)
                g_loss_total += g_loss
                d_loss_total += d_loss
                num_batches += 1

            avg_g_loss = g_loss_total / max(num_batches, 1)
            avg_d_loss = d_loss_total / max(num_batches, 1)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"G loss: {avg_g_loss:.4f} | D loss: {avg_d_loss:.4f} | "
                f"Time: {time.time() - start:.1f}s"
            )

            save_sample_images(self.generator, self.seed, epoch, self.samples_dir)

            if epoch % save_every == 0:
                self.ckpt_manager.save()
                print(f"Checkpoint saved at epoch {epoch}")

    def save_models(self, save_dir: str = None):
        """Saves the trained Generator and Discriminator as .keras files."""
        save_dir = save_dir or self.output_dir
        self.generator.save(os.path.join(save_dir, "generator.keras"))
        self.discriminator.save(os.path.join(save_dir, "discriminator.keras"))
        print(f"Models saved to {save_dir}")
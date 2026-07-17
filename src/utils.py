"""
utils.py
--------
Small helper functions shared by the training pipeline.
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt


def ensure_dirs(*dirs: str) -> None:
    """Create any of the given directories if they don't already exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_sample_images(generator: tf.keras.Model, seed: tf.Tensor, epoch: int, output_dir: str) -> str:
    """
    Generates a 4x4 grid of images from a fixed noise seed and saves it
    as a PNG so you can visually track training progress across epochs.
    Returns the path the image was saved to.
    """
    predictions = generator(seed, training=False)
    predictions = (predictions * 127.5 + 127.5) / 255.0  # back to [0, 1]

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i])
        plt.axis("off")

    out_path = os.path.join(output_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(out_path)
    plt.close(fig)
    return out_path
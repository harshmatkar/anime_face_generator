"""
dataset.py
----------
Loads and preprocesses the anime face image dataset for GAN training.

Expected layout (Keras requires images inside a subfolder):
    data/
        anime_faces/
            img_0001.jpg
            img_0002.jpg
            ...
"""

import tensorflow as tf


def get_dataset(data_dir: str, image_size: int = 64, batch_size: int = 128) -> tf.data.Dataset:
    """
    Builds a tf.data.Dataset of images scaled to [-1, 1]
    (to match the Generator's Tanh output).
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
    )

    normalize = lambda x: (tf.cast(x, tf.float32) - 127.5) / 127.5
    ds = ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
"""
model.py
--------
DCGAN architecture (Generator + Discriminator) for generating fake anime
face images, built with TensorFlow / Keras using OOP principles
(Keras Model subclassing API).

Design assumptions:
    - Output/Input image size: 64 x 64 x 3
    - Latent (noise) vector size: 100 (configurable)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model #type: ignore


class BaseGAN(Model):
    """
    Common base class for Generator and Discriminator.
    Holds shared config so both subclasses follow a consistent
    OOP structure (inheritance + polymorphic call()).
    """

    def __init__(self, channels: int = 3, feature_maps: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.feature_maps = feature_maps
        self.net = tf.keras.Sequential()  # to be defined by subclasses

    def call(self, x, training: bool = False):
        return self.net(x, training=training)


class Generator(BaseGAN):
    """
    Maps a latent noise vector (latent_dim,) -> a 64x64xchannels
    fake anime image using transposed convolutions.
    """

    def __init__(self, latent_dim: int = 100, feature_maps: int = 64, channels: int = 3, **kwargs):
        super().__init__(channels=channels, feature_maps=feature_maps, **kwargs)
        self.latent_dim = latent_dim
        fm = feature_maps

        self.net = tf.keras.Sequential([
            layers.Input(shape=(latent_dim,)),

            layers.Dense(4 * 4 * fm * 8, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Reshape((4, 4, fm * 8)),

            self._block(fm * 4, kernel_size=4, stride=2),   # -> (8, 8, fm*4)
            self._block(fm * 2, kernel_size=4, stride=2),   # -> (16, 16, fm*2)
            self._block(fm, kernel_size=4, stride=2),       # -> (32, 32, fm)

            layers.Conv2DTranspose(
                channels, kernel_size=4, strides=2, padding="same", use_bias=False
            ),                                               # -> (64, 64, channels)
            layers.Activation("tanh"),
        ])

    @staticmethod
    def _block(out_channels: int, kernel_size: int, stride: int) -> tf.keras.Sequential:
        return tf.keras.Sequential([
            layers.Conv2DTranspose(
                out_channels, kernel_size, strides=stride, padding="same", use_bias=False
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

    def call(self, z, training: bool = False):
        return self.net(z, training=training)


class Discriminator(BaseGAN):
    """
    Takes a 64x64xchannels image and outputs a single probability
    (real vs fake) using strided convolutions.
    """

    def __init__(self, feature_maps: int = 64, channels: int = 3, **kwargs):
        super().__init__(channels=channels, feature_maps=feature_maps, **kwargs)
        fm = feature_maps

        self.net = tf.keras.Sequential([
            layers.Input(shape=(64, 64, channels)),

            layers.Conv2D(fm, kernel_size=4, strides=2, padding="same", use_bias=False),
            layers.LeakyReLU(0.2),                                   # -> (32, 32, fm)

            self._block(fm * 2, kernel_size=4, stride=2),            # -> (16, 16, fm*2)
            self._block(fm * 4, kernel_size=4, stride=2),            # -> (8, 8, fm*4)
            self._block(fm * 8, kernel_size=4, stride=2),            # -> (4, 4, fm*8)

            layers.Flatten(),
            layers.Dense(1, activation="sigmoid"),
        ])

    @staticmethod
    def _block(out_channels: int, kernel_size: int, stride: int) -> tf.keras.Sequential:
        return tf.keras.Sequential([
            layers.Conv2D(out_channels, kernel_size, strides=stride, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
        ])

    def call(self, img, training: bool = False):
        return self.net(img, training=training)
import tensorflow as tf

from eenx15_19_21_gan.GAN.networks.models.generator import Generator
from eenx15_19_21_gan.GAN.networks.models.layers import *

class WirehedGenerator(Generator):
    def __init__(self, image_size):
        Generator.__init__(self)
        
        self.image_size = image_size

        with tf.variable_scope("generator") as scope:
            self.model = tf.keras.Sequential()
            """Currently input size is 100x100"""

            # Creating the model
            self.model = tf.keras.Sequential()
            self.model.add(tf.keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100, 100, 3)))
            self.model.add(tf.keras.layers.BatchNormalization())  # Normalize the nodes values between 0-1 (instead of 0-255)
            self.model.add(tf.keras.layers.LeakyReLU())           # No negative values

            # Creating 7, 7, channels=256 layer
            self.model.add(tf.keras.layers.Reshape((7, 7, 256)))
            # Simple assert to check resolution and color channels
            assert self.model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

            # Strides=(1, 1) to keep the resolution
            # This layer only changes the color channels from 256 -> 128
            # Kernel size 5x5
            self.model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
            assert self.model.output_shape == (None, 7, 7, 128)
            self.model.add(tf.keras.layers.BatchNormalization())
            self.model.add(tf.keras.layers.LeakyReLU())

            # Strides > 1 to increase image size and (1, 1) for square ratio
            # Color channel=64
            self.model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
            assert self.model.output_shape == (None, 14, 14, 64)
            self.model.add(tf.keras.layers.BatchNormalization())
            self.model.add(tf.keras.layers.LeakyReLU())

            # Stride (2, 2) to double the size
            # Output Color channel=3 for RGB
            self.model.add(tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
            assert self.model.output_shape == (None, 28, 28, 3)


    def __call__(self, image, is_training=False):
        with tf.variable_scope("generator") as scope:
            return self.model(image, is_training) 
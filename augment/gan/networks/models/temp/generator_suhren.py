import tensorflow as tf
import math

from eenx15_19_21_gan.GAN.networks.models.generator import Generator
from eenx15_19_21_gan.GAN.networks.models.layers import *

class SuhrenGenerator(Generator):
    def __init__(self, image_size, mid_size):
        Generator.__init__(self)
        
        assert(util.is_pow2(image_size) and mid_size <= image_size)
        
        self.image_size = image_size

        with tf.variable_scope("generator") as scope:
            self.model = tf.keras.Sequential()
            self.model.add(tf.keras.layers.InputLayer(input_shape=(image_size, image_size, 3)))
            
            a = int(math.log(image_size) / math.log(2))
            b = int(math.log(mid_size) / math.log(2))
            n = a - b

            # Go down in size
            for i in range(n):
                self.model.add(tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same'))
                self.model.add(tf.keras.layers.LeakyReLU(alpha=0.8))
                self.model.add(tf.keras.layers.BatchNormalization())
            
            # Go up in size
            for i in range(n):
                self.model.add(tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same'))
                self.model.add(tf.keras.layers.LeakyReLU(alpha=0.8))
                self.model.add(tf.keras.layers.BatchNormalization())
            
            self.model.add(tf.keras.layers.Dense(3, activation='tanh'))

    def __call__(self, image, is_training=False):
        with tf.variable_scope("generator") as scope:
            return self.model(image, is_training) 
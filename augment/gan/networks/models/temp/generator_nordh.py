import tensorflow as tf
import math

from eenx15_19_21_gan.GAN.networks.models.generator import Generator
from eenx15_19_21_gan.GAN.networks.models.layers import *

class NordhGenerator(Generator):
    def __init__(self, image_size):
        Generator.__init__(self)
        
        self.image_size = image_size

        with tf.variable_scope("generator") as scope:
            self.model = tf.keras.Sequential()
            """ 256x256x3 """
            #self.model.add(tf.keras.layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same', input_shape=(image_size, image_size, 3)))
            """ 128x128x64 """
            #self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
            #self.model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
            """ 64x64x64 """
            #self.model.add(tf.keras.layers.BatchNormalization())
            #self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.model.add(tf.keras.layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same', input_shape=(image_size, image_size, 3)))
            """ 32x32x128 """
            #self.model.add(tf.keras.layers.BatchNormalization())
            self.model.add(tf.keras.layers.LeakyReLU(alpha=0.8))
            self.model.add(tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))

            """ 16x16x256 """
            #self.model.add(tf.keras.layers.BatchNormalization())
            self.model.add(tf.keras.layers.LeakyReLU(alpha=0.8))
            self.model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same'))
            """ 32x32x128 """
            #self.model.add(tf.keras.layers.BatchNormalization())
            self.model.add(tf.keras.layers.LeakyReLU(alpha=0.8))
            self.model.add(tf.keras.layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same'))

            self.model.add(tf.keras.layers.Dense(3, activation='tanh'))

            """ 64x64x128 """
            #self.model.add(tf.keras.layers.BatchNormalization())
            #self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
            #self.model.add(tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'))
            """ 128x128x64 """
            #self.model.add(tf.keras.layers.BatchNormalization())
            #self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
            #self.model.add(tf.keras.layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'))
            """ 256x256x32 """
            #self.model.add(tf.keras.layers.Dense(3, activation='tanh'))


    def __call__(self, image, is_training=False):
        with tf.variable_scope("generator") as scope:
            return self.model(image, is_training) 
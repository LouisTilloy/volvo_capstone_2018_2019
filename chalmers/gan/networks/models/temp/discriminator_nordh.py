import tensorflow as tf

from eenx15_19_21_gan.GAN.networks.models.discriminator import Disctriminator
from eenx15_19_21_gan.GAN.networks.models.layers import *

class NordhDisctriminator(Disctriminator):
    def __init__(self, image_size):
        Disctriminator.__init__(self)
        
        with tf.variable_scope("discriminator") as scope:
            m = tf.keras.Sequential()
            """ 256x256x3 """
            #self.m.add(tf.keras.layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
            """ 128x128x32 """
            #self.m.add(tf.keras.layers.LeakyReLU(alpha=0.2))
            #self.m.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
            """ 64x64x64 """
            #self.m.add(tf.keras.layers.BatchNormalization())
            #self.m.add(tf.keras.layers.LeakyReLU(alpha=0.2))
            m.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', input_shape=(image_size, image_size, 3)))
            """ 32x32x128 """
            m.add(tf.keras.layers.BatchNormalization())
            m.add(tf.keras.layers.LeakyReLU(alpha=0.8))
            m.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
            """ 16x16x256 """
            m.add(tf.keras.layers.BatchNormalization())
            m.add(tf.keras.layers.LeakyReLU(alpha=0.8))
            m.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
            """ 8x8x256 """
            m.add(tf.keras.layers.LeakyReLU(alpha=0.8))

            """ Some fully connected layers"""
            m.add(tf.keras.layers.Flatten())
            #m.add(tf.keras.layers.Dense(16))
            #m.add(tf.keras.layers.Dense(32))
            #m.add(tf.keras.layers.Dense(100))
            #m.add(tf.keras.layers.Dense(200))
            #m.add(tf.keras.layers.Dense(100))
            m.add(tf.keras.layers.Dense(50))
            m.add(tf.keras.layers.Dense(10))
            m.add(tf.keras.layers.Dense(1, name="out"))
            m.add(tf.keras.layers.Softmax(name="out_logits"))

            # Use named layers and create model here in order to get output from both two last layers
            self.model = tf.keras.Model(inputs=m.input, outputs=[m.get_layer("out_logits").output, m.get_layer("out").output])

    def __call__(self, image, reuse=False, is_training=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            return self.model(image, is_training)
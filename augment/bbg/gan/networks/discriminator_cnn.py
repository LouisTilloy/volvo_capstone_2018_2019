import tensorflow as tf

from .discriminator import Discriminator
from .layers import *


class CNNDiscriminator(Discriminator):
    def __init__(self, image_size):
        Discriminator.__init__(self)
        self.fm = 32
        self.image_size = image_size

    def __call__(self, image, reuse=False, is_training=False):
        with tf.variable_scope("discriminator") as scope:
            with tf.variable_scope("cnn") as scope:
                if reuse:
                    scope.reuse_variables()

                #256x256x3
                x = conv2d(image, self.fm, (3, 3), (1, 1), name="d_00_conv")
                x = lrelu(x, 0.2)

                #256x256xfm
                x = conv2d(x, self.fm * 2, (3, 3), (2, 2), name="d_01_conv")
                x = lrelu(x, 0.2)
                x = batch_norm(x, training=is_training, name="d_00_bns")

                #128x128x2fm
                x = conv2d(x, self.fm * 4, (3, 3), (2, 2), name="d_02_conv")
                x = lrelu(x, 0.2)
                x = batch_norm(x, training=is_training, name="d_02_bns")

                #64x64x4fm
                x = conv2d(x, self.fm * 8, (3, 3), (2, 2), name="d_h03_conv")
                x = lrelu(x, 0.2)
                x = batch_norm(x, training=is_training, name="d_03_bns")

                #32x32x8fm
                x = flatten(x)
                x = linear(x, 32, "d_04_lin")

                #32
                x = linear(x, 1, "d_05_lin")

                #1
                return x
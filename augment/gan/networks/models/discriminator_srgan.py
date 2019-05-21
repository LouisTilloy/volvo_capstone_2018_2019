import tensorflow as tf

from .discriminator import Discriminator
from .layers import *


class SRGANDiscriminator(Discriminator):
    def __init__(self, image_size):
        Discriminator.__init__(self)
        assert(image_size == 256)

    def __call__(self, image, reuse=False, is_training=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            x = conv2d(image, 64, (3, 3), (1, 1), name="d_00_conv")
            x = lrelu(x, 0.2)

            x = conv2d(x, 128, (3, 3), (2, 2), name="d_01_conv")
            x = lrelu(x, 0.2)
            x = batch_norm(x, training=is_training, name="d_01_bns")

            # 128x128x128
            x = conv2d(x, 128, (3, 3), (2, 2), name="d_02_conv")
            x = lrelu(x, 0.2)
            x = batch_norm(x, training=is_training, name="d_02_bns")

            # 64x64x128
            x = conv2d(x, 256, (3, 3), (2, 2), name="d_03_conv")
            x = lrelu(x, 0.2)
            x = batch_norm(x, training=is_training, name="d_03_bns")

            # 32x32x128
            x = conv2d(x, 256, (3, 3), (2, 2), name="d_04_conv")
            x = lrelu(x, 0.2)
            x = batch_norm(x, training=is_training, name="d_04_bns")

            # 16x16x256
            x = conv2d(x, 256, (3, 3), (2, 2), name="d_05_conv")
            x = lrelu(x, 0.2)
            x = batch_norm(x, training=is_training, name="d_05_bns")

            # 8x8x256
            x = conv2d(x, 512, (2, 2), (2, 2), name="d_06_conv")
            x = lrelu(x, 0.2)
            x = batch_norm(x, training=is_training, name="d_06_bns")

            # 4x4x512
            x = conv2d(x, 1024, (1, 1), (2, 2), name="d_07_conv")
            x = lrelu(x, 0.2)
            x = batch_norm(x, training=is_training, name="d_07_bns")

            # 2x2x1024
            x = flatten(x)
            x = linear(x, 100, "d_08_lin")

            # 100
            x = linear(x, 10, "d_09_lin")

            # 10
            x = linear(x, 1, "d_10_lin")

            return x
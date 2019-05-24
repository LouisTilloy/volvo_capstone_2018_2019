import tensorflow as tf

from .generator import Generator
from .layers import *


class EvenGenerator(Generator):
    def __init__(self, image_size):
        Generator.__init__(self)
        self.image_size = image_size
        assert(image_size == 256)

    def __call__(self, image, is_training=False):
        with tf.variable_scope("generator"):
            with tf.variable_scope("even"):
                
                x = conv2d(image, 256, (3, 3), (1, 1), name="g_00_conv")
                x = lrelu(x, 0.2)

                x = conv2d(x, 128, (3, 3), (1, 1), name="g_01_conv")
                x = lrelu(x, 0.2)
                x = batch_norm(x, training=is_training, name="g_01_bns")

                x = conv2d(x, 96, (3, 3), (1, 1), name="g_02_conv")
                x = lrelu(x, 0.2)
                x = batch_norm(x, training=is_training, name="g_02_bns")

                x = conv2d(x, 64, (3, 3), (1, 1), name="g_03_conv")
                x = lrelu(x, 0.2)
                x = batch_norm(x, training=is_training, name="g_03_bns")

                x = conv2d(x, 32, (3, 3), (1, 1), name="g_04_conv")
                x = lrelu(x, 0.2)
                x = batch_norm(x, training=is_training, name="g_04_bns")

                x = conv2d(x, 16, (3, 3), (1, 1), name="g_05_conv")
                x = lrelu(x, 0.2)
                x = batch_norm(x, training=is_training, name="g_05_bns")

                x = conv2d(x, 3, (1, 1), (1, 1), name="g_06_conv")

                return tf.nn.sigmoid(x)
import tensorflow as tf

from eenx15_19_21_gan.GAN.networks.models.discriminator import Disctriminator
from eenx15_19_21_gan.GAN.networks.models.layers import *

class CycleGANDisctriminator(Disctriminator):
    def __init__(self, image_size):
        Disctriminator.__init__(self)
        assert(image_size == 256)
        with tf.variable_scope("discriminator") as scope:
            self.bns = [batch_norm(name="d_bn0{}".format(i,)) for i in range(7)]

    def __call__(self, image, reuse=False, is_training=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            """ 256x256x3 """
            x = conv2d(image, 64, (3, 3), (1, 1), name='d_00_conv')
            x = lrelu(x, 0.95)

            x = conv2d(x, 64, (3, 3), (1, 1), name='d_01_conv')
            x = lrelu(x, 0.95)
            x = max_pool(x, (2,2), (2,2), name='d_02_maxpool')
            """ 128x128x64"""
            x = conv2d(image, 64, (3, 3), (1, 1), name='d_10_conv')
            x = lrelu(x, 0.95)

            x = conv2d(x, 64, (3, 3), (1, 1), name='d_11_conv')
            x = lrelu(x, 0.95)
            x = max_pool(x, (2,2), (2,2), name='d_12_maxpool')

            """ 64x64x128 """
            x = conv2d(image, 128, (3, 3), (1, 1), name='d_20_conv')
            x = lrelu(x, 0.95)

            x = conv2d(x, 128, (3, 3), (1, 1), name='d_21_conv')
            x = lrelu(x, 0.95)

            x = conv2d(image, 64, (3, 3), (1, 1), name='d_22_conv')
            x = lrelu(x, 0.95)

            x = conv2d(x, 128, (3, 3), (1, 1), name='d_23_conv')
            x = lrelu(x, 0.95)

            x = max_pool(x, (2,2), (2,2), name='d_24_maxpool')


            """ 32x32x128 """
            x = conv2d(image, 256, (3, 3), (1, 1), name='d_30_conv')
            x = lrelu(x, 0.95)

            x = conv2d(x, 256, (3, 3), (1, 1), name='d_31_conv')
            x = lrelu(x, 0.95)

            x = conv2d(image, 256, (3, 3), (1, 1), name='d_32_conv')
            x = lrelu(x, 0.95)

            x = conv2d(x, 256, (3, 3), (1, 1), name='d_33_conv')
            x = lrelu(x, 0.95)

            x = max_pool(x, (2, 2), (2, 2), name='d_34_maxpool')
            ''' 16x16x256 '''
            x = conv2d(image, 256, (3, 3), (1, 1), name='d_40_conv')
            x = lrelu(x, 0.95)

            x = conv2d(x, 256, (3, 3), (1, 1), name='d_41_conv')
            x = lrelu(x, 0.95)

            x = conv2d(image, 256, (3, 3), (1, 1), name='d_42_conv')
            x = lrelu(x, 0.95)

            x = conv2d(x, 256, (3, 3), (1, 1), name='d_43_conv')
            x = lrelu(x, 0.95)

            x = max_pool(x, (2, 2), (2, 2), name='d_44_maxpool')
            ''' 8x8x256 '''

            x = linear(tf.reshape(x, [-1, 8 * 8 * 256]), 4096, "d_50_lin")
            """4096"""
            x = linear(x, 1000, "d_51_lin")
            """1000"""
            x = linear(x, 1, "d_52_lin")

            return x, tf.nn.softmax(x, name="d_53_lin")

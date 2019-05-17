import tensorflow as tf

from eenx15_19_21_gan.GAN.networks.models.discriminator import Disctriminator
from eenx15_19_21_gan.GAN.networks.models.layers import *

class DIYDisctriminator(Disctriminator):
    def __init__(self, image_size):
        Disctriminator.__init__(self)
        assert(image_size == 256)
        with tf.variable_scope("discriminator") as scope:
            self.bns = [batch_norm(name="d_bn0{}".format(i,)) for i in range(8)]

    def __call__(self, image, reuse=False, is_training=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            x = conv2d(image, 32, (5, 5), (1, 1), name='d_00_conv')
            x = lrelu(x, 0.95)

            x = conv2d(x, 64, (3, 3), (2, 2), name='d_01_conv')
            x = self.bns[0](x, is_training)
            x = lrelu(x, 0.95)
            """ 128x128x128"""
            x = conv2d(x, 64, (3, 3), (2, 2), name='d_02_conv')
            x = self.bns[1](x, is_training)
            x = lrelu(x, 0.95)

            """ 64x64x128 """
            x = conv2d(x, 128, (3, 3), (2, 2), name='d_03_conv')
            x = self.bns[2](x, is_training)
            x = lrelu(x, 0.95)

            """ 32x32x128 """
            x = conv2d(x, 128, (3, 3), (2, 2), name='d_04_conv')
            x = self.bns[3](x, is_training)
            x = lrelu(x, 0.95)
            ''' 16x16x256 '''
            x = conv2d(x, 256, (2, 2), (2, 2), name='d_05_conv')
            x = self.bns[4](x, is_training)
            x = lrelu(x, 0.95)
            ''' 8x8x256 '''
            x = conv2d(x, 256, (2, 2), (2, 2), name='d_06_conv')
            x = self.bns[5](x, is_training)
            x = lrelu(x, 0.95)

            """ 4x4x256 """
            x = conv2d(x, 256, (1, 1), (2, 2), name='d_07_conv')
            x = self.bns[6](x, is_training)
            x = lrelu(x, 0.95)
            """ 2x2x256 """
            x = conv2d(x, 512, (1, 1), (1, 1), name='d_08_conv')
            x = self.bns[7](x, is_training)
            x = lrelu(x, 0.95)

            x = linear(tf.reshape(x, [-1, 2 * 2 * 256]), 84, "d_09_lin")
            """ 8 """
            x = linear(x, 8, "d_10_lin")
            """ 1 """
            x = linear(x, 1, "d_11_lin")

            return x, tf.nn.sigmoid(x)

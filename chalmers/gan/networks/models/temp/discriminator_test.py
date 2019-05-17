import tensorflow as tf

from eenx15_19_21_gan.GAN.networks.models.discriminator import Disctriminator
from eenx15_19_21_gan.GAN.networks.models.layers import *

class TestDisctriminator(Disctriminator):
    def __init__(self, image_size):
        Disctriminator.__init__(self)
        assert(image_size == 256)
        with tf.variable_scope("discriminator") as scope:
            self.bns = [batch_norm(name="d_bn0{}".format(i,)) for i in range(2)]

    def __call__(self, image, reuse=False, is_training=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            
            #256x256x3
            x = lrelu(conv2d(image, 256, (3, 3), (2, 2), name="d_00_conv"), leak=0.8)
            #128x128x3
            x = lrelu(conv2d(image, 128, (3, 3), (2, 2), name="d_01_conv"), leak=0.8)
            #64x64x3
            x = lrelu(conv2d(image, 128, (3, 3), (2, 2), name="d_02_conv"), leak=0.8)
            #32x32x128
            x = lrelu(self.bns[0](conv2d(x, 128, (3, 3), (2, 2), name="d_03_conv"), is_training), leak=0.8)
            #16x16x256
            x = lrelu(self.bns[1](conv2d(x, 128, (3, 3), (2, 2), name="d_04_conv"), is_training), leak=0.8)
            #8x8x256
            x = linear(tf.reshape(x, [-1, 8*8*256]), 32, "d_04_lin")
            #32
            x = linear(x, 16, "d_05_lin")
            #16
            x = linear(x, 1, "d_06_lin")

            return x

            #return self.model(image, is_training)
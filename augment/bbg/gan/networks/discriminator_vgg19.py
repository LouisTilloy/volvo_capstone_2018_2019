import tensorflow as tf

from .discriminator import Discriminator
from .layers import *


# Usually handles 224x224x3 images
class VGG19Discriminator(Discriminator):
    def __init__(self, image_size):
        Discriminator.__init__(self)
        self.image_size = image_size

    def __call__(self, image, reuse=False, is_training=False):
        with tf.variable_scope("discriminator") as scope:
            with tf.variable_scope("vgg19") as scope:
                if reuse:
                    scope.reuse_variables()

                # 256x256x3
                x = conv2d(image, 64, (3, 3), (1, 1), name="vgg_00_conv")
                x = relu(x)
                x = conv2d(x, 64, (3, 3), (1, 1), name="vgg_01_conv")
                x = relu(x)
                x = max_pool(x, (2, 2), (2, 2), name="vgg_02_maxpool")

                # 128x128x64
                x = conv2d(x, 128, (3, 3), (1, 1), name="vgg_03_conv")
                x = relu(x)
                x = conv2d(x, 128, (3, 3), (1, 1), name="vgg_04_conv")
                x = relu(x)
                x = max_pool(x, (2, 2), (2, 2), name="vgg_05_maxpool")


                # 64x64x128
                x = conv2d(x, 256, (3, 3), (1, 1), name="vgg_06_conv")
                x = relu(x)
                x = conv2d(x, 256, (3, 3), (1, 1), name="vgg_07_conv")
                x = relu(x)
                x = conv2d(x, 256, (3, 3), (1, 1), name="vgg_08_conv")
                x = relu(x)
                x = conv2d(x, 256, (3, 3), (1, 1), name="vgg_09_conv")
                x = relu(x)
                x = max_pool(x, (2, 2), (2, 2), name="vgg_10_maxpool")


                #32x32x256
                x = conv2d(x, 512, (3, 3), (1, 1), name="vgg_11_conv")
                x = relu(x)
                x = conv2d(x, 512, (3, 3), (1, 1), name="vgg_12_conv")
                x = relu(x)
                x = conv2d(x, 512, (3, 3), (1, 1), name="vgg_13_conv")
                x = relu(x)
                x = conv2d(x, 512, (3, 3), (1, 1), name="vgg_14_conv")
                x = relu(x)
                x = max_pool(x, (2, 2), (2, 2), name="vgg_15_maxpool")

                # 16x16x512
                x = conv2d(x, 512, (3, 3), (1, 1), name="vgg_16_conv")

                x = relu(x)
                x = conv2d(x, 512, (3, 3), (1, 1), name="vgg_17_conv")
                x = relu(x)
                x = conv2d(x, 512, (3, 3), (1, 1), name="vgg_18_conv")
                x = relu(x)
                x = conv2d(x, 512, (3, 3), (1, 1), name="vgg_19_conv")
                x = relu(x)
                x = max_pool(x, (2, 2), (2, 2), name="vgg_20_maxpool")

                #8x8x256
                x = flatten(x)
                x = linear(x, 1024, name="vgg_21_dense")
                x = dropout(x, rate=0.5, name="vgg_22_dropout")
                x = linear(x, 1024, name="vgg_23_dense")
                x = dropout(x, rate=0.5, name="vgg_24_dropout")
                x = linear(x, 1, name="vgg_25_dense")

                return x
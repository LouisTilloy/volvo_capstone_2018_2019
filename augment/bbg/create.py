import argparse
import tensorflow as tf

from util import generators, discriminators
from gan.gan import GAN


def main():
    g = generators[args.generator](args.size)
    d = discriminators[args.discriminator](args.size)
    gan = GAN(args.name, g, d, image_size=args.size)
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        gan.save(sess)
        gan.save_config()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new GAN model")
    parser.add_argument("name", type=str, help="The name of the model")
    parser.add_argument("generator", type=str, help="The generator model")
    parser.add_argument("discriminator", type=str, help="The discriminator model")
    parser.add_argument("size", type=int, default=256, help="Image size [n x n]")
    args = parser.parse_args()

    main()
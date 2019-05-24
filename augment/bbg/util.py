import os

from gan.gan import GAN
from gan.networks.generator_feedforward import FeedForwardGenerator
from gan.networks.generator_autoenc import AutoEncoderGenerator
from gan.networks.generator_even import EvenGenerator
from gan.networks.discriminator_srgan import SRGANDiscriminator
from gan.networks.discriminator_vgg19 import VGG19Discriminator
from gan.networks.discriminator_cnn import CNNDiscriminator

generators = {
    FeedForwardGenerator.__name__ : FeedForwardGenerator,
    AutoEncoderGenerator.__name__ : AutoEncoderGenerator,
    EvenGenerator.__name__ : EvenGenerator
}
discriminators = {
    SRGANDiscriminator.__name__ : SRGANDiscriminator,
    VGG19Discriminator.__name__ : VGG19Discriminator,
    CNNDiscriminator.__name__ : CNNDiscriminator
}

def load_model(model_name):
    path = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(path, "models", model_name)
    config_path = os.path.join(model_dir, "config.txt")

    config = {}
    with open(config_path, "r") as f:
        for line in f.readlines():
            k = line.split(",")[0].strip()
            v = line.split(",")[1].strip()
            config[k] = v

    c_dim = int(config["c_dim"])
    image_size = int(config["image_size"])
    bbox_weight = float(config["bbox_weight"])
    image_weight = float(config["image_weight"])
    g = generators[config["generator"]](image_size)
    d = discriminators[config["discriminator"]](image_size)
    
    return GAN(model_name, g, d, model_dir, image_size=image_size, c_dim=c_dim, bbox_weight=bbox_weight, image_weight=image_weight)
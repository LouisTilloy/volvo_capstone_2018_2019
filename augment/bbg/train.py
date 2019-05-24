# https://mlnotebook.github.io/post/GAN5/
import os
import argparse
import numpy as np
import tensorflow as tf

from util import load_model

# Disable some TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)

# Options to limit GPU usage
# Fixed the problem: tensorflow/stream_executor/cuda/cuda_dnn.cc:334] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True 
config.intra_op_parallelism_threads = 8

# DEFINE THE FLAGS FOR RUNNING SCRIPT FROM THE TERMINAL
# (ARG1, ARG2, ARG3) = (NAME OF THE FLAG, DEFAULT VALUE, DESCRIPTION)

parser = argparse.ArgumentParser(description="Train the GAN.")

parser.add_argument("model_name", type=str, help="The model.")

parser.add_argument("dataset_input", type=str, help="Input dataset directory or .txt file")

parser.add_argument("dataset_real", type=str, help="Real dataset directory")

parser.add_argument("-d", "--discriminator", type=bool, dest="train_discriminator", help="Only train the discriminator")

parser.add_argument("--epochs", type=int, dest="epochs", default=200, help="Number of epochs to train")

parser.add_argument("--learning-rate", type=float, dest="learning_rate", default=0.0002, help="Learning rate for adam optimiser")

parser.add_argument("--beta1", type=float, dest="beta1", default=0.5, help="Momentum term for adam optimiser")

parser.add_argument("--train-size", type=int, dest="train_size", default=np.inf, help="The number of training images")

parser.add_argument("--batch-size", type=int, dest="batch_size", default=1, help="The batch-size (number of images to train at once)")

parser.add_argument("--output-dir", type=str, dest="output_dir", default="outputs", help="Directory to save the outputs of the model")

parser.add_argument("--input-transform", type=str, dest="input_transform", default="resize", choices=["resize","crop"], help="How to pre-process the input images")

parser.add_argument("--sample-interval", type=int, dest="sample_interval", default=16, help="Number of epochs between samples")

parser.add_argument("--sample-size", type=int, dest="sample_size", default=16, help="Number of samples to generate")

parser.add_argument("--checkpoint-interval", type=int, dest="checkpoint_interval", default=16, help="Number of epochs between checkpoints")

args = parser.parse_args()

def main():
    gan = load_model(args.model_name)

    with tf.Session(config=config) as sess:
        if args.train_discriminator:
            gan.train_discriminator(sess, args)
        else:
            gan.train(sess, args)

        #gan_even.infer(sess, "tests/test_inference", "inference_output", FLAGS)

if __name__ == "__main__":
    main()
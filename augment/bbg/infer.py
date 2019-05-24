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

parser = argparse.ArgumentParser(description="Run inference on the model.")

parser.add_argument("model_name", type=str, help="The model.")

parser.add_argument("dataset_input", type=str, default="", help="Input dataset directory")

parser.add_argument("output_dir", type=str, default="outputs", help="Directory to save the outputs of the model")

parser.add_argument("--batch-size", type=int, dest="batch_size", default=1, help="The batch-size (number of images to train at once)")

parser.add_argument("--input-transform", type=str, dest="input_transform", default="resize", choices=["resize","crop"], help="How to pre-process the input images")

args = parser.parse_args()

def main():
    gan = load_model(args.model_name)
    with tf.Session(config=config) as sess:
        gan.infer(sess, args)

if __name__ == "__main__":
    main()
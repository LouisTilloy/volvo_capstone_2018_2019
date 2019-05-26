import os
import cv2
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

parser = argparse.ArgumentParser(description="Run inference on the model using a camera.")

parser.add_argument("model_name", type=str, help="The model.")

args = parser.parse_args()

def main():
    gan = load_model(args.model_name)
    
    cv2.namedWindow(gan.model_name, cv2.WINDOW_NORMAL)
    vc = cv2.VideoCapture(cv2.CAP_DSHOW)
    
    if vc.isOpened():
        rval, frame_in = vc.read()
    else:
        rval = False
    
    with tf.Session(config=config) as sess:
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        gan.load_checkpoints(sess)

        while rval:
            frame_in = cv2.resize(frame_in, (256, 256))
            frame_out = gan.infer_img(sess, frame_in)
            output = np.concatenate((frame_in, frame_out), axis=1)
            cv2.imshow(gan.model_name, output)
            rval, frame_in = vc.read()
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break
    
    cv2.destroyWindow(gan.model_name)

    

if __name__ == "__main__":
    main()
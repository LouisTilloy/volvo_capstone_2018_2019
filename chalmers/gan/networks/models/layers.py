import numpy as np
import tensorflow as tf
import scipy.misc

def batch_norm(x, training=False, name=None):
    return tf.layers.batch_normalization(x, training=training, name=name)   

def conv2d(x, output_dim, k, s, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [k[0], k[1], x.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(x, w, strides=[1, s[0], s[1], 1], padding="SAME")

        biases = tf.get_variable("biases", [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        return conv   
    
def deconv2d(x, output_shape, k, s, stddev=0.02, name="conv2d_transpose", with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [k[0], k[0], output_shape[-1], x.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(x, w, output_shape=output_shape,
                            strides=[1, s[0], s[1], 1])

        biases = tf.get_variable("biases", [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv    
    
def lrelu(x, leak, name="lrelu"):
    with tf.variable_scope(name):
        return tf.nn.leaky_relu(x, leak, name)

def relu(x, name="relu"):
    with tf.variable_scope(name):
        return tf.nn.relu(x, name)

def dropout(x, rate=0.5, name="dropout"):
    with tf.variable_scope(name):
        return tf.nn.dropout(x, rate=rate)

def linear(x, output_size, name=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = x.get_shape().as_list()

    with tf.variable_scope(name or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(x, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x, matrix) + bias

def flatten(x):
    return tf.layers.flatten(x)

def max_pool(x, k, s, name="MaxPool"):
    with tf.variable_scope(name):
        return tf.nn.max_pool(x, ksize=[1, k[0], k[1], 1], strides=[1, s[0], s[1], 1], padding="SAME", name=name)
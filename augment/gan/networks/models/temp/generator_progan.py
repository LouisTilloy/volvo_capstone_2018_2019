# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

from ..util.Nvidia_altered_functions import *


# ----------------------------------------------------------------------------
# Generator network used in the paper.

def G_paper(
        latents_in,  # First input: Latent vectors [minibatch, latent_size].
        labels_in,  # Second input: Labels [minibatch, label_size].
        num_channels=1,  # Number of output color channels. Overridden based on dataset.
        resolution=32,  # Output resolution. Overridden based on dataset.
        label_size=0,  # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
        fmap_base=8192,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        latent_size=None,  # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
        normalize_latents=True,  # Normalize latent vectors before feeding them to the network?
        use_wscale=True,  # Enable equalized learning rate?
        use_pixelnorm=True,  # Enable pixelwise feature vector normalization?
        pixelnorm_epsilon=1e-8,  # Constant epsilon for pixelwise feature vector normalization.
        use_leakyrelu=True,  # True = leaky ReLU, False = ReLU.
        dtype='float32',  # Data type to use for activations and outputs.
        fused_scale=True,  # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
        structure=None,  # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
        is_template_graph=False,  # True = template graph constructed by the Network class, False = actual evaluation.
        **kwargs):  # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2 ** resolution_log2 and resolution >= 4

    def nf(stage):
        return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

    def PN(x):
        return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x

    if latent_size is None: latent_size = nf(0)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu if use_leakyrelu else tf.nn.relu

    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    combo_in = tf.cast(tf.concat([latents_in, labels_in], axis=1), dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    # Building blocks.
    def block(x, res):  # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2 ** res, 2 ** res)):
            if res == 2:  # 4x4
                if normalize_latents: x = pixel_norm(x, epsilon=pixelnorm_epsilon)
                with tf.variable_scope('Dense'):
                    x = dense(x, fmaps=nf(res - 1) * 16, gain=np.sqrt(2) / 4,
                              use_wscale=use_wscale)  # override gain to match the original Theano implementation
                    x = tf.reshape(x, [-1, nf(res - 1), 4, 4])
                    x = PN(act(apply_bias(x)))
                with tf.variable_scope('Conv'):
                    x = PN(act(apply_bias(conv2d(x, fmaps=nf(res - 1), kernel=3, use_wscale=use_wscale))))
            else:  # 8x8 and up
                if fused_scale:
                    with tf.variable_scope('Conv0_up'):
                        x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=nf(res - 1), kernel=3, use_wscale=use_wscale))))
                else:
                    x = upscale2d(x)
                    with tf.variable_scope('Conv0'):
                        x = PN(act(apply_bias(conv2d(x, fmaps=nf(res - 1), kernel=3, use_wscale=use_wscale))))
                with tf.variable_scope('Conv1'):
                    x = PN(act(apply_bias(conv2d(x, fmaps=nf(res - 1), kernel=3, use_wscale=use_wscale))))
            return x

    def torgb(x, res):  # res = 2..resolution_log2
        lod = resolution_log2 - res
        with tf.variable_scope('ToRGB_lod%d' % lod):
            return apply_bias(conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale))

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        x = block(combo_in, 2)
        images_out = torgb(x, 2)
        for res in range(3, resolution_log2 + 1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = torgb(x, res)
            images_out = upscale2d(images_out)
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out = lerp_clip(img, images_out, lod_in - lod)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(x, res, lod):
            y = block(x, res)
            img = lambda: upscale2d(torgb(y, res), 2 ** lod)
            if res > 2: img = cset(img, (lod_in > lod),
                                   lambda: upscale2d(lerp(torgb(y, res), upscale2d(torgb(x, res - 1)), lod_in - lod),
                                                     2 ** lod))
            if lod > 0: img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))
            return img()

        images_out = grow(combo_in, 2, resolution_log2 - 2)

    assert images_out.dtype == tf.as_dtype(dtype)
    images_out = tf.identity(images_out, name='images_out')
    return images_out


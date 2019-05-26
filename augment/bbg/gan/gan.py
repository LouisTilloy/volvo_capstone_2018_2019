from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf

from . import util


class GAN():
    def __init__(self, model_name, generator, discriminator, output_dir, image_size=64, c_dim=3, bbox_weight=1.0, image_weight=1.0):

        assert(util.is_pow2(image_size) and image_size >= 8)

        self.c_dim = c_dim
        self.image_size = image_size
        self.image_shape = [image_size, image_size, c_dim]
        
        self.is_input_annotations = False

        self.bbox_weight = bbox_weight
        self.image_weight = image_weight

        self.generator = generator
        self.discriminator = discriminator
        
        self.model_name = model_name
        self.output_dir = output_dir
        self.log_dir =  os.path.join(self.output_dir, "logs")
        self.sample_dir = os.path.join(self.output_dir, "samples")
        self.checkpoint_dir =  os.path.join(self.output_dir, "checkpoints")
        self.checkpoint_dir_g = os.path.join(self.checkpoint_dir, self.generator.name())
        self.checkpoint_path_g = os.path.join(self.checkpoint_dir_g, self.generator.name())
        self.checkpoint_dir_d = os.path.join(self.checkpoint_dir, self.discriminator.name())
        self.checkpoint_path_d = os.path.join(self.checkpoint_dir_d, self.discriminator.name())

        self.build_model()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.images_real = tf.placeholder(tf.float32, [None] + self.image_shape, name="images_real")
        self.images_fake = tf.placeholder(tf.float32, [None] + self.image_shape, name="images_fake")
        self.images_input = tf.placeholder(tf.float32, [None] + self.image_shape, name="images_input")
        self.bboxes = tf.placeholder(tf.int32, shape=(None, 5))
        self.batch_size = tf.placeholder(tf.int32, shape=())

        self.images_fake = self.generator(self.images_input, is_training=self.is_training)

        self.D_real = self.discriminator(self.images_real, is_training=self.is_training)
        self.D_fake = self.discriminator(self.images_fake, reuse=True, is_training=self.is_training)

        self.d_real_sum = tf.summary.histogram("d_real", self.D_real)
        self.d_fake_sum = tf.summary.histogram("d_fake", self.D_fake)

        str_input = "input/"
        str_fake = "fake/"
        str_real = "real/"

        with tf.name_scope(None):
            with tf.name_scope(str_input):
                self.input_sum = tf.summary.image("input_image", self.images_input)
            with tf.name_scope(str_fake):
                self.fake_sum = tf.summary.image("fake_image", self.images_fake)
            with tf.name_scope(str_real):
                self.real_sum = tf.summary.image("real_image", self.images_real)

        self.d_correct_real = tf.math.reduce_sum(tf.math.round(tf.nn.sigmoid(self.D_real)))
        self.d_correct_fake = tf.math.reduce_sum(tf.math.round(1.0 - tf.nn.sigmoid(self.D_fake)))

        d_loss_real_logits = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(self.D_real))
        d_loss_fake_logits = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake))
        g_loss_image_logits = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake))

        self.d_loss_real = tf.reduce_mean(d_loss_real_logits)
        self.d_loss_fake = tf.reduce_mean(d_loss_fake_logits)
        self.g_loss_image = self.image_weight * tf.reduce_mean(g_loss_image_logits)
        
        def loop_body(i, losses, bb_imgs_in, bb_imgs_out):
            img_in = self.images_input[i]
            img_out = self.images_fake[i]
            bb = self.bboxes[i]
            img_in_cropped = tf.image.crop_to_bounding_box(img_in, bb[1], bb[0], bb[3] - bb[1], bb[2] - bb[0])
            img_out_cropped = tf.image.crop_to_bounding_box(img_out, bb[1], bb[0], bb[3] - bb[1], bb[2] - bb[0])
            loss = tf.losses.mean_squared_error(labels=img_in_cropped, predictions=img_out_cropped)
            losses = tf.concat([losses, [loss]], 0)
            size = tf.shape(bb_imgs_in)[1]
            img_in_cropped_resized = tf.image.resize_images(img_in_cropped, size=[size, size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)
            img_out_cropped_resized = tf.image.resize_images(img_out_cropped, size=[size, size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)
            bb_imgs_in = tf.concat([bb_imgs_in, [img_in_cropped_resized]], 0)
            bb_imgs_out = tf.concat([bb_imgs_out, [img_out_cropped_resized]], 0)
            i = tf.add(i, 1)
            return [i, losses, bb_imgs_in, bb_imgs_out]

        def g_loss_function():
            # https://stackoverflow.com/questions/41233462/tensorflow-while-loop-dealing-with-lists
            losses = tf.Variable([])
            bb_size = 64
            bb_imgs_in = tf.zeros([0, bb_size, bb_size, self.c_dim])
            bb_imgs_out = tf.zeros([0, bb_size, bb_size, self.c_dim])
            i = tf.constant(0)
            loop_cond = lambda i, losses, bb_imgs_in, bb_imgs_out: tf.less(i, self.batch_size)
            [i, losses, bb_imgs_in, bb_imgs_out] = tf.while_loop(loop_cond, loop_body, [i, losses, bb_imgs_in, bb_imgs_out], shape_invariants=[i.get_shape(), tf.TensorShape([None]), tf.TensorShape([None, bb_size, bb_size, self.c_dim]), tf.TensorShape([None, bb_size, bb_size, self.c_dim])])
            # Adding a name scope ensures logical grouping of the layers in the graph.
            with tf.name_scope(None):
                with tf.name_scope(str_input):
                    self.images_input_cropped = bb_imgs_in
                    self.input_cropped_sum = tf.summary.image("input_crop", self.images_input_cropped)
                with tf.name_scope(str_fake):
                    self.images_output_cropped = bb_imgs_out
                    self.fake_cropped_sum = tf.summary.image("fake_crop", self.images_output_cropped)

            return tf.reduce_mean(losses)

        self.use_bboxes = tf.placeholder(tf.bool, name="use_bboxes")
        self.g_loss_bbox = self.bbox_weight * tf.cond(self.use_bboxes, g_loss_function, lambda: 0.0)

        self.g_loss = self.g_loss_image + self.g_loss_bbox
        self.d_loss = self.d_loss_real + self.d_loss_fake
        
        self.g_loss_bbox_sum = tf.summary.scalar("g_loss_bbox", self.g_loss_bbox)
        self.g_loss_image_sum = tf.summary.scalar("g_loss_image", self.g_loss_image)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")

        self.saver_generator = tf.train.Saver(max_to_keep=1, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator"))
        self.saver_discriminator = tf.train.Saver(max_to_keep=1, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator"))
    
    def train(self, sess, config):
        sample_width = (int)(math.sqrt(config.sample_size))
        data_real = util.get_paths(config.dataset_real)

        if os.path.isdir(config.dataset_input):  
            print("Input dataset is a directory")
            self.is_input_annotations = False
            data_input = util.get_paths(config.dataset_input)
            dict_input = {path : [0]*5 for path in data_input}
        else:
            print("Input dataset is an annotations .txt file")  
            self.is_input_annotations = True
            dict_input = util.load_data(config.dataset_input)
            data_input = list(dict_input.keys())
            dict_input = {key : val[0] for key, val in dict_input.items()}
            dict_input = util.resize_bounding_boxes(dict_input, self.image_size)
            
        np.random.shuffle(data_input)
        np.random.shuffle(data_real)

        assert(len(data_input) > 0 and len(data_real) > 0)
        
        # https://github.com/tensorflow/tensorflow/issues/16455
        # https://git.alphagriffin.com/O.P.P/FiryZeplin-deep-learning/src/ee5b04a3ff360d8276d881e41265d7d45f47ccc9/batch-norm/Batch_Normalization_Solutions.ipynb
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
        
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        
        self.g_sum = tf.summary.merge([
            self.d_fake_sum,
            self.input_sum,
            self.fake_sum,
            self.real_sum,
            self.d_loss_fake_sum,
            self.g_loss_sum,
            self.g_loss_image_sum])

        if (self.is_input_annotations):
            self.g_sum = tf.summary.merge([
            self.g_sum,
            self.g_loss_bbox_sum,
            self.input_cropped_sum,
            self.fake_cropped_sum])

        self.d_sum = tf.summary.merge([
            self.d_real_sum,
            self.d_loss_real_sum,
            self.d_loss_sum])

        self.writer = tf.summary.FileWriter(self.log_dir, sess.graph)
        
        sample_files_input = data_input[0:config.sample_size]
        sample_input = [util.get_image(sample_file, self.image_size, input_transform=config.input_transform) for sample_file in sample_files_input]
        sample_images_input = np.array(sample_input).astype(np.float32)
        
        sample_files_real = data_real[0:config.sample_size]
        sample_real = [util.get_image(sample_file, self.image_size, input_transform=config.input_transform) for sample_file in sample_files_real]
        sample_images_real = np.array(sample_real).astype(np.float32)
        
        sample_bboxes = np.array([dict_input[key] for key in sample_files_input]).astype(np.int32)

        counter = 1
        start_time = time.time()
        
        self.load_checkpoints(sess)
        
        for epoch in range(config.epochs):
            batch_idxs = min(len(data_input), config.train_size) // config.batch_size

            for idx in range(batch_idxs):
                batch_files_input = data_input[idx*config.batch_size:(idx+1)*config.batch_size]
                batch_files_real = data_real[idx*config.batch_size:(idx+1)*config.batch_size]
                batch_input = [util.get_image(batch_file, self.image_size, input_transform=config.input_transform) for batch_file in batch_files_input]
                batch_real = [util.get_image(batch_file, self.image_size, input_transform=config.input_transform) for batch_file in batch_files_real]
                batch_images_input = np.array(batch_input).astype(np.float32)
                batch_images_real = np.array(batch_real).astype(np.float32)
                
                batch_bboxes = np.array([dict_input[key] for key in batch_files_input]).astype(np.int32)
                
                #update D network
                _, summary_str = sess.run([d_optim, self.d_sum],
                    feed_dict={
                        self.batch_size : config.batch_size,
                        self.images_real : batch_images_real,
                        self.images_input : batch_images_input,
                        self.is_training: True})

                self.writer.add_summary(summary_str, counter)
                
                #update G network
                _, summary_str = sess.run([g_optim, self.g_sum],
                    feed_dict={
                        self.batch_size : config.batch_size,
                        self.images_input : batch_images_input,
                        self.images_real : batch_images_real,
                        self.bboxes : batch_bboxes,
                        self.is_training : True,
                        self.use_bboxes : self.is_input_annotations})

                self.writer.add_summary(summary_str, counter)
                
                #run g_optim twice to make sure that d_loss does not go to zero (not in the paper)
                _, summary_str = sess.run([g_optim, self.g_sum],
                    feed_dict={
                        self.batch_size : config.batch_size,
                        self.images_input : batch_images_input,
                        self.images_real : batch_images_real,
                        self.bboxes : batch_bboxes,
                        self.is_training: True,
                        self.use_bboxes : self.is_input_annotations})

                self.writer.add_summary(summary_str, counter)
                
                errD_fake = self.d_loss_fake.eval({
                    self.batch_size : config.batch_size,
                    self.images_input : batch_images_input,
                    self.is_training : False})

                errD_real = self.d_loss_real.eval({
                    self.batch_size : config.batch_size,
                    self.images_real : batch_images_real,
                    self.is_training : False})

                errG = self.g_loss.eval({
                    self.batch_size : config.batch_size,
                    self.images_input : batch_images_input,
                    self.bboxes : batch_bboxes,
                    self.is_training : False,
                    self.use_bboxes : self.is_input_annotations})
            
                counter += 1
                print("Epoch [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
                        epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))
                
                if np.mod(counter, config.sample_interval) == 0:
                    samples = np.zeros((config.sample_size, self.image_size, self.image_size, self.c_dim))
                    g_losses = 0.0
                    d_losses = 0.0
                    for i in range(config.sample_size):
                        sample, d_loss, g_loss = sess.run([self.images_fake, self.d_loss, self.g_loss], 
                                feed_dict={
                                    self.batch_size : 1,
                                    self.images_input : [sample_images_input[i]],
                                    self.bboxes : [sample_bboxes[i]],
                                    self.images_real : [sample_images_real[i]],
                                    self.is_training : True,
                                    self.use_bboxes : self.is_input_annotations})
                        d_losses += d_loss
                        g_losses += g_loss
                        samples[i] = sample[0]
                    
                    util.save_mosaic(samples, [sample_width, sample_width], os.path.join(self.sample_dir, "train_{:02d}-{:04d}.png".format(epoch, idx)))
                    print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss / config.batch_size, g_loss / config.batch_size))
                    
                if np.mod(counter, config.checkpoint_interval) == 0:
                    self.save(sess, step=counter)
    
    def train_discriminator(self, sess, config):
        dir_fake = config.dataset_input
        dir_real = config.dataset_real
        paths_fake = util.get_paths(dir_fake)
        paths_real = util.get_paths(dir_real)

        assert(len(paths_fake) > 0 and len(paths_real) > 0)

        np.random.shuffle(paths_fake)
        np.random.shuffle(paths_real)

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.d_sum = tf.summary.merge([
            self.fake_sum,
            self.real_sum,
            self.d_fake_sum,
            self.d_real_sum,
            self.d_loss_real_sum,
            self.d_loss_fake_sum,
            self.d_loss_sum])

        self.writer = tf.summary.FileWriter(self.log_dir, sess.graph)

        counter = 1
        start_time = time.time()
        
        self.load_checkpoints(sess)

        for epoch in range(config.epochs):
            batch_idxs = min(len(paths_fake), config.train_size) // config.batch_size

            for idx in range(batch_idxs):
                batch_files_fake = paths_fake[idx * config.batch_size : (idx + 1) * config.batch_size]
                batch_files_real = paths_real[idx * config.batch_size : (idx + 1) * config.batch_size]
                batch_fake = [util.get_image(f, self.image_size, input_transform=config.input_transform) for f in batch_files_fake]
                batch_real = [util.get_image(f, self.image_size, input_transform=config.input_transform) for f in batch_files_real]
                batch_images_fake = np.array(batch_fake).astype(np.float32)
                batch_images_real = np.array(batch_real).astype(np.float32)

                errD, errD_fake, errD_real, d_correct_fake, d_correct_real, summary_str = sess.run(
                    [d_optim, self.d_loss_fake, self.d_loss_real, self.d_correct_fake, self.d_correct_real, self.d_sum],
                    feed_dict={
                        self.batch_size : config.batch_size,
                        self.images_fake : batch_images_fake,
                        self.images_real : batch_images_real,
                        self.is_training: True})
                
                self.writer.add_summary(summary_str, counter)
                
                counter += 1

                print("Epoch [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, d_loss_fake: {:.8f}, d_loss_real: {:.8f}, d_correct_fake: {:.0f}/{:.0f}, d_correct_real: {:.0f}/{:.0f}".format(
                        epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errD_fake, errD_real, d_correct_fake, config.batch_size, d_correct_real, config.batch_size))
                    
                if np.mod(counter, config.checkpoint_interval) == 2:
                    self.save(sess, step=counter)

    def infer(self, sess, config):
        if os.path.isdir(config.dataset_input):  
            print("Input dataset is a directory")
            paths_in = util.get_paths(config.dataset_input)
        else:
            print("Input dataset is an annotations .txt file")
            dict_input = util.load_data(config.dataset_input)
            paths_in = list(dict_input.keys())

        dir_out = config.output_dir

        print("Running inference:")
        print("Input: %s" % config.dataset_input)
        print("Output: %s" % dir_out)

        n = len(paths_in)
        assert(n > 0)
        print("Number of images: %i" % n)
        
        paths_out = []
        input_dir = os.path.dirname(config.dataset_input)
        for path in paths_in:
            rel_path = os.path.relpath(path, input_dir)
            path_list = os.path.normpath(rel_path).split(os.sep)
            path_list[0] += "_bbg"
            new_path = os.path.join(*path_list)
            new_path = os.path.join(dir_out, new_path)
            paths_out.append(new_path)
        
        print("Initializing model...")
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        
        self.load_checkpoints(sess)
        
        n_batches = n // config.batch_size
        n_extra = n % config.batch_size

        ia = 0
        ib = n_extra if n_extra > 0 else config.batch_size

        start_time = time.time()

        for i in range(n_batches):
            b_pth_out = paths_out[ia:ib]
            b_pth_in = paths_in[ia:ib]
            b_img = [util.get_image(f, self.image_size, input_transform=config.input_transform) for f in b_pth_in]
            b_img = np.array(b_img).astype(np.float32)
            
            output, = sess.run([self.images_fake], feed_dict={
                self.batch_size : config.batch_size,
                self.images_input : b_img,
                self.is_training : True})
            
            util.save_images(output, b_pth_out)
            print("Images [{:4d}/{:4d}]".format(ib, n))

            ia = ib
            ib += config.batch_size

        print("Inference completed in {:4.4f}".format(time.time() - start_time))
    
    def infer_img(self, sess, img):
        b_img = [np.array(img).astype(np.float32)]
        
        output, = sess.run([self.images_fake], feed_dict={
            self.batch_size : 1,
            self.images_input : b_img,
            self.is_training : True})
        
        return util.norm2byte(output[0])
            
    def save_config(self):
        config = {
            "model_name" : self.model_name,
            "generator" : self.generator.name(),
            "discriminator" : self.discriminator.name(),
            "image_size" : self.image_size,
            "c_dim" : self.c_dim,
            "bbox_weight" : self.bbox_weight,
            "image_weight" : self.image_weight
        }
        config_path = os.path.join("models", self.model_name, "config.txt")
        with open(config_path, "w") as f:
            for k, v in config.items():
                f.write(f"{k},{v}\n")

    def save(self, sess, step=0):
        os.makedirs(self.checkpoint_dir_g, exist_ok=True)
        self.saver_generator.save(sess, self.checkpoint_path_g, global_step=step)
        os.makedirs(self.checkpoint_dir_d, exist_ok=True)
        self.saver_discriminator.save(sess, self.checkpoint_path_d, global_step=step)

    def load_checkpoints(self, sess):
        print("[*] Reading generator checkpoints for model %s..." % self.generator.name())
        ckpt_g = tf.train.get_checkpoint_state(self.checkpoint_dir_g)

        if ckpt_g and ckpt_g.model_checkpoint_path:
            self.saver_generator.restore(sess, ckpt_g.model_checkpoint_path)
            print("An existing generator model for %s was found - delete the directory or specify a new one with --checkpoint_dir" % self.generator.name())
        else:
            print("No generator model for %s found - initializing a new one" % self.generator.name())

        print("[*] Reading discriminator checkpoints for model %s..." % self.discriminator.name())
        ckpt_d = tf.train.get_checkpoint_state(self.checkpoint_dir_d)

        if ckpt_d and ckpt_d.model_checkpoint_path:
            self.saver_discriminator.restore(sess, ckpt_d.model_checkpoint_path)
            print("An existing discriminator model for %s was found - delete the directory or specify a new one with --checkpoint_dir" % self.discriminator.name())
        else:
            print("No discriminator model for %s found - initializing a new one" % self.discriminator.name())
            
    def name(self):
        return self.__class__.__name__
# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/model.py
#   + License: MIT
# [2016-08-05] Modifications for Completion: Brandon Amos (http://bamos.github.io)
#   + Source: https://github.com/bamos/dcgan-completion.tensorflow/blob/master/model.py
#   + License: MIT
# [2016-09] Modifications for Projection and Face Math: Irmak Sirer (http://www.irmaksirer.com)
#   + License: MIT

from __future__ import division
import os
import time
import errno
from glob import glob
import tensorflow as tf
from six.moves import xrange

from ops import *
from utils import *

class DCGAN(object):
    def __init__(self, sess, image_size=64, is_crop=False,
                 batch_size=64, sample_size=64,
                 z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3,
                 checkpoint_dir=None, lam=0.1):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = [image_size, image_size, 3]

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.lam = lam

        self.c_dim = 3

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.checkpoint_dir = checkpoint_dir
        self.build_model()

        self.model_name = "DCGAN.model"

    def build_model(self):
        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')
        self.sample_images= tf.placeholder(
            tf.float32, [None] + self.image_shape, name='sample_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.histogram_summary("z", self.z)

        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(self.images)

        self.sampler = self.sampler(self.z)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = tf.histogram_summary("d", self.D)
        self.d__sum = tf.histogram_summary("d_", self.D_)
        self.G_sum = tf.image_summary("G", self.G)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits,
                                                    tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_,
                                                    tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_,
                                                    tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=1)

        # Projection
        # l1 norm
        # self.full_contextual_loss = tf.reduce_sum(
        #     tf.contrib.layers.flatten(tf.abs(self.G - self.images)), 1)
        # l2 norm
        self.full_contextual_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(tf.square(self.G - self.images)), 1)
        self.perceptual_loss = self.g_loss
        self.project_loss = self.full_contextual_loss + self.lam*self.perceptual_loss
        self.grad_project_loss = tf.gradients(self.project_loss, self.z)
        
        # Completion 
        self.mask = tf.placeholder(tf.float32, [None] + self.image_shape, name='mask')
        self.masked_contextual_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(tf.mul(self.mask, self.G) - tf.mul(self.mask, self.images))), 1)
        self.complete_loss = self.masked_contextual_loss + self.lam*self.perceptual_loss
        self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)

    def train(self, config):
        data = glob(os.path.join(config.dataset, "*.png"))
        #np.random.shuffle(data)
        assert(len(data) > 0)

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        tf.initialize_all_variables().run()

        self.g_sum = tf.merge_summary(
            [self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            data = glob(os.path.join(config.dataset, "*.png"))
            batch_idxs = min(len(data), config.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                         for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                            .astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                    feed_dict={ self.images: batch_images, self.z: batch_z })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict={ self.z: batch_z })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict={ self.z: batch_z })
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.images: batch_images})
                errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 1:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss, self.g_loss],
                        feed_dict={self.z: sample_z, self.images: sample_images}
                    )
                    save_images(samples, [8, 8],
                                './samples/train_{:02d}_{:05d}.png'.format(epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)


    def find_optimal_z(self,
                       loss_function,
                       loss_gradient,
                       images,
                       mask_for_all_images=None,
                       current_batch_size = None,
                       init_z_hats=None,
                       n_iterations=1000,
                       learning_rate=0.01,
                       momentum=0.9,
                       output_every_nth_step=50,
                       projected_img_output_dir=None,
                       z_vectors_output_dir=None):
        # initialize (sizes, starting vectors and starting velocity)
        if current_batch_size is None:
            current_batch_size = self.batch_size
        if init_z_hats is None:
            init_z_hats = np.random.uniform(-1, 1, size=(self.batch_size,
                                                         self.z_dim))
        z_hats = init_z_hats
        v = 0

        # steps of gradient descent with momentum
        for step in xrange(n_iterations):

            feed_dict = {
                self.z: z_hats,
                self.images: images,
            }
            if mask_for_all_images is not None:
                feed_dict[self.mask] = mask_for_all_images

            run = [loss_function, loss_gradient, self.G]
            loss, gradient, generated_images = self.sess.run(run, feed_dict=feed_dict)

            # update velocity
            v_prev = np.copy(v)
            v = momentum * v_prev - learning_rate * gradient[0]

            # update our current best z_hat vectors
            z_hats += -momentum * v_prev + (1 + momentum) * v

            # if this update pushed us out of the (-1,1) domain of z,
            # clip it to stay in. This makes it "projected" gradient descent
            # check here for a concise explanation:
            # http://math.stackexchange.com/questions/571068/what-is-the-difference-between-projected-gradient-descent-and-ordinary-gradient
            z_hats = np.clip(z_hats, -1, 1)

            # log the progress and save the intermediary z_hats and generated images
            # we get along the way during optimization
            if step % output_every_nth_step == 0 or step == (n_iterations-1):
                loss_value = np.mean(loss[0:self.batch_size])
                msg = "Searching z, step {}. Loss = {}".format(step, loss_value)
                print(msg)

                if projected_img_output_dir:
                    output_path = os.path.join(projected_img_output_dir, 'step_{:05d}.png'.format(step))
                    save_image_batch(generated_images,
                                     current_batch_size,
                                     output_path)

                if z_vectors_output_dir:
                    output_path = os.path.join(z_vectors_output_dir, 'last-z')
                    save_z_vector_batch(z_hats,
                                        self.batch_size,
                                        output_path)
                        
        # at the end of these iterations, we're done, we have found
        # the z_hat vectors and the related generated images for this batch
        return z_hats, generated_images


    def z_to_image(self, z):
        # requires a model to be already initialized and ran
        num_z_vectors = z.shape[0]
        assert num_z_vectors <= self.batch_size, "Can't draw more images than batch size in one go"
        padded_z = np.zeros(shape=(self.batch_size, self.z_dim), dtype=np.float32)
        padded_z[:num_z_vectors] = z
        image = self.sess.run(self.sampler, feed_dict={self.z: padded_z})
        return image
        

    def interpolate(self, config=None, num_frames=64):
        # initialize and load a trained checkpoint model
        tf.initialize_all_variables().run()
        isLoaded = self.load(self.checkpoint_dir)
        assert(isLoaded)

        # load vectors if given (use random vectors if not)
        if config.vector1:
            z1 = load_z_vector(config.vector1)
        else:
            z1 = np.random.uniform(-1, 1, size=(1, self.z_dim))
        if config.vector2:
            z2 = load_z_vector(config.vector2)
        else:
            z2 = np.random.uniform(-1, 1, size=(1, self.z_dim))

        dz = (z2 - z1) / float(num_frames)
        z = np.array([z1 + i*dz for i in xrange(num_frames)], dtype=np.float32)
        z = z.reshape(-1, self.z_dim)
        
        transition_frames = self.z_to_image(z)
        
        ensure_directory(config.outDir)
        for frame_no in xrange(num_frames):
            output_path = os.path.join(config.outDir, "frame_{:02d}.png".format(frame_no))
            frame = transition_frames[frame_no, :, :]
            save_single_image(frame, output_path)
        

        
    def project(self, config):

        # create the output directories
        output_dir = config.outDir
        projected_img_output_dir = os.path.join(output_dir, 'projected')
        z_vectors_output_dir = os.path.join(output_dir, 'z_vectors')
        for directory in (projected_img_output_dir, z_vectors_output_dir):
            ensure_directory(directory)
            
        # initialize tensorflow variables
        tf.initialize_all_variables().run()

        # load a trained checkpoint model
        isLoaded = self.load(self.checkpoint_dir)
        assert(isLoaded)
        
        num_images = len(config.imgs)
        num_batches = int(np.ceil(num_images/self.batch_size))

        for batch_no in xrange(0, num_batches): 
            # create subdirectory for output (projected images for this batch)
            batch_img_dir = os.path.join(projected_img_output_dir, 'batch_{:03d}'.format(batch_no))
            batch_vectors_dir = os.path.join(z_vectors_output_dir, 'batch_{:03d}'.format(batch_no))
            ensure_directory(batch_img_dir)
            ensure_directory(batch_vectors_dir)

            # read images of this batch into an array
            batch_start_id = batch_no * self.batch_size
            batch_end_id  = min((batch_no+1) * self.batch_size,num_images)
            current_batch_size = batch_end_id - batch_start_id
            batch_files = config.imgs[batch_start_id:batch_end_id]
            batch_images = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                     for batch_file in batch_files]
            batch_images = np.array(batch_images).astype(np.float32)

            # if this is the final batch, pad the array with zeros to
            # make this final batch the same size as the others
            if current_batch_size < self.batch_size:
                pad_size = ((0, int(self.batch_size-current_batch_size)), (0,0), (0,0), (0,0))
                batch_images = np.pad(batch_images, pad_size, 'constant')
                batch_images = batch_images.astype(np.float32)

            # save the original image in the output directory for convenience
            # make a matrix of images (8 columns)
            save_image_batch(batch_images,
                             current_batch_size,
                             os.path.join(config.outDir, 'batch_{:03d}-0riginals.png'.format(batch_no)))

            # projected gradient descent with momentum to find the z_hats
            # that maximize the projection loss (full contextual loss and perceptual loss)
            (z_hats,
             generated_images) = self.find_optimal_z(loss_function = self.project_loss,
                                                     loss_gradient = self.grad_project_loss,
                                                     images = batch_images,
                                                     current_batch_size = current_batch_size,
                                                     n_iterations= config.nIter,
                                                     learning_rate = config.lr,
                                                     momentum = config.momentum,
                                                     output_every_nth_step=25,
                                                     projected_img_output_dir = batch_img_dir,
                                                     z_vectors_output_dir = batch_vectors_dir)

            # save the final z vectors for all images in the batch
            output_path = os.path.join(z_vectors_output_dir, 'batch_{:03d}'.format(batch_no))
            save_z_vector_batch(z_hats,
                                current_batch_size,
                                output_path)

            # take the average of all returned z_hats and save the corresponding image
            # it is the "average" of all images in this batch (but averaged in z-space,
            # not pixel-space, of course)
            # (since we are working with batches, this will be a single image in a whole
            # batch, the rest is zero padding)
            nonzero_z_hats = z_hats[:current_batch_size, :]
            average_z = nonzero_z_hats.mean(axis=0)
            average_image = self.z_to_image(average_z)
            output_path = os.path.join(projected_img_output_dir, 'batch_{:03d}-average-z-img.png'.format(batch_no))
            save_image_batch(average_image, 1, output_path)

            
            

    def complete(self, config):
        try:
            os.makedirs(os.path.join(config.outDir, 'hats_imgs'))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        try:
            os.makedirs(os.path.join(config.outDir, 'completed'))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        tf.initialize_all_variables().run()

        isLoaded = self.load(self.checkpoint_dir)
        assert(isLoaded)

        # data = glob(os.path.join(config.dataset, "*.png"))
        nImgs = len(config.imgs)

        batch_idxs = int(np.ceil(nImgs/self.batch_size))
        if config.maskType == 'random':
            raise NotImplementedError('random mask not yet implemented')
        elif config.maskType == 'center':
            scale = 0.25
            assert(scale <= 0.5)
            mask = np.ones(self.image_shape)
            sz = self.image_size
            l = int(self.image_size*scale)
            u = int(self.image_size*(1.0-scale))
            mask[l:u, l:u, :] = 0.0
        elif config.maskType == 'left':
            raise NotImplementedError('left mask not yet implemented')
        elif config.maskType == 'full':
            mask = np.ones(self.image_shape)
        else:
            assert(False)

        for idx in xrange(0, batch_idxs):
            l = idx*self.batch_size
            u = min((idx+1)*self.batch_size, nImgs)
            batchSz = u-l
            batch_files = config.imgs[l:u]
            batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                     for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            if batchSz < self.batch_size:
                print(batchSz)
                padSz = ((0, int(self.batch_size-batchSz)), (0,0), (0,0), (0,0))
                batch_images = np.pad(batch_images, padSz, 'constant')
                batch_images = batch_images.astype(np.float32)

            batch_mask = np.resize(mask, [self.batch_size] + self.image_shape)
            zhats = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            v = 0

            nRows = np.ceil(batchSz/8)
            nCols = 8
            save_images(batch_images[:batchSz,:,:,:], [nRows,nCols],
                        os.path.join(config.outDir, 'before.png'))
            masked_images = np.multiply(batch_images, batch_mask)
            save_images(masked_images[:batchSz,:,:,:], [nRows,nCols],
                        os.path.join(config.outDir, 'masked.png'))

            for i in xrange(config.nIter):
                fd = {
                    self.z: zhats,
                    self.mask: batch_mask,
                    self.images: batch_images,
                }
                run = [self.complete_loss, self.grad_complete_loss, self.G]
                loss, g, G_imgs = self.sess.run(run, feed_dict=fd)

                v_prev = np.copy(v)
                v = config.momentum*v - config.lr*g[0]
                zhats += -config.momentum * v_prev + (1+config.momentum)*v
                np.clip(zhats, -1, 1)

                if i % 50 == 0:
                    print(i, np.mean(loss[0:batchSz]))
                    imgName = os.path.join(config.outDir,
                                           'hats_imgs/{:05d}.png'.format(i))
                    nRows = np.ceil(batchSz/8)
                    nCols = 8
                    save_images(G_imgs[:batchSz,:,:,:], [nRows,nCols], imgName)

                    inv_masked_hat_images = np.multiply(G_imgs, 1.0-batch_mask)
                    completeed = masked_images + inv_masked_hat_images
                    imgName = os.path.join(config.outDir,
                                           'completed/{:05d}.png'.format(i))
                    save_images(completeed[:batchSz,:,:,:], [nRows,nCols], imgName)

    def discriminator(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4

    def generator(self, z):
        self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*4*4, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(self.z_, [-1, 4, 4, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = conv2d_transpose(h0,
            [self.batch_size, 8, 8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = conv2d_transpose(h1,
            [self.batch_size, 16, 16, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = conv2d_transpose(h2,
            [self.batch_size, 32, 32, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = conv2d_transpose(h3,
            [self.batch_size, 64, 64, 3], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)

    def sampler(self, z, y=None):
        tf.get_variable_scope().reuse_variables()

        h0 = tf.reshape(linear(z, self.gf_dim*8*4*4, 'g_h0_lin'),
                        [-1, 4, 4, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = conv2d_transpose(h0, [self.batch_size, 8, 8, self.gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = conv2d_transpose(h1, [self.batch_size, 16, 16, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = conv2d_transpose(h2, [self.batch_size, 32, 32, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = conv2d_transpose(h3, [self.batch_size, 64, 64, 3], name='g_h4')

        return tf.nn.tanh(h4)

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False

import sys
import os.path
sys.path.insert(0, os.path.abspath("./simple-dnn"))

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy
import time

from simple_dnn.util.format import UnitPosNegScale, reshape_pad

class OpenNetBase(object):
    """ OpenNet base class.
    """
    def __init__(self, x_dim, y_dim,
                 z_dim=6,
                 x_scale=UnitPosNegScale.scale,
                 x_inverse_scale=UnitPosNegScale.inverse_scale,
                 x_reshape=None,
                 opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                 recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                 c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                 dist='mean_separation_spread',
                 decision_dist_fn = 'euclidean',
                 threshold_type='global',
                 dropout = True, keep_prob=0.7,
                 batch_size=128, iterations=5000,
                 display_step=500, save_step=500,
                 model_directory=None,  # Directory to save trained model to.
                 density_estimation_factory=None, # Depricated
                 ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True,
                 div_loss=False, combined_loss=False,
                 contamination=0.01,
                 ):
        """
        Args:
        :param x_dim - dimension of the input
        :param y_dim - number of known classes.
        :param z_dim - the number of latent variables.
        :param x_scale - an input scaling function. Default scale to range of [-1, 1].
                         If none, the input will not be scaled.
        :param x_inverse_scale - reverse scaling fn. by rescaling from [-1, 1] to original input scale.
                                 If None, the the output of decoder(if there is a decoder) will rescaled.
        :param x_reshape - a function to reshape the input before feeding to the networks input layer.
                            If None, the input will not be reshaped.
        :param opt - the Optimizer used when updating based on ii-loss.
                     Used when inter_loss and intra_loss are enabled. Default is AdamOptimizer.
        :param recon_opt - the Optimizer used when updating based on reconstruction-loss (Not used ii, ii+ce or ce).
                           Used when recon_loss is enabled. Default is AdamOptimizer.
        :param c_opt - the Optimizer used when updating based on cross entropy loss.
                       Used for ce and ii+ce modes (i.e. ce_loss is enabled). Default is AdamOptimizer.
        :param batch_size - training barch size.
        :param iterations - number of training iterations.
        :param display_step - training info displaying interval.
        :param save_step - model saving interval.
        :param model_directory - derectory to save model in.
        :param dist - ii-loss calculation mode. Only 'mean_separation_spread' should be used.
        :param decision_dist_fn - outlier score distance functions
        :param threshold_type - outlier threshold mode. 'global' appears to give better results.
        :param ce_loss - Consider cross entropy loss. When enabled with intra_loss and inter_loss gives (ii+ce) mode.
        :param recon_loss - Experimental! avoid enabling them.
        :param inter_loss - Consider inter-class separation. Should be enabled together with intra_loss for (ii-loss).
        :param intra_loss - Consider intra-class spread. Should be enabled together with inter_loss for (ii-loss).
        :param div_loss and combined_loss - Experimental. avoid enabling them.
        :param contamination - contamination ratio used for outlier threshold estimation.
        """
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.x_scale = x_scale
        self.x_inverse_scale = x_inverse_scale
        self.x_reshape = x_reshape

        self.z_dim = z_dim

        self.dropout = dropout
        self.is_training = False
        self.keep_prob = keep_prob

        self.contamination = contamination

        self.opt = opt
        self.recon_opt = recon_opt
        self.c_opt = c_opt

        assert dist in ['class_mean', 'all_pair', 'mean_separation_spread', 'min_max']
        self.dist = dist

        self.decision_dist_fn = decision_dist_fn

        assert threshold_type in ['global', 'perclass']
        self.threshold_type = threshold_type

        # Training Config
        self.batch_size = batch_size
        self.iterations = iterations
        self.display_step = display_step
        self.save_step = save_step
        self.model_directory = model_directory

        self.enable_ce_loss, self.enable_recon_loss, \
            self.enable_inter_loss, self.enable_intra_loss, self.div_loss = \
                ce_loss, recon_loss, inter_loss, intra_loss, div_loss

        self.graph = tf.Graph()

        self.model_params = ['x_dim', 'y_dim', 'z_dim', 'dropout', 'keep_prob',
                             'contamination', 'decision_dist_fn', 'dist', 'batch_size',
                             'batch_size', 'iterations', 'enable_ce_loss',
                             'enable_recon_loss', 'enable_inter_loss', 'enable_intra_loss',
                             'div_loss', 'threshold_type']

        with self.graph.as_default():
            self.sess = tf.Session()
            self.build_model()

            # To save and restore all the variables.
            self.saver = tf.train.Saver()

    def model_config(self):
        return {field: val for field, val in vars(self).items() if field in self.model_params}

    def x_reformat(self, xs):
      """ Rescale and reshape x if x_scale and x_reshape functions are provided.
      """
      if self.x_scale is not None:
        xs = self.x_scale(xs)
      if self.x_reshape is not None:
        xs = self.x_reshape(xs)
      return xs

    def _next_batch(self, x, y):
        index = np.random.randint(0, high=x.shape[0], size=self.batch_size)
        return x[index], y[index]


    def encoder(self, x, reuse=False):
        """ Encoder network.
        Args:
            :param x - input x.
            :param reuse - whether to reuse old network on create new one.
        Returns:
            latent var z
        """
        pass

    def decoder(self, z, reuse=False):
        """ Decoder Network. Experimental!
        Args:
            :param z - latent variables z.
            :param reuse - whether to reuse old network on create new one.
        Returns:
            The reconstructed x
        """
        pass

    def bucket_mean(self, data, bucket_ids, num_buckets):
        total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
        count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
        return total / count

    def bucket_max(self, data, bucket_ids, num_buckets):
        b_max = tf.unsorted_segment_max(data, bucket_ids, num_buckets)
        return b_max

    def sq_difference_from_mean(self, data, class_mean):
        """ Calculates the squared difference from clas mean.
        """
        sq_diff_list = []
        for i in range(self.y_dim):
            sq_diff_list.append(tf.reduce_mean(
                tf.squared_difference(data, class_mean[i]), axis=1))

        return tf.stack(sq_diff_list, axis=1)


    def inter_min_intra_max(self, data, labels, class_mean):
        """ Calculates intra-class spread as max distance from class means.
        Calculates inter-class separation as the distance between the two closest class means.
        """
        _, inter_min = self.inter_separation_intra_spred(data, labels, class_mean)

        sq_diff = self.sq_difference_from_mean(data, class_mean)

        # Do element wise mul with labels to use as mask
        masked_sq_diff = tf.multiply(sq_diff, tf.cast(labels, dtype=tf.float32))
        intra_max = tf.reduce_sum(tf.reduce_max(masked_sq_diff, axis=0))

        return intra_max, inter_min

    def inter_intra_diff(self, data, labels, class_mean):
        """ Calculates the intra-class and inter-class distance
        as the average distance from the class means.
        """
        sq_diff = self.sq_difference_from_mean(data, class_mean)

        inter_intra_sq_diff = self.bucket_mean(sq_diff, labels, 2)
        inter_class_sq_diff = inter_intra_sq_diff[0]
        intra_class_sq_diff = inter_intra_sq_diff[1]
        return intra_class_sq_diff, inter_class_sq_diff

    def inter_separation_intra_spred(self, data, labels, class_mean):
        """ Calculates intra-class spread as average distance from class means.
        Calculates inter-class separation as the distance between the two closest class means.
        Returns:
        intra-class spread and inter-class separation.
        """
        intra_class_sq_diff, _ = self.inter_intra_diff(data, labels, class_mean)

        ap_dist = self.all_pair_distance(class_mean)
        dim = tf.shape(class_mean)[0]
        not_diag_mask = tf.logical_not(tf.cast(tf.eye(dim), dtype=tf.bool))
        inter_separation = tf.reduce_min(tf.boolean_mask(tensor=ap_dist, mask=not_diag_mask))
        return intra_class_sq_diff, inter_separation

    def all_pair_distance(self, A):
        r = tf.reduce_sum(A*A, 1)

        # turn r into column vector
        r = tf.reshape(r, [-1, 1])
        D = r - 2*tf.matmul(A, A, transpose_b=True) + tf.transpose(r)
        return D

    def all_pair_inter_intra_diff(self, xs, ys):
        """ Calculates the intra-class and inter-class distance
        as the average distance between all pair of instances intra and inter class
        instances.
        """

        def outer(ys):
            return tf.matmul(ys, ys, transpose_b=True)

        ap_dist = self.all_pair_distance(xs)
        mask = outer(ys)

        dist = self.bucket_mean(ap_dist, mask, 2)
        inter_class_sq_diff = dist[0]
        intra_class_sq_diff = dist[1]
        return intra_class_sq_diff, inter_class_sq_diff


    def build_model(self):
        """ Builds the network graph.
        """
        pass

    def loss_fn_training_op(self, x, y, z, logits, x_recon, class_means):
        """ Computes the loss functions and creates the update ops.

        :param x - input X
        :param y - labels y
        :param z - z layer transform of X.
        :param logits - softmax logits if ce loss is used. Can be None if only ii-loss.
        :param recon - reconstructed X. Experimental! Can be None.
        :class_means - the class means.
        """
        # Calculate intra class and inter class distance
        if self.dist == 'class_mean':   # For experimental pupose only
            self.intra_c_loss, self.inter_c_loss = self.inter_intra_diff(
                z, tf.cast(y, tf.int32), class_means)
        elif self.dist == 'all_pair':   # For experimental pupose only
            self.intra_c_loss, self.inter_c_loss = self.all_pair_inter_intra_diff(
                z, tf.cast(y, tf.int32))
        elif self.dist == 'mean_separation_spread':  # ii-loss
            self.intra_c_loss, self.inter_c_loss = self.inter_separation_intra_spred(
                z, tf.cast(y, tf.int32), class_means)
        elif self.dist == 'min_max':   # For experimental pupose only
            self.intra_c_loss, self.inter_c_loss = self.inter_min_intra_max(
                z, tf.cast(y, tf.int32), class_means)

        # Calculate reconstruction loss
        if self.enable_recon_loss:    # For experimental pupose only
            self.recon_loss = tf.reduce_mean(tf.squared_difference(x, x_recon))

        if self.enable_intra_loss and self.enable_inter_loss:        # The correct ii-loss
            self.loss = tf.reduce_mean(self.intra_c_loss - self.inter_c_loss)
        elif self.enable_intra_loss and not self.enable_inter_loss:  # For experimental pupose only
            self.loss = tf.reduce_mean(self.intra_c_loss)
        elif not self.enable_intra_loss and self.enable_inter_loss:  # For experimental pupose only
            self.loss = tf.reduce_mean(-self.inter_c_loss)
        elif self.div_loss:                                          # For experimental pupose only
            self.loss = tf.reduce_mean(self.intra_c_loss / self.inter_c_loss)
        else:                                                        # For experimental pupose only
            self.loss = tf.reduce_mean((self.recon_loss * 1. if self.enable_recon_loss else 0.)
                                       + (self.intra_c_loss * 1. if self.enable_intra_loss else 0.)
                                       - (self.inter_c_loss * 1. if self.enable_inter_loss else 0.)
                                      )

        # Classifier loss
        if self.enable_ce_loss:
            self.ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y))

        tvars = tf.trainable_variables()
        e_vars = [var for var in tvars if 'enc_' in var.name ]
        classifier_vars = [var for var in tvars if 'enc_' in var.name or 'classifier_' in var.name]
        recon_vars = [var for var in tvars if 'enc_' in var.name or 'dec_' in var.name]

        # Training Ops
        if self.enable_recon_loss:
            self.recon_train_op = self.recon_opt.minimize(self.recon_loss, var_list=recon_vars)

        if self.enable_inter_loss or self.enable_intra_loss or self.div_loss:
            self.train_op = self.opt.minimize(self.loss, var_list=e_vars)

        if self.enable_ce_loss:
            self.ce_train_op = self.c_opt.minimize(self.ce_loss, var_list=classifier_vars)


    def fit(self, X, y, X_val=None, y_val=None):
        """ Fit model.
        """
        assert y.shape[1] == self.y_dim
        start = time.time()
        self.is_training = True
        count_skip = 0
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            i = 0
            while i < self.iterations:
                xs, ys = self._next_batch(X, y)
                xs = self.x_reformat(xs)

                intra_c_loss, inter_c_loss, recon_loss, loss, ce_loss, acc, val_acc = \
                    None, None, None, None, None, None, None

                if len(np.unique(np.argmax(ys, axis=1))) != self.y_dim:
                    count_skip += 1
                    continue

                if self.enable_inter_loss or self.enable_intra_loss or self.div_loss:
                    _, intra_c_loss, inter_c_loss, loss = self.sess.run(
                        [self.train_op, self.intra_c_loss, self.inter_c_loss, self.loss],
                        feed_dict={self.x:xs, self.y:ys})

                if self.enable_recon_loss:
                    _, recon_loss = self.sess.run(
                        [self.recon_train_op, self.recon_loss],
                        feed_dict={self.x:xs, self.y:ys})

                if self.enable_ce_loss:
                    _, ce_loss, acc = self.sess.run(
                        [self.ce_train_op, self.ce_loss, self.acc],
                        feed_dict={self.x:xs, self.y:ys})
                    if X_val is not None and y_val is not None:
                        val_acc = self.sess.run(
                            self.acc,
                            feed_dict={self.x:self.x_reformat(X_val), self.y:y_val})

                if i % self.display_step == 0 and self.display_step > 0:
                    if (self.enable_inter_loss and self.enable_intra_loss) or self.div_loss:
                        self.update_class_stats(X, y)
                        acc = (self.predict(xs, reformat=False) == np.argmax(ys, axis=1)).mean()
                        if X_val is not None and y_val is not None:
                            val_acc = (self.predict(X_val) == np.argmax(y_val, axis=1)).astype(np.float).mean()

                    self._iter_stats(i, start,  intra_c_loss, inter_c_loss, recon_loss, loss, ce_loss,
                                     acc, val_acc)
                if i % self.save_step == 0 and i != 0 and self.model_directory is not None:
                    self.save_model('model-'+str(i)+'.cptk')
                    print "Saved Model"

                i += 1

        if self.display_step > 0:
            if (self.enable_inter_loss and self.enable_intra_loss) or self.div_loss:
                self.update_class_stats(X, y)
                acc = (self.predict(xs, reformat=False) == np.argmax(ys, axis=1)).mean()
                if X_val is not None and y_val is not None:
                    val_acc = (self.predict(X_val) == np.argmax(y_val, axis=1)).mean()

            self._iter_stats(i, start,  intra_c_loss, inter_c_loss, recon_loss, loss, ce_loss, acc, val_acc)
        if self.model_directory is not None:
            self.save_model('model-'+str(i)+'.cptk')
            print "Saved Model"

        # Save class means and cov
        self.update_class_stats(X, y)

        # Compute and store the selected thresholds for each calls
        self.class_thresholds(X, y)
#         print 'n_skipped_batches = ', count_skip

        self.is_training = False

    def update_class_stats(self, X, y):
        """ Recalculates class means.
        """
        z = self.latent(X)
        self.is_training = False

        # No need to feed z_test here. because self.latent() already used z_test
        self.c_means = self.sess.run(self.class_means,
                                     feed_dict={self.z:z, self.y:y})
        if self.decision_dist_fn == 'mahalanobis':
            self.c_cov, self.c_cov_inv = self.class_covarience(z, y)
        self.is_training = True

    def class_covarience(self, Z, y):
        dim = self.z_dim

        per_class_cov = np.zeros((y.shape[1], dim, dim))
        per_class_cov_inv = np.zeros_like(per_class_cov)
        for c in range(y.shape[1]):
            per_class_cov[c, :, :] = np.cov((Z[y[:, c].astype(bool)]).T)
            per_class_cov_inv[c, :, :] = np.linalg.pinv(per_class_cov[c, :, :])

        return per_class_cov, per_class_cov_inv

    def _iter_stats(self, i, start_time,  intra_c_loss, inter_c_loss, recon_loss, loss, ce_loss, acc, val_acc):
        if i == 0:
            print '{0:5}|{1:7}|{2:7}|{3:7}|{4:7}|{5:7}|{6:7}|{7:7}|{8:7}|'.format(
                'i', 'Intra', 'Inter', 'Recon', 'Total', 'CrossE', 'Acc', 'V_Acc', 'TIME(s)')

        print '{0:5}|{1:7.4}|{2:7.4}|{3:7.4}|{4:7.4}|{5:7.4}|{6:7.4}|{7:7.4}|{8:7}|'.format(
            i,  intra_c_loss, inter_c_loss, recon_loss, loss, ce_loss, acc, val_acc,
            int(time.time()-start_time))

    def latent(self, X, reformat=True):
        """ Computes the z-layer output.
        """
        self.is_training = False
        z = np.zeros((X.shape[0], self.z_dim))
        batch = self.batch_size
        with self.graph.as_default():
            for i in range(0, X.shape[0], batch):
                start = i
                end =  min(i+batch, X.shape[0])
                z[start: end] = self.sess.run(
                    self.z_test, feed_dict={self.x:self.x_reformat(X[start: end]) if reformat else X[start: end]})

        self.is_training = True
        return z

    def reconstruct(self, X):
        self.is_training = False
        with self.graph.as_default():
            x_recon =  self.sess.run(self.x_recon_test,
                                     feed_dict={self.x: self.x_reformat(X)})
        self.is_training = True
        return x_recon

    def distance_from_all_classes(self, X, reformat=True):
        """ Computes the distance of each instance from all class means.
        """
        z = self.latent(X, reformat=reformat)
        dist = np.zeros((z.shape[0], self.y_dim))
        for j in range(self.y_dim):
            if self.decision_dist_fn == 'euclidean': # squared euclidean
                dist[:, j] = np.sum(np.square(z - self.c_means[j]), axis=1)
            elif self.decision_dist_fn == 'mahalanobis':
                dist[:, j] = scipy.spatial.distance.cdist(
                    z, self.c_means[j][None, :],
                    'mahalanobis', VI=self.c_cov_inv[j]).reshape((z.shape[0]))
            else:
                ValueError('Error: Unsupported decision_dist_fn "{0}"'.format(self.decision_dist_fn))

        return dist

    def decision_function(self, X):
        """ Computes the outlier score. The larger the score the more likely it is an outlier.
        """
        dist = self.distance_from_all_classes(X)
        return np.amin(dist, axis=1)

    def predict_prob(self, X, reformat=True):
        """ Predicts class probabilities for X over known classes.
        """
        self.is_training = False
        if self.enable_ce_loss:
            with self.graph.as_default():
                batch = self.batch_size
                prob = np.zeros((X.shape[0], self.y_dim))
                for i in range(0, X.shape[0], batch):
                    start = i
                    end =  min(i+batch, X.shape[0])
                    prob[start:end] =  self.sess.run(
                        self.pred_prob_test,
                        feed_dict={self.x: self.x_reformat(X[start:end]) if reformat else X[start:end]})

        elif (self.enable_inter_loss and self.enable_intra_loss) or (self.div_loss) or self.enable_recon_loss:
            dist = self.distance_from_all_classes(X, reformat=reformat)

            prob = np.exp(-dist)
            prob = prob / prob.sum(axis=1)[:,None]

        self.is_training = True
        return prob

    def predict(self, X, reformat=True):
        """ Performs closed set classification (i.e. prediction over known classes).
        """
        prob = self.predict_prob(X, reformat=reformat)
        return np.argmax(prob, axis=1)

    def predict_open(self, X):
        """ Performs open set recognition/classification.
        """
        pred = self.predict(X)
        unknown_class_label = self.y_dim
        score = self.decision_function(X)
        for i in range(X.shape[0]):
            if score[i] > self.threshold[pred[i]]:
                pred[i] = unknown_class_label

        return pred


    def class_thresholds(self, X, y):
        """ Computes class thresholds. Shouldn't be called from outside.
        """
        score = self.decision_function(X)
        if self.threshold_type == 'global':
            self.threshold = np.ones(self.y_dim)
            cutoff_idx = max(1, int(score.shape[0] * 0.01))
            self.threshold *= sorted(score)[-cutoff_idx]
        elif self.threshold_type == 'perclass':
            c_count = y.sum(axis=0)
            self.threshold = np.zeros_like(c_count)
            for c in range(y.shape[1]):
                sorted_c_scores = sorted(score[y[:, c].astype(bool)])
                cutoff_idx = max(1, int(c_count[c] * self.contamination))
                self.threshold[c] = sorted_c_scores[-cutoff_idx]

        return self.threshold

class OpenNetFlat(OpenNetBase):
    """ OpenNet with only fully connected layers.
    """
    def __init__(self, x_dim, y_dim,
                 z_dim=6, h_dims=[128],
                 activation_fn=tf.nn.relu,
                 x_scale=UnitPosNegScale.scale,
                 x_inverse_scale=UnitPosNegScale.inverse_scale,
                 x_reshape=None,
                 opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                 recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                 c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                 dist='mean_separation_spread',
                 decision_dist_fn = 'euclidean',
                 threshold_type='global',
                 dropout = True, keep_prob=0.7,
                 batch_size=128, iterations=5000,
                 display_step=500, save_step=500,
                 model_directory=None,  # Directory to save trained model to.
                 density_estimation_factory=None, # Depricated
                 ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True,
                 div_loss=False, combined_loss=False,
                 contamination=0.02,
                 ):
        """
        Args:
        :param x_dim - dimension of the input
        :param y_dim - number of known classes.
        :param z_dim - the number of latent variables.
        :param h_dims - an int or a list; number of units in the fully conected hidden layers of the
                        encoder network. The decoder network (if used) will simply be the reverse.
        :param x_scale - an input scaling function. Default scale to range of [-1, 1].
                         If none, the input will not be scaled.
        :param x_inverse_scale - reverse scaling fn. by rescaling from [-1, 1] to original input scale.
                                 If None, the the output of decoder(if there is a decoder) will rescaled.
        :param x_reshape - a function to reshape the input before feeding to the networks input layer.
                            If None, the input will not be reshaped.
        :param opt - the Optimizer used when updating based on ii-loss.
                     Used when inter_loss and intra_loss are enabled. Default is AdamOptimizer.
        :param recon_opt - the Optimizer used when updating based on reconstruction-loss (Not used ii, ii+ce or ce).
                           Used when recon_loss is enabled. Default is AdamOptimizer.
        :param c_opt - the Optimizer used when updating based on cross entropy loss.
                       Used for ce and ii+ce modes (i.e. ce_loss is enabled). Default is AdamOptimizer.
        :param batch_size - training batch size.
        :param iterations - number of training iterations.
        :param display_step - training info displaying interval.
        :param save_step - model saving interval.
        :param model_directory - directory to save model in.
        :param dist - ii-loss calculation mode. Only 'mean_separation_spread' should be used.
        :param decision_dist_fn - outlier score distance functions
        :param threshold_type - outlier threshold mode. 'global' appears to give better results.
        :param ce_loss - Consider cross entropy loss. When enabled with intra_loss and inter_loss gives (ii+ce) mode.
        :param recon_loss - Experimental! Avoid enabling this.
        :param inter_loss - Consider inter-class separation. Should be enabled together with intra_loss for (ii-loss).
        :param intra_loss - Consider intra-class spread. Should be enabled together with inter_loss for (ii-loss).
        :param div_loss and combined_loss - Experimental. Avoid enabling them.
        :param contamination - contamination ratio used for outlier threshold estimation.
        """

        # Network Setting
        if isinstance(h_dims, list) or isinstance(h_dims, tuple):
            self.h_dims = h_dims
        else:
            self.h_dims = [h_dims]

        self.activation_fn = activation_fn

        assert decision_dist_fn in ['euclidean', 'mahalanobis']

        super(OpenNetFlat, self).__init__(
            x_dim, y_dim, z_dim=z_dim,
            x_scale=x_scale, x_inverse_scale=x_inverse_scale, x_reshape=x_reshape,
            opt=opt, recon_opt=recon_opt, c_opt=c_opt, threshold_type=threshold_type,
            dist=dist, decision_dist_fn=decision_dist_fn, dropout=dropout, keep_prob=keep_prob,
            batch_size=batch_size, iterations=iterations,
            display_step=display_step, save_step=save_step,
            model_directory=model_directory,
            ce_loss=ce_loss, recon_loss=recon_loss, inter_loss=inter_loss, intra_loss=intra_loss,
            div_loss=div_loss, contamination=contamination)

        self.model_params += ['h_dims', 'activation_fn']


    def encoder(self, x, reuse=False):
        """ Encoder network.
        Args:
            :param x - input x.
            :param reuse - whether to reuse old network on create new one.
        Returns:
            A tuple z, softmax input logits 
        """
        net = x
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            activation_fn=self.activation_fn):
            for i, num_unit in enumerate(self.h_dims):
                net = slim.fully_connected(
                    net, num_unit,
                    normalizer_fn=slim.batch_norm,
                    reuse=reuse, scope='enc_{0}'.format(i))
                if self.dropout:
                    net = slim.dropout(net, keep_prob=self.keep_prob, is_training=self.is_training)
        # It is very important to batch normalize the output of encoder.
        z = slim.fully_connected(
            net, self.z_dim, activation_fn=None,
            normalizer_fn=slim.batch_norm,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            reuse=reuse, scope='enc_z')

        # This used when CE cost is enabled.
        logits = slim.fully_connected(
            z, self.y_dim, activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            scope='classifier_logits', reuse=reuse)

        return z, logits

    def decoder(self, z, reuse=False):
        """ Decoder Network.
        Args:
            :param z - latent variables z.
            :param reuse - whether to reuse old network on create new one.
        Returns:
            The reconstructed x
        """
        net = z

        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            activation_fn=self.activation_fn):
            h_dims_revese = [self.h_dims[i]
                             for i in range(len(self.h_dims) - 1, -1, -1)]
            for i, num_unit in enumerate(h_dims_revese):
                net = slim.fully_connected(
                    net, num_unit,
                    normalizer_fn=slim.batch_norm,
                    reuse=reuse, scope='dec_{0}'.format(i))
                if self.dropout:
                    net = slim.dropout(net, keep_prob=self.keep_prob, is_training=self.is_training)

        dec_out = slim.fully_connected(
            net, self.x_dim, activation_fn=tf.nn.tanh,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            reuse=reuse, scope='dec_out')
        return dec_out


    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.x_dim])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])

        self.z, logits = self.encoder(self.x)

        if self.enable_recon_loss:
            self.x_recon = self.decoder(self.z)
        else:
            self.x_recon = None

        # Calculate class mean
        self.class_means = self.bucket_mean(self.z, tf.argmax(self.y, axis=1), self.y_dim)

        self.loss_fn_training_op(self.x, self.y, self.z, logits, self.x_recon, self.class_means)

        self.pred_prob = tf.nn.softmax(logits=logits)
        pred = tf.argmax(self.pred_prob, axis=1)
        actual = tf.argmax(self.y, axis=1)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(pred, actual), tf.float32))

        # For Inference, set is_training
        self.is_training = False
        self.z_test, logits_test = self.encoder(self.x, reuse=True)
        self.pred_prob_test = tf.nn.softmax(logits=logits_test)
        if self.enable_recon_loss:
            self.x_recon_test = self.decoder(self.z_test, reuse=True)
        self.is_training = True


class OpenNetCNN(OpenNetBase):
    """ OpenNet with convolutional and fully connected layers.
    Current supports simple architecture with alternating cov and pooling layers.
    """
    def __init__(self, x_dim, x_ch, y_dim, conv_units, hidden_units,
                 z_dim=6,
                 kernel_sizes=[5,5], strides=[1, 1], paddings='SAME',
                 pooling_enable=False, pooling_kernel=[2,2],
                 pooling_stride=[2,2], pooling_padding='SAME',
                 pooling_type='max', # 'avg' or 'max'
                 activation_fn=tf.nn.relu,

                 x_scale=UnitPosNegScale.scale,
                 x_inverse_scale=UnitPosNegScale.inverse_scale,
                 x_reshape=None,

                 opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                 recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                 c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),

                 dist='mean_separation_spread',
                 decision_dist_fn = 'euclidean',
                 threshold_type='global',
                 dropout = True, keep_prob=0.7,
                 batch_size=128, iterations=5000,
                 display_step=500, save_step=500,
                 model_directory=None,  # Directory to save trained model to.
                 density_estimation_factory=None, # deprecated
                 ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True,
                 div_loss=False,
                 contamination=0.01,
                 ):
        """
        Args:
        :param x_dim - dimension of the input
        :param y_dim - number of known classes.
        :param conv_units - a list of ints. The number of filters in each convolutional layer.
        :param hidden_units - a list of ints. The of units in each fully connected layer.
        :param z_dim - the number of latent variables.
        :param h_dims - an int or a list; number of units in the fully connected hidden layers of the
                        encoder network.
        :param kernel_sizes - a list or a list of lists. Size of the kernel of the conv2d.
                              If a list with two ints all layers use the same kernel size.
                              Otherwise if a list of list (example [[5,5], [4,4]]) each layer
                              will have different kernel size.
        :param strides - a list or a list of lists. The strides of each conv2d kernel.
        :param paddings - padding for each conv2d. Default 'SAME'.
        :param pooling_enable - if True, add pooling layer after each conv2d layer.
        :param pooling_kernel - a list or a list of lists. The size of the pooling kernel.
                                If a list with two ints all layers use the same kernel size.
                                Otherwise if a list of list (example [[5,5], [4,4]]) each layer
                                will have different kernel size.
        :param pooling_stride - a list or a list of lists. The strides of each pooing kernel.
        :param pooling_padding - padding for each pool2d layer. Default 'SAME'.
        :param pooling_type - pooling layer type. supported 'avg' or 'max'. Default max_pool2d.
        :param x_scale - an input scaling function. Default scale to range of [-1, 1].
                         If none, the input will not be scaled.
        :param x_inverse_scale - reverse scaling fn. by rescaling from [-1, 1] to original input scale.
                                 If None, the the output of decoder(if there is a decoder) will rescaled.
        :param x_reshape - a function to reshape the input before feeding to the networks input layer.
                            If None, the input will not be reshaped.
        :param opt - the Optimizer used when updating based on ii-loss.
                     Used when inter_loss and intra_loss are enabled. Default is AdamOptimizer.
        :param recon_opt - the Optimizer used when updating based on reconstruction-loss (Not used ii, ii+ce or ce).
                           Used when recon_loss is enabled. Default is AdamOptimizer.
        :param c_opt - the Optimizer used when updating based on cross entropy loss.
                       Used for ce and ii+ce modes (i.e. ce_loss is enabled). Default is AdamOptimizer.
        :param batch_size - training batch size.
        :param iterations - number of training iterations.
        :param display_step - training info displaying interval.
        :param save_step - model saving interval.
        :param model_directory - directory to save model in.
        :param dist - ii-loss calculation mode. Only 'mean_separation_spread' should be used.
        :param decision_dist_fn - outlier score distance functions
        :param threshold_type - outlier threshold mode. 'global' appears to give better results.
        :param ce_loss - Consider cross entropy loss. When enabled with intra_loss and inter_loss gives (ii+ce) mode.
        :param recon_loss - Experimental! Avoid enabling this.
        :param inter_loss - Consider inter-class separation. Should be enabled together with intra_loss for (ii-loss).
        :param intra_loss - Consider intra-class spread. Should be enabled together with inter_loss for (ii-loss).
        :param div_loss and combined_loss - Experimental. Avoid enabling them.
        :param contamination - contamination ratio used for outlier threshold estimation.
        """
        self.x_ch = x_ch

        # Conv layer config
        self.conv_units = conv_units
        if isinstance(kernel_sizes[0], list) or isinstance(kernel_sizes[0], tuple):
            assert len(conv_units) == len(kernel_sizes)
            self.kernel_sizes = kernel_sizes
        else:
            self.kernel_sizes = [kernel_sizes] * len(conv_units)

        if isinstance(strides[0], list) or isinstance(strides[0], tuple):
            assert len(conv_units) == len(strides)
            self.strides = strides
        else:
            self.strides = [strides] * len(conv_units)

        if isinstance(paddings, list):
            assert len(conv_units) == len(paddings)
            self.paddings = paddings
        else:
            self.paddings = [paddings] * len(conv_units)

        # Conv pooling config
        self.pooling_enable = pooling_enable
        assert pooling_type in ['avg', 'max']   # supported pooling types.
        self.pooling_type = pooling_type

        if isinstance(pooling_kernel[0], list) or isinstance(pooling_kernel[0], tuple):
            assert len(conv_units) == len(pooling_kernel)
            self.pooling_kernels = pooling_kernel
        else:
            self.pooling_kernels = [pooling_kernel] * len(conv_units)

        if isinstance(pooling_stride[0], list) or isinstance(pooling_stride[0], tuple):
            assert len(conv_units) == len(pooling_stride)
            self.pooling_strides = pooling_stride
        else:
            self.pooling_strides = [pooling_stride] * len(conv_units)

        if isinstance(pooling_padding, list):
            assert len(conv_units) == len(pooling_padding)
            self.pooling_paddings = pooling_padding
        else:
            self.pooling_paddings = [pooling_padding] * len(conv_units)

        # Fully connected layer config
        self.hidden_units = hidden_units

        self.activation_fn = activation_fn

        assert decision_dist_fn in ['euclidean', 'mahalanobis']

        super(OpenNetCNN, self).__init__(
            x_dim, y_dim, z_dim=z_dim,
            x_scale=x_scale, x_inverse_scale=x_inverse_scale, x_reshape=x_reshape,
            opt=opt, recon_opt=recon_opt, c_opt=c_opt, threshold_type=threshold_type,
            dist=dist, decision_dist_fn=decision_dist_fn, dropout=dropout, keep_prob=keep_prob,
            batch_size=batch_size, iterations=iterations,
            display_step=display_step, save_step=save_step,
            model_directory=model_directory,
            ce_loss=ce_loss, recon_loss=recon_loss, inter_loss=inter_loss, intra_loss=intra_loss,
            div_loss=div_loss, contamination=contamination)


        self.model_params += ['x_ch', 'conv_units', 'kernel_sizes', 'strides', 'paddings',
                              'pooling_enable', 'pooling_type', 'pooling_kernel', 'pooling_strides',
                              'pooling_padding', 'hidden_units', 'activation_fn']


    def build_conv(self, x, reuse=False):
        """ Builds the convolutional layers.
        """
        net = x
        with slim.arg_scope([slim.conv2d], padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer(),#tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            activation_fn=self.activation_fn):
            for i, (c_unit, kernel_size, stride, padding, p_kernel, p_stride, p_padding) in enumerate(zip(
                    self.conv_units, self.kernel_sizes, self.strides, self.paddings,
                    self.pooling_kernels, self.pooling_strides, self.pooling_paddings)):
                # Conv
                net = slim.conv2d(net, c_unit, kernel_size, stride=stride,
                                  normalizer_fn=slim.batch_norm,
                                  reuse=reuse, padding=padding, scope='enc_conv{0}'.format(i))

                if self.display_step > 0:
                    print 'Conv_{0}.shape = {1}'.format(i, net.get_shape())
                # Pooling
                if self.pooling_enable:
                    if self.pooling_type == 'max':
                        net = slim.max_pool2d(net, kernel_size=p_kernel, scope='enc_pool{0}'.format(i),
                                              stride=p_stride, padding=p_padding)
                    elif self.pooling_type == 'avg':
                        net = slim.avg_pool2d(net, kernel_size=p_kernel, scope='enc_pool{0}'.format(i),
                                              stride=p_stride, padding=p_padding)

                    if self.display_step > 0:
                        print 'Pooling_{0}.shape = {1}'.format(i, net.get_shape())
                # Dropout: Do NOT use dropout for conv layers. Experiments show it gives poor result.
        return net

    def encoder(self,  x, reuse=False):
        """ Builds the network.
        Args:
            :param x - input x.
            :param reuse - whether to reuse old network on create new one.
        Returns:
            Latent variables z and logits(which will be used if ce_loss is enabled.)
        """
        # Conv Layers
        net = self.build_conv(x, reuse=reuse)
        net = slim.flatten(net)

        # Fully Connected Layer
        with slim.arg_scope([slim.fully_connected], reuse=reuse,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            activation_fn=self.activation_fn):
            for i, h_unit in enumerate(self.hidden_units):
                net = slim.fully_connected(net, h_unit,
                                           normalizer_fn=slim.batch_norm,
                                           scope='enc_full{0}'.format(i))
                if self.dropout:
                    net = slim.dropout(net, keep_prob=self.keep_prob, is_training=self.is_training,
                                       scope='enc_full_dropout{0}'.format(i))

        # Latent Variable
        # It is very important to batch normalize the output of encoder.
        z = slim.fully_connected(
            net, self.z_dim, activation_fn=None,
            normalizer_fn=slim.batch_norm,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            reuse=reuse, scope='enc_z')

        logits = slim.fully_connected(
            z, self.y_dim, activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            reuse=reuse, scope='classifier_logits')

        return z, logits



    def decoder(self, z, reuse=False):
        """ Decoder Network. Experimental and not complete.
        Args:
            :param z - latent variables z.
            :param reuse - whether to reuse old network on create new one.
        Returns:
            The reconstructed x
        """
        net = z

        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            activation_fn=self.activation_fn):
            h_dims_revese = [self.hidden_units[i]
                             for i in range(len(self.hidden_units) - 1, -1, -1)]
            for i, num_unit in enumerate(h_dims_revese):
                net = slim.fully_connected(
                    net, num_unit,
                    normalizer_fn=slim.batch_norm,
                    reuse=reuse, scope='dec_{0}'.format(i))
                if self.dropout:
                    net = slim.dropout(net, keep_prob=self.keep_prob, is_training=self.is_training)

        dec_out = slim.fully_connected(
            net, self.x_dim[0] * self.x_dim[1] * self.x_ch, activation_fn=tf.nn.tanh,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            reuse=reuse, scope='dec_out')
        return dec_out



    def build_model(self):
        """ Builds the network graph.
        """
        self.x = tf.placeholder(tf.float32, [None, self.x_dim[0], self.x_dim[1], self.x_ch])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])

        self.z, logits = self.encoder(self.x)

        if self.enable_recon_loss:
            self.x_recon = self.decoder(self.z)
        else:
            self.x_recon = None

        # Calculate class mean
        self.class_means = self.bucket_mean(self.z, tf.argmax(self.y, axis=1), self.y_dim)

        self.loss_fn_training_op(slim.flatten(self.x), self.y, self.z,
                                 logits, self.x_recon, self.class_means)

        self.pred_prob = tf.nn.softmax(logits=logits)
        pred = tf.argmax(self.pred_prob, axis=1)
        actual = tf.argmax(self.y, axis=1)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(pred, actual), tf.float32))

        # For Inference, set is_training. Can be done in a better, this should do for now.
        self.is_training = False
        self.z_test, logits_test = self.encoder(self.x, reuse=True)
        self.pred_prob_test = tf.nn.softmax(logits=logits_test)
        if self.enable_recon_loss:
            self.x_recon_test = self.decoder(self.z_test, reuse=True)
        self.is_training = True

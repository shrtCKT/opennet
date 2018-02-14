import sys
import os.path
sys.path.insert(0, os.path.abspath("./simple-dnn"))

import numpy as np
import scipy
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time

from simple_dnn.util.format import UnitPosNegScale, reshape_pad

from open_net import OpenNetBase

# Download OSDN from https://github.com/abhijitbendale/OSDN and compile libmr.
sys.path.insert(0, os.path.abspath("./OSDN"))
import libmr

class OpenMaxBase(OpenNetBase):
    def __init__(self, x_dim, y_dim,
                 h_dims=[128],
                 activation_fn=tf.nn.relu,
                 x_scale=UnitPosNegScale.scale,
                 x_inverse_scale=UnitPosNegScale.inverse_scale,
                 x_reshape=None,
                 c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                 decision_dist_fn = 'eucos',
                 dropout = True, keep_prob=0.7,
                 batch_size=128, iterations=5000,
                 display_step=500, save_step=500,
                 model_directory=None,  # Directory to save trained model to.
                 tailsize = 20,
                 alpharank = 4,
                 ):
        """
        Args:
        :param x_dim - dimension of the input
        :param y_dim - number of known classes.
        :param h_dims - a list of ints. The of units in each fully connected layer.
        :param x_inverse_scale - reverse scaling fn. by rescaling from [-1, 1] to original input scale.
                                If None, the the output of decoder(if there is a decoder) will rescaled.
        :param x_reshape - a function to reshape the input before feeding to the networks input layer.
                           If None, the input will not be reshaped.
        :param c_opt - the Optimizer used when updating based on cross entropy loss. Default is AdamOptimizer.
        :param decision_dist_fn - distance function used when calculating distance from MAV.
        :param batch_size - training barch size.
        :param iterations - number of training iterations.
        :param display_step - training info displaying interval.
        :param save_step - model saving interval.
        :param model_directory - directory to save model in.
        :param tailsize - int, openmax parameter which specifies the number instances to consider when performing the
                           weibull tail fitting.
        :param alpharank = int, openmax parameter which specifies the  number of top-k activation values to take
                           values when redistributing the activation vector values.
        """
        assert decision_dist_fn in ['euclidean', 'eucos']
        self.tailsize = tailsize
        assert alpharank < y_dim
        self.alpharank = alpharank

        super(OpenMaxBase, self).__init__(
            x_dim, y_dim, z_dim=y_dim,
            x_scale=x_scale, x_inverse_scale=x_inverse_scale, x_reshape=x_reshape,
            opt=None, recon_opt=None, c_opt=c_opt,
            decision_dist_fn=decision_dist_fn, dropout=dropout, keep_prob=keep_prob,
            batch_size=batch_size, iterations=iterations,
            display_step=display_step, save_step=save_step,
            model_directory=model_directory,
            ce_loss=True, recon_loss=False, inter_loss=False, intra_loss=False,
            div_loss=False)

    def dist_from_mav(self, Z, c_mu):
        if self.decision_dist_fn == 'euclidean':
            return scipy.spatial.distance.cdist(Z, c_mu, metric=self.decision_dist_fn) / 200
        elif self.decision_dist_fn == 'eucos':
            return (scipy.spatial.distance.cdist(Z, c_mu, metric='euclidean') / 200) + \
                    scipy.spatial.distance.cdist(Z, c_mu, metric='cosine')

    def update_class_stats(self, X, y):
        z = self.latent(X)
        pred_y = self.predict(X)
        correct = (pred_y == np.argmax(y, axis=1))
        z = z[correct]
        pred_y = pred_y[correct]

        # fit weibull model for each class
        self.mr_model = {}
        self.c_means = np.zeros((self.y_dim, z.shape[1]))

        for c in range(self.y_dim):
            # Calculate Class Mean
            z_c = z[pred_y == c]
            mu_c = z_c.mean(axis=0)
            # Fit Weibull
            mr = libmr.MR()
            tailtofit = sorted(self.dist_from_mav(z_c, mu_c[None, :]).ravel())[-self.tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            self.mr_model[c] = mr
            self.c_means[c, :] = mu_c


    def predict_prob_open(self, X):
        """ Predicts open set class probabilities for X
        """
        z =  self.latent(X)
        pred_test = self.predict(X)

        alpha_weights = [((self.alpharank+1) - i)/float(self.alpharank) for i in range(1, self.alpharank+1)]
        descending_argsort = np.fliplr(np.argsort(z, axis=1))
        z_normalized = np.zeros((z.shape[0], z.shape[1]+1))

        # compute distance from MAV
        all_dist = self.dist_from_mav(z, self.c_means)

        # Compute OpenMax Prob
        for i in range(z.shape[0]):  # for each data point
            for alpha in range(self.alpharank):
                c = descending_argsort[i, alpha]
                ws_c = 1 - self.mr_model[c].w_score(all_dist[i, c]) * alpha_weights[alpha]

                z_normalized[i, c] = z[i,c] * ws_c
                z_normalized[i, -1] += z[i,c] * (1 - ws_c)

        open_prob = np.exp(z_normalized)
        if np.any(open_prob.sum(axis=1)[:,None] == np.inf):
            print 'Error: Inf value has been returned from w_score function. Consider training with larger tailsize value.'
        open_prob = open_prob / open_prob.sum(axis=1)[:,None]
        return open_prob

    def predict_open(self, X):
        """ Predicts closed set class probabilities for X
        """
        open_prob = self.predict_prob_open(X)
        return np.argmax(open_prob, axis=1)

    def distance_from_all_classes(self, X, reformat=True):
        """ Computes distance of X from all class MAVs.
        """
        z = self.latent(X, reformat=reformat)
        dist = self.dist_from_mav(z, self.c_means)

        return dist

    def decision_function(self, X):
        open_prob = self.predict_prob_open(X)
        return open_prob[:, -1]


class OpenMaxFlat(OpenMaxBase):
    def __init__(self, x_dim, y_dim,
                 h_dims=[128],
                 activation_fn=tf.nn.relu,
                 x_scale=UnitPosNegScale.scale,
                 x_inverse_scale=UnitPosNegScale.inverse_scale,
                 x_reshape=None,
                 c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                 decision_dist_fn = 'eucos',
                 dropout = True, keep_prob=0.7,
                 batch_size=128, iterations=5000,
                 display_step=500, save_step=500,
                 model_directory=None,  # Directory to save trained model to.
                 tailsize = 20,
                 alpharank = 4,
                 ):
        """
        Args:
        :param x_dim - dimension of the input
        :param y_dim - number of known classes.
        :param h_dims - a list of ints. The of units in each fully connected layer.
        :param x_inverse_scale - reverse scaling fn. by rescaling from [-1, 1] to original input scale.
                                 If None, the the output of decoder(if there is a decoder) will rescaled.
        :param x_reshape - a function to reshape the input before feeding to the networks input layer.
                            If None, the input will not be reshaped.
        :param c_opt - the Optimizer used when updating based on cross entropy loss. Default is AdamOptimizer.
        :param batch_size - training batch size.
        :param iterations - number of training iterations.
        :param display_step - training info displaying interval.
        :param save_step - model saving interval.
        :param model_directory - directory to save model in.
        :param decision_dist_fn - distance function used when calculating distance from MAV.
        :param tailsize - int, openmax parameter which specifies the number instances to consider when performing the
                            weibull tail fitting.
        :param alpharank = int, openmax parameter which specifies the  number of top-k activation values to take
                            values when redistributing the activation vector values.
        """

        # Network Setting
        if isinstance(h_dims, list) or isinstance(h_dims, tuple):
            self.h_dims = h_dims
        else:
            self.h_dims = [h_dims]

        self.activation_fn = activation_fn

        super(OpenMaxFlat, self).__init__(
            x_dim, y_dim,
            x_scale=x_scale, x_inverse_scale=x_inverse_scale, x_reshape=x_reshape,
            c_opt=c_opt,
            decision_dist_fn=decision_dist_fn, dropout=dropout, keep_prob=keep_prob,
            batch_size=batch_size, iterations=iterations,
            display_step=display_step, save_step=save_step,
            model_directory=model_directory,
            tailsize=tailsize, alpharank=alpharank)

        self.model_params += ['h_dims', 'activation_fn']


    def encoder(self, x, reuse=False):
        """ Encoder network.
        Args:
            :param x - input x.
            :param reuse - whether to reuse old network on create new one.
        Retuns:
            z
        """
        net = x
        with slim.arg_scope([slim.fully_connected], normalizer_fn=slim.batch_norm,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            activation_fn=self.activation_fn):
            for i, num_unit in enumerate(self.h_dims):
                net = slim.fully_connected(
                    net, num_unit,
                    reuse=reuse, scope='enc_{0}'.format(i))
                if self.dropout:
                    net = slim.dropout(net, keep_prob=self.keep_prob, is_training=self.is_training)
        # It is very important to batch normalize the output of encoder.
        z = slim.fully_connected(
            net, self.z_dim, activation_fn=None, normalizer_fn=slim.batch_norm,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            reuse=reuse, scope='enc_z')

        return z


    def build_model(self):
        """
        """
        self.x = tf.placeholder(tf.float32, shape=[None, self.x_dim])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])

        self.z = self.encoder(self.x)

        self.x_recon = None

        logits = self.z

        # Calculate class mean
        self.class_means = self.bucket_mean(self.z, tf.argmax(self.y, axis=1), self.y_dim)

        self.loss_fn_training_op(self.x, self.y, self.z, logits, self.x_recon, self.class_means)

        self.pred_prob = tf.nn.softmax(logits=logits)
        pred = tf.argmax(self.pred_prob, axis=1)
        actual = tf.argmax(self.y, axis=1)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(pred, actual), tf.float32))

        # For Inference, set is_training
        self.is_training = False
        self.z_test = self.encoder(self.x, reuse=True)
        self.pred_prob_test = tf.nn.softmax(logits=self.z_test)
        self.is_training = True

class OpenMaxCNN(OpenMaxBase):
    def __init__(self, x_dim, x_ch, y_dim, conv_units, hidden_units,
                 kernel_sizes=[5,5], strides=[1, 1], paddings='SAME',
                 pooling_enable=False, pooling_kernel=[2,2],
                 pooling_stride=[2,2], pooling_padding='SAME',
                 pooling_type='avg', # 'avg' or 'max'
                 activation_fn=tf.nn.relu,

                 x_scale=UnitPosNegScale.scale,
                 x_inverse_scale=UnitPosNegScale.inverse_scale,
                 x_reshape=None,

                 c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                 decision_dist_fn = 'eucos',
                 dropout=True, keep_prob=0.7,
                 batch_size=128, iterations=5000,
                 display_step=500, save_step=500,
                 model_directory=None,  # Directory to save trained model to.
                 tailsize = 20,
                 alpharank = 4,
                 ):
        """
        Args:
        :param x_dim - dimension of the input
        :param y_dim - number of known classes.
        :param conv_units - a list of ints. The number of filters in each convolutional layer.
        :param hidden_units - a list of ints. The of units in each fully connected layer.
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
        :param c_opt - the Optimizer used when updating based on cross entropy loss. Default is AdamOptimizer.
        :param batch_size - training batch size.
        :param iterations - number of training iterations.
        :param display_step - training info displaying interval.
        :param save_step - model saving interval.
        :param model_directory - directory to save model in.
        :param decision_dist_fn - distance function used when calculating distance from MAV.
        :param tailsize - int, openmax parameter which specifies the number instances to consider when performing the
                            weibull tail fitting.
        :param alpharank = int, openmax parameter which specifies the  number of top-k activation values to take
                            values when redistributing the activation vector values.
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

        super(OpenMaxCNN, self).__init__(
            x_dim, y_dim,
            x_scale=x_scale, x_inverse_scale=x_inverse_scale, x_reshape=x_reshape,
            c_opt=c_opt,
            decision_dist_fn=decision_dist_fn, dropout=dropout, keep_prob=keep_prob,
            batch_size=batch_size, iterations=iterations,
            display_step=display_step, save_step=save_step,
            model_directory=model_directory,
            tailsize=tailsize, alpharank=alpharank)

        self.model_params += ['x_ch', 'conv_units', 'kernel_sizes', 'strides', 'paddings',
                              'pooling_enable', 'pooling_type', 'pooling_kernel', 'pooling_strides',
                              'pooling_padding', 'hidden_units', 'activation_fn']

    def build_conv(self, x, reuse=False):
        net = x
        with slim.arg_scope([slim.conv2d], padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            activation_fn=self.activation_fn):
            for i, (c_unit, kernel_size, stride, padding, p_kernel, p_stride, p_padding) in enumerate(zip(
                    self.conv_units, self.kernel_sizes, self.strides, self.paddings,
                    self.pooling_kernels, self.pooling_strides, self.pooling_paddings)):
                # Conv
                net = slim.conv2d(net, c_unit, kernel_size, stride=stride,
                                  normalizer_fn=slim.batch_norm, reuse=reuse,
                                  padding=padding, scope='enc_conv{0}'.format(i))

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
        """ Encoder network.
        Args:
            :param x - input x.
            :param reuse - whether to reuse old network on create new one.
        Retuns:
            Latent variables z
        """
        # Conv Layers
        net = self.build_conv(x, reuse=reuse)
        net = slim.flatten(net)

        # Fully Connected Layer
        with slim.arg_scope([slim.fully_connected], normalizer_fn=slim.batch_norm,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            activation_fn=self.activation_fn):
            for i, h_unit in enumerate(self.hidden_units):
                net = slim.fully_connected(net, h_unit, normalizer_fn=slim.batch_norm,
                                           reuse=reuse, scope='enc_full{0}'.format(i))
                if self.dropout:
                    net = slim.dropout(net, keep_prob=self.keep_prob, is_training=self.is_training,
                                       scope='enc_full_dropout{0}'.format(i))

        # Latent Variable
        # It is very important to batch normalize the output of encoder.
        z = slim.fully_connected(
            net, self.z_dim, activation_fn=None, normalizer_fn=slim.batch_norm,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            reuse=reuse, scope='enc_z')

        return z

    def build_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.x_dim[0], self.x_dim[1], self.x_ch])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])

        self.z = self.encoder(self.x)

        self.x_recon = None

        logits = self.z

        # Calculate class mean
        self.class_means = self.bucket_mean(self.z, tf.argmax(self.y, axis=1), self.y_dim)

        self.loss_fn_training_op(slim.flatten(self.x), self.y, self.z,
                                 logits, self.x_recon, self.class_means)

        self.pred_prob = tf.nn.softmax(logits=logits)
        pred = tf.argmax(self.pred_prob, axis=1)
        actual = tf.argmax(self.y, axis=1)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(pred, actual), tf.float32))

        # For Inference, set is_training
        self.is_training = False
        self.z_test = self.encoder(self.x, reuse=True)
        self.pred_prob_test = tf.nn.softmax(logits=self.z_test)
        self.is_training = True

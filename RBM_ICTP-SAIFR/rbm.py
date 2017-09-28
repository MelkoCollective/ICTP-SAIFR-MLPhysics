########## ICTP-SAIFR Minicourse on Machine Learning for Many-Body Physics ##########
### Roger Melko, Juan Carrasquilla, Lauren Hayward Sierens and Giacomo Torlai
### Tutorial 4: Restricted Boltzmann Machine (RBM)
#####################################################################################

import tensorflow as tf
import itertools as it
import numpy as np

class RBM(object):
  
    ### Constructor ###
    def __init__(self, num_hidden, num_visible, num_samples=128, weights=None, visible_bias=None, hidden_bias=None):
        self.num_hidden = num_hidden   #number of hidden units
        self.num_visible = num_visible #number of visible units

        #visible bias:
        default = tf.zeros(shape=(self.num_visible, 1))
        self.visible_bias = self._create_parameter_variable(visible_bias, default)
        
        #hidden bias:
        default = tf.zeros(shape=(self.num_hidden, 1))
        self.hidden_bias = self._create_parameter_variable(hidden_bias, default)
        
        #pairwise weights:
        default = tf.random_normal(shape=(self.num_visible, self.num_hidden), mean=0, stddev=0.05)
        self.weights = self._create_parameter_variable(weights, default)

        #variables for sampling:
        self.num_samples = num_samples #number of samples to return when sample_p_of_v is called.

        self.hidden_samples = tf.Variable(
            self.sample_binary_tensor(tf.constant(0.5), self.num_samples, self.num_hidden),
            trainable=False, name='hidden_samples'
        )

        self.p_of_v = None

    ### Method to initialize variables: ###
    @staticmethod
    def _create_parameter_variable(initial_value=None, default=None):
        if initial_value is None:
            initial_value = default
        return tf.Variable(initial_value)

    def p_of_h_given(self, v):
        ''' Conditional probability of hidden layer given visible state '''
        # type: (tf.Tensor) -> tf.Tensor
        return tf.nn.sigmoid(tf.matmul(v, self.weights) + tf.transpose(self.hidden_bias))

    def p_of_v_given(self, h):
        ''' Conditional probability of visible layer given hidden state '''
        # type: (tf.Tensor) -> tf.Tensor
        return tf.nn.sigmoid(tf.matmul(h, self.weights, transpose_b=True) + tf.transpose(self.visible_bias))

    def sample_h_given(self, v):
        ''' Sample the hidden nodes given a visible state '''
        # type: (tf.Tensor) -> (tf.Tensor, tf.Tensor)
        b = tf.shape(v)[0]  # number of samples
        m = self.num_hidden
        prob_h = self.p_of_h_given(v)
        samples = self.sample_binary_tensor(prob_h, b, m)
        return samples, prob_h

    def sample_v_given(self, h):
        ''' Samples the visible nodes given a hidden state '''
        # type: (tf.Tensor) -> (tf.Tensor, tf.Tensor)
        b = tf.shape(h)[0]  # number rof samples
        n = self.num_visible
        prob_v = self.p_of_v_given(h)
        samples = self.sample_binary_tensor(prob_v, b, n)
        return samples, prob_v

    def stochastic_maximum_likelihood(self, num_iterations):
        # type: (int) -> (tf.Tensor, tf.Tensor, tf.Tensor)
        """
        Define persistent CD_k. Stores the results of `num_iterations` of contrastive divergence in
        class variables.
        :param int num_iterations: The 'k' in CD_k.
        """
        h_samples = self.hidden_samples
        v_samples = None
        p_of_v = 0
        for i in range(num_iterations):
            v_samples, p_of_v = self.sample_v_given(h_samples)
            h_samples, _ = self.sample_h_given(v_samples)

        self.hidden_samples = self.hidden_samples.assign(h_samples)
        self.p_of_v = p_of_v
        return self.hidden_samples, v_samples
 
    def energy(self, hidden_samples, visible_samples):
        # type: (tf.Tensor, tf.Tensor) -> tf.Tensor
        """Compute the energy:
            E = - aT*v - bT*h - vT*W*h.

        Note that since we want to support larger batch sizes, we do element-wise multiplication between
        vT*W and h, and sum along the columns to get a Tensor of shape batch_size by 1

        :param hidden_samples: Tensor of shape batch_size by num_hidden
        :param visible_samples:  Tensor of shae batch_size by num_visible
        """
        return (-tf.matmul(hidden_samples, self.hidden_bias)  # b x m * m x 1
                - tf.matmul(visible_samples, self.visible_bias)  # b x n * n x 1
                - tf.reduce_sum(tf.matmul(visible_samples, self.weights) * hidden_samples, 1))

    def free_energy(self, visible_samples):
        ''' Compute the free energy:
            F = aT*v + sum(softplus(a + vT*W)) '''
        free_energy = (tf.matmul(visible_samples, self.visible_bias)
                       + tf.reduce_sum(tf.nn.softplus(tf.matmul(visible_samples, self.weights)
                                                      + tf.transpose(self.hidden_bias)), 1, keep_dims=True))
        return free_energy   
    
    def neg_log_likelihood_grad(self, visible_samples, model_samples=None, num_gibbs=2):
        # type: (tf.Tensor, tf.Tensor, int) -> tf.Tensor

        hidden_samples, _ = self.sample_h_given(visible_samples)
        expectation_from_data = tf.reduce_mean(self.energy(hidden_samples, visible_samples))
        model_hidden, model_visible = self.split_samples(model_samples) or self.stochastic_maximum_likelihood(num_gibbs)
        expectation_from_model = tf.reduce_mean(self.energy(model_hidden, model_visible))

        return expectation_from_data - expectation_from_model

    def neg_log_likelihood(self, visible_samples, log_Z):
        ''' Compute the average negative log likelihood over a batch of visible samples
            NLL = - <log(p(v))> = - <F> + log(Z) '''
        free_energy = (tf.matmul(visible_samples, self.visible_bias)
                       + tf.reduce_sum(tf.nn.softplus(tf.matmul(visible_samples, self.weights)
                                                      + tf.transpose(self.hidden_bias)), 1))
        return -tf.reduce_mean(free_energy - log_Z)

    def split_samples(self, samples):
        if samples is None:
            return None

        visible_samples = tf.slice(samples, begin=(0, 0), size=(self.num_samples, self.num_visible))
        hidden_samples = tf.slice(samples, begin=(0, self.num_visible), size=(self.num_samples, self.num_hidden))
        return hidden_samples, visible_samples

    @staticmethod
    def sample_binary_tensor(prob, m, n):
        # type: (tf.Tensor, int, int) -> tf.Tensor
        """
        Convenience method for generating a binary Tensor using a given probability.

        :param prob: Tensor of shape (m, n)
        :param m: number of rows in result.
        :param n: number of columns in result.
        """
        return tf.where(
            tf.less(tf.random_uniform(shape=(m, n)), prob),
            tf.ones(shape=(m, n)),
            tf.zeros(shape=(m, n))
        )


import abc
import tensorflow as tf
from gpflow import Parameterized, Param, params_as_tensors, settings
import numpy as np

float_type = settings.dtypes.float_type


class Reward(Parameterized):
    def __init__(self):
        Parameterized.__init__(self)

    @abc.abstractmethod
    def compute_reward(self, m, s):
        raise NotImplementedError


class ExponentialReward(Reward):
    def __init__(self, state_dim):
        Reward.__init__(self)
        self.state_dim = state_dim
        self.W = Param(np.random.rand(state_dim, state_dim), trainable=False)
        self.t = Param(np.random.rand(1, state_dim), trainable=False)

    @params_as_tensors
    def compute_reward(self, m, s):
        '''
        Reward function, calculating mean and variance of rewards, given
        mean and variance of state distribution, along with the target State
        and a weight matrix.
        Input m : [1, k]
        Input s : [k, k]

        Output M : [1, 1]
        Output S  : [1, 1]
        '''
        # TODO: Clean up this

        SW = s @ self.W

        iSpW = tf.transpose(
                tf.matrix_solve( (tf.eye(self.state_dim, dtype=float_type) + SW),
                tf.transpose(self.W), adjoint=True))

        muR = tf.exp(-(m-self.t) @  iSpW @ tf.transpose(m-self.t)/2) / \
                tf.sqrt( tf.linalg.det(tf.eye(self.state_dim, dtype=float_type) + SW) )

        i2SpW = tf.transpose(
                tf.matrix_solve( (tf.eye(self.state_dim, dtype=float_type) + 2*SW),
                tf.transpose(self.W), adjoint=True))

        r2 =  tf.exp(-(m-self.t) @ i2SpW @ tf.transpose(m-self.t)) / \
                tf.sqrt( tf.linalg.det(tf.eye(self.state_dim, dtype=float_type) + 2*SW) )

        sR = r2 - muR @ muR
        muR.set_shape([1, 1])
        sR.set_shape([1, 1])
        return muR, sR
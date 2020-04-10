# from collections import abc
import tensorflow as tf
from gpflow import Parameter, Module
import numpy as np

from gpflow import config
float_type = config.default_float()


# class Reward(Module):
#     @abc.abstractmethod
#     def compute_reward(self, m, s):
#         raise NotImplementedError


class ExponentialReward(Module):
    def __init__(self, state_dim, W=None, t=None):
        self.state_dim = state_dim
        if W is not None:
            self.W = Parameter(np.reshape(W, (state_dim, state_dim)), trainable=False)
        else:
            self.W = Parameter(np.eye(state_dim), trainable=False)
        if t is not None:
            self.t = Parameter(np.reshape(t, (1, state_dim)), trainable=False)
        else:
            self.t = Parameter(np.zeros((1, state_dim)), trainable=False)

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
                tf.linalg.solve( (tf.eye(self.state_dim, dtype=float_type) + SW),
                tf.transpose(self.W), adjoint=True))

        muR = tf.exp(-(m-self.t) @  iSpW @ tf.transpose(m-self.t)/2) / \
                tf.sqrt( tf.linalg.det(tf.eye(self.state_dim, dtype=float_type) + SW) )

        i2SpW = tf.transpose(
                tf.linalg.solve( (tf.eye(self.state_dim, dtype=float_type) + 2*SW),
                tf.transpose(self.W), adjoint=True))

        r2 =  tf.exp(-(m-self.t) @ i2SpW @ tf.transpose(m-self.t)) / \
                tf.sqrt( tf.linalg.det(tf.eye(self.state_dim, dtype=float_type) + 2*SW) )

        sR = r2 - muR @ muR
        muR.set_shape([1, 1])
        sR.set_shape([1, 1])
        return muR, sR

class LinearReward(Module):
    def __init__(self, state_dim, W):
        self.state_dim = state_dim
        self.W = Parameter(np.reshape(W, (state_dim, 1)), trainable=False)

    def compute_reward(self, m, s):
        muR = tf.reshape(m, (1, self.state_dim)) @ self.W
        sR = tf.transpose(self.W) @ s @ self.W
        return muR, sR


class CombinedRewards(Module):
    def __init__(self, state_dim, rewards=[], coefs=None):
        self.state_dim = state_dim
        self.base_rewards = rewards
        if coefs is not None:
            self.coefs = Parameter(coefs, trainable=False)
            # self.coefs = tf.Variable(coefs, dtype=float_type, trainable=False)
        else:
            self.coefs = Parameter(np.ones(len(rewards)), dtype=float_type, trainable=False)
            #self.coefs = tf.Variable(np.ones(len(rewards)), dtype=float_type, trainable=False)

    def compute_reward(self, m, s):
        muR = 0
        sR = 0
        for c,r in enumerate(self.base_rewards):
            tmp1, tmp2 = r.compute_reward(m, s)
            muR += self.coefs[c] * tmp1
            sR += self.coefs[c]**2 * tmp2
        return muR, sR

import tensorflow as tf
import numpy as np
import gpflow

from .models import MGPR
from gpflow import settings
float_type = settings.dtypes.float_type

def squash_sin(m, s, e=None):
    '''
    Squashing function, passing the controls mean and variance
    through a sinus, as in gSin.m.
    '''
    k = tf.shape(m)[1]
    if e is None:
        e = 10*tf.ones((1,k), dtype=float_type)  #squashes in [-1,1] by default
    mu = e*tf.exp(-tf.diag_part(s) / 2) * tf.sin(m)

    lq = -(tf.reshape(tf.diag_part(s), shape=[k, 1])
           + tf.reshape(tf.diag_part(s), shape=[1, k])) / 2
    q = tf.exp(lq)
    su = (tf.exp(lq + s) - q) * tf.cos(tf.reshape(m, shape=[k, 1])
                                       - tf.reshape(m, shape=[1, k])) \
         - (tf.exp(lq - s) - q) * tf.cos(tf.reshape(m, shape=[k, 1])
                                         + tf.reshape(m, shape=[1, k]))
    su = tf.reshape(e, shape=[1, k]) * tf.reshape(e, shape=[k, 1]) * su / 2
    #C = tf.diag( tf.transpose(e) @ tf.exp(-tf.diag_part(s)/2) * tf.cos(m))
    C = e*tf.diag( tf.exp(-tf.diag_part(s)/2) * tf.cos(m))
    return mu, su, tf.reshape(C,shape=[k,k])


class LinearController(gpflow.Parameterized):
    def __init__(self, state_dim, control_dim, W=None, b=None):
        gpflow.Parameterized.__init__(self)
        self.W = gpflow.Param(np.random.rand(control_dim, state_dim))
        self.b = gpflow.Param(np.random.rand(1, control_dim))

    @gpflow.params_as_tensors
    def compute_action(self, m, s, squash=True):
        '''
        Simple affine action:  M <- W(m-t) - b
        IN: mean (m) and variance (s) of the state
        OUT: mean (M) and variance (S) of the action
        '''
        M = m @ tf.transpose(self.W) + self.b # mean output
        S = self.W @ s @ tf.transpose(self.W) # output variance
        V = tf.transpose(self.W) #input output covariance
        if squash:
            M, S, V2 = squash_sin(M, S)
            V = V @ V2
        return M, S, V


class RBF_Controller(MGPR):
    def __init__(self, points, values):
        MGPR.__init__(self, points, values)
        for model in self.models:
            model.kern.variance = 1.0
            model.kern.variance.trainable = False

    @gpflow.params_as_tensors
    def compute_action(self, m, s, squash=True):
        '''
        RBF Controller. See Deisenroth's Thesis Section
        IN: mean (m) and variance (s) of the state
        OUT: mean (M) and variance (S) of the action
        '''
        iK, beta = self.calculate_factorizations()
        M, S, V = self.predict_given_factorizations(m, s, 0.0*iK, beta)
        S = S - tf.diag(self.variance - 1e-6)
        if squash:
            M, S, V2 = squash_sin(M, S)
            V = V @ V2
        return M, S, V

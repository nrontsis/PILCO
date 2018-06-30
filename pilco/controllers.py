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
        e = tf.ones((1,k), dtype=float_type)  #squashes in [-1,1] by default
    else:
        e = e * tf.ones((1,k), dtype=float_type)
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
    def __init__(self, state_dim, control_dim, W=None, b=None, e=None):
        gpflow.Parameterized.__init__(self)
        self.W = gpflow.Param(np.random.rand(control_dim, state_dim))
        self.b = gpflow.Param(np.random.rand(1, control_dim))
        self.e = e

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
            M, S, V2 = squash_sin(M, S, self.e)
            V = V @ V2
        return M, S, V


class FakeGPR(gpflow.Parameterized):
    def __init__(self, X, Y, kernel):
        gpflow.Parameterized.__init__(self)
        self.X = gpflow.Param(X)
        self.Y = gpflow.Param(Y)
        self.kern = kernel
        self.likelihood = gpflow.likelihoods.Gaussian()

class RbfController(MGPR):
    '''
    An RBF Controller implemented as a deterministic GP
    See Deisenroth et al 2015: Gaussian Processes for Data-Efficient Learning in Robotics and Control
    Section 5.3.2.
    '''
    def __init__(self, state_dim, control_dim, num_basis_functions, e=None):
        MGPR.__init__(self,
            np.random.rand(num_basis_functions, state_dim),
            np.random.rand(num_basis_functions, control_dim)
        )
        for model in self.models:
            model.kern.variance = 1.0
            model.kern.variance.trainable = False
            self.e = e

    def create_models(self, X, Y):
        self.models = gpflow.params.ParamList([])
        for i in range(self.num_outputs):
            kern = gpflow.kernels.RBF(input_dim=X.shape[1], ARD=True)
            self.models.append(FakeGPR(X, Y[:, i:i+1], kern))

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
            M, S, V2 = squash_sin(M, S, self.e)
            V = V @ V2
        return M, S, V

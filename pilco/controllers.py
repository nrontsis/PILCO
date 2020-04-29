import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
import gpflow
from gpflow import Parameter
from gpflow import set_trainable
from gpflow.utilities import positive
f64 = gpflow.utilities.to_default_float

from .models import MGPR
float_type = gpflow.config.default_float()

def squash_sin(m, s, max_action=None):
    '''
    Squashing function, passing the controls mean and variance
    through a sinus, as in gSin.m. The output is in [-max_action, max_action].
    IN: mean (m) and variance(s) of the control input, max_action
    OUT: mean (M) variance (S) and input-output (C) covariance of the squashed
         control input
    '''
    k = tf.shape(m)[1]
    if max_action is None:
        max_action = tf.ones((1,k), dtype=float_type)  #squashes in [-1,1] by default
    else:
        max_action = max_action * tf.ones((1,k), dtype=float_type)

    M = max_action * tf.exp(-tf.linalg.diag_part(s) / 2) * tf.sin(m)

    lq = -( tf.linalg.diag_part(s)[:, None] + tf.linalg.diag_part(s)[None, :]) / 2
    q = tf.exp(lq)
    S = (tf.exp(lq + s) - q) * tf.cos(tf.transpose(m) - m) \
        - (tf.exp(lq - s) - q) * tf.cos(tf.transpose(m) + m)
    S = max_action * tf.transpose(max_action) * S / 2

    C = max_action * tf.linalg.diag( tf.exp(-tf.linalg.diag_part(s)/2) * tf.cos(m))
    return M, S, tf.reshape(C,shape=[k,k])


class LinearController(gpflow.Module):
    def __init__(self, state_dim, control_dim, max_action=1.0, t=None):
        gpflow.Module.__init__(self)
        self.W = Parameter(np.random.rand(control_dim, state_dim))
        self.b = Parameter(np.random.rand(1, control_dim))
        if t is None:
            self.t = 0.0
        else:
            self.t = tf.constant(t, shape=[1, state_dim])
        self.max_action = max_action

    def compute_action(self, m, s, squash=True):
        '''
        Simple affine action:  M <- W(m-t) - b
        IN: mean (m) and variance (s) of the state
        OUT: mean (M) and variance (S) of the action
        '''
        M = (m - self.t) @ tf.transpose(self.W) + self.b # mean output
        S = self.W @ s @ tf.transpose(self.W) # output variance
        V = tf.transpose(self.W) #input output covariance
        if squash:
            M, S, V2 = squash_sin(M, S, self.max_action)
            V = V @ V2
        return M, S, V

    def randomize(self):
        mean = 0; sigma = 1
        self.W.assign(mean + sigma*np.random.normal(size=self.W.shape))
        self.b.assign(mean + sigma*np.random.normal(size=self.b.shape))


class FakeGPR(gpflow.Module):
    def __init__(self, data, kernel, X=None, likelihood_variance=1e-4):
        gpflow.Module.__init__(self)
        if X is None:
            self.X = Parameter(data[0], name="DataX", dtype=gpflow.default_float())
        else:
            self.X = X
        self.Y = Parameter(data[1], name="DataY", dtype=gpflow.default_float())
        self.data = [self.X, self.Y]
        self.kernel = kernel
        self.likelihood = gpflow.likelihoods.Gaussian()
        self.likelihood.variance.assign(likelihood_variance)
        set_trainable(self.likelihood.variance, False)

class RbfController(MGPR):
    '''
    An RBF Controller implemented as a deterministic GP
    See Deisenroth et al 2015: Gaussian Processes for Data-Efficient Learning in Robotics and Control
    Section 5.3.2.
    '''
    def __init__(self, state_dim, control_dim, num_basis_functions, max_action=1.0):
        MGPR.__init__(self,
            [np.random.randn(num_basis_functions, state_dim),
            0.1*np.random.randn(num_basis_functions, control_dim)]
        )
        for model in self.models:
            model.kernel.variance.assign(1.0)
            set_trainable(model.kernel.variance, False)
        self.max_action = max_action

    def create_models(self, data):
        self.models = []
        for i in range(self.num_outputs):
            kernel = gpflow.kernels.SquaredExponential(lengthscales=tf.ones([data[0].shape[1],], dtype=float_type))
            transformed_lengthscales = Parameter(kernel.lengthscales, transform=positive(lower=1e-3))
            kernel.lengthscales = transformed_lengthscales
            kernel.lengthscales.prior = tfd.Gamma(f64(1.1),f64(1/10.0))
            if i == 0:
                self.models.append(FakeGPR((data[0], data[1][:,i:i+1]), kernel))
            else:
                self.models.append(FakeGPR((data[0], data[1][:,i:i+1]), kernel, self.models[-1].X))

    def compute_action(self, m, s, squash=True):
        '''
        RBF Controller. See Deisenroth's Thesis Section
        IN: mean (m) and variance (s) of the state
        OUT: mean (M) and variance (S) of the action
        '''
        with tf.name_scope("controller") as scope:
            iK, beta = self.calculate_factorizations()
            M, S, V = self.predict_given_factorizations(m, s, 0.0 * iK, beta)
            S = S - tf.linalg.diag(self.variance - 1e-6)
        if squash:
            M, S, V2 = squash_sin(M, S, self.max_action)
            V = V @ V2
        return M, S, V

    def randomize(self):
        print("Randomising controller")
        for m in self.models:
            m.X.assign(np.random.normal(size=m.data[0].shape))
            m.Y.assign(self.max_action / 10 * np.random.normal(size=m.data[1].shape))
            mean = 1; sigma = 0.1
            m.kernel.lengthscales.assign(mean + sigma*np.random.normal(size=m.kernel.lengthscales.shape))

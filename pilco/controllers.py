import tensorflow as tf
import numpy as np
import gpflow

from .models import MGPR


class LinearController(gpflow.Parameterized):
    def __init__(self, state_dim, control_dim, W=None, b=None):
        gpflow.Parameterized.__init__(self)
        self.W = gpflow.Param(np.random.rand(control_dim, state_dim))
        self.b = gpflow.Param(np.random.rand(1, control_dim))

    @gpflow.params_as_tensors
    def compute_action(self, m, s):
        '''
        Simple affine action:  M <- W(m-t) - b
        IN: mean (m) and variance (s) of the state
        OUT: mean (M) and variance (S) of the action
        '''
        M = m @ tf.transpose(self.W) + self.b # mean output
        S = self.W @ s @ tf.transpose(self.W) # output variance
        V = tf.transpose(self.W) #input output covariance
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
    def __init__(self, state_dim, control_dim, num_basis_functions):
        MGPR.__init__(self,
            np.random.rand(num_basis_functions, state_dim),
            np.random.rand(num_basis_functions, control_dim)
        )
        for model in self.models:
            model.kern.variance = 1.0
            model.kern.variance.trainable = False

    def create_models(self, X, Y):
        self.models = gpflow.params.ParamList([])
        for i in range(self.num_outputs):
            kern = gpflow.kernels.RBF(input_dim=X.shape[1], ARD=True)
            self.models.append(FakeGPR(X, Y[:, i:i+1], kern))

    def compute_action(self, m, s):
        '''
        RBF Controller. See Deisenroth's Thesis Section
        IN: mean (m) and variance (s) of the state
        OUT: mean (M) and variance (S) of the action
        '''
        iK, beta = self.calculate_factorizations()
        M, S, V = self.predict_given_factorizations(m, s, 0.0*iK, beta)
        S = S - tf.diag(self.variance - 1e-6)
        return M, S, V
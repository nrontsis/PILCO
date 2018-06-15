import tensorflow as tf
import numpy as np
import gpflow

from .models import MGPR


class LinearController(gpflow.Parameterized):
    def __init__(self, state_dim, control_dim):
        gpflow.Parameterized.__init__(self)
        self.W = gpflow.Param(np.zeros((control_dim, state_dim)))
        self.t = gpflow.Param(np.zeros((1, state_dim)))
        self.b = gpflow.Param(np.zeros((1, control_dim)))

    @gpflow.params_as_tensors
    def compute_action(self, m, s):
        '''
        Simple affine action:  M <- W(m-t) - b
        IN: mean (m) and variance (s) of the state
        OUT: mean (M) and variance (S) of the action
        '''
        M = (m-self.t) @ tf.transpose(self.W) - self.b # mean output
        S = self.W @ s @ tf.transpose(self.W) # output variance
        V = tf.transpose(self.W) #input output covariance
        return M, S, V


class RBF_Controller(MGPR):
    def __init__(self):
        MGPR.__init__(self)
        for model in self.models:
            model.kern.variance = 1.0
            model.kern.variance.trainable = False

    @gpflow.params_as_tensors
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

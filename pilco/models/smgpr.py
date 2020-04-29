import gpflow
import tensorflow as tf
import numpy as np

from .mgpr import MGPR

from gpflow import config
float_type = config.default_float()


class SMGPR(MGPR):
    def __init__(self, data, num_induced_points, name=None):
        self.num_induced_points = num_induced_points
        MGPR.__init__(self, data, name)

    def create_models(self, data):
        self.models = []
        for i in range(self.num_outputs):
            kern = gpflow.kernels.SquaredExponential(lengthscales=tf.ones([data[0].shape[1],], dtype=float_type))
            Z = np.random.rand(self.num_induced_points, self.num_dims)
            #TODO: Maybe fix noise for better conditioning
            self.models.append(gpflow.models.GPRFITC((data[0], data[1][:, i:i+1]), kern, inducing_variable=Z))

    def calculate_factorizations(self):
        batched_eye = tf.eye(self.num_induced_points, batch_shape=[self.num_outputs], dtype=float_type)
        # TODO: Change 1e-6 to the respective constant of GPflow
        Kmm = self.K(self.Z) + 1e-6 * batched_eye
        Kmn = self.K(self.Z, self.X)
        L = tf.linalg.cholesky(Kmm)
        V = tf.linalg.triangular_solve(L, Kmn)
        G = self.variance[:, None] - tf.reduce_sum(tf.square(V), axis=[1])
        G = tf.sqrt(1.0 + G/self.noise[:, None])
        V = V/G[:, None]
        Am = tf.linalg.cholesky(tf.matmul(V, V, transpose_b=True) + \
                self.noise[:, None, None] * batched_eye)
        At = tf.matmul(L, Am)
        iAt = tf.linalg.triangular_solve(At, batched_eye)
        Y_ = tf.transpose(self.Y)[:, :, None]
        beta = tf.linalg.triangular_solve(L,
            tf.linalg.cholesky_solve(Am, (V/G[:, None]) @ Y_),
            adjoint=True
        )[:, :, 0]
        iB = tf.matmul(iAt, iAt, transpose_a=True) * self.noise[:, None, None]
        iK = tf.linalg.cholesky_solve(L, batched_eye) - iB
        return iK, beta

    def centralized_input(self, m):
        return self.Z - m

    @property
    def Z(self):
        return self.models[0].inducing_variable.Z

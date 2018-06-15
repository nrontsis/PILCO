import gpflow
import tensorflow as tf
import numpy as np

from .mgpr import MGPR

float_type = gpflow.settings.dtypes.float_type


class SMGPR(MGPR):
    def __init__(self, X, Y, num_induced_points, name=None):
        gpflow.Parameterized.__init__(self, name)
        self.num_induced_points = num_induced_points
        MGPR.__init__(self, X, Y, name)

    def create_models(self, X, Y):
        self.models = []
        for i in range(self.num_outputs):
            kern = gpflow.kernels.RBF(input_dim=X.shape[1], ARD=True)
            Z = np.random.rand(self.num_induced_points, self.num_dims)
            #TODO: Maybe fix noise for better conditioning
            self.models.append(gpflow.models.SGPR(X, Y[:, i:i+1], kern, Z=Z))
            self.models[i].clear(); self.models[i].compile()
    
    def calculate_factorizations(self):
        batched_eye = tf.eye(self.num_induced_points, batch_shape=[self.num_outputs], dtype=float_type)
        # TODO: Change 1e-6 to the respective constant of GPflow
        Kmm = self.K(self.Z) + 1e-6 * batched_eye
        Kmn = self.K(self.Z, self.X)
        L = tf.cholesky(Kmm)
        V = tf.matrix_triangular_solve(L, Kmn)
        G = self.variance[:, None] - tf.reduce_sum(tf.square(V), axis=[1])
        G = tf.sqrt(1.0 + G/self.noise[:, None])
        V = V/G[:, None]
        Am = tf.cholesky(tf.matmul(V, V, transpose_b=True) + \
                self.noise[:, None, None] * batched_eye)
        At = tf.matmul(L, Am)
        iAt = tf.matrix_triangular_solve(At, batched_eye)
        Y_ = tf.transpose(self.Y)[:, :, None]
        beta = tf.matrix_triangular_solve(L,
            tf.cholesky_solve(Am, (V/G[:, None]) @ Y_),
            adjoint=True
        )[:, :, 0]
        iB = tf.matmul(iAt, iAt, transpose_a=True) * self.noise[:, None, None]
        iK = tf.cholesky_solve(L, batched_eye) - iB

        return iK, beta

    def centralized_input(self, m):
        return self.Z - m

    @property
    def Z(self):
        return self.models[0].feature.Z.parameter_tensor
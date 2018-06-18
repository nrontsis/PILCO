import numpy as np
import tensorflow as tf
import gpflow

from .mgpr import MGPR
from .smgpr import SMGPR
from .. import controllers
from .. import rewards

float_type = gpflow.settings.dtypes.float_type


class PILCO(gpflow.models.Model):
    def __init__(self, X, Y, num_induced_points=None, horizon=30, controller=None, reward=None, name=None):
        super(PILCO, self).__init__(name)
        if not num_induced_points:
            self.mgpr = MGPR(X, Y)
        else:
            self.mgpr = SMGPR(X, Y, num_induced_points)
        self.state_dim = Y.shape[1]
        self.control_dim = X.shape[1] - Y.shape[1]
        self.horizon = horizon

        if controller is None:
            self.controller = controllers.LinearController(self.state_dim, self.control_dim)
        if reward is None:
            self.reward = rewards.ExponentialReward(self.state_dim)

        iK, beta = self.mgpr.get_factorizations()
        self.iK = gpflow.DataHolder(iK)
        self.beta = gpflow.DataHolder(beta)

    @gpflow.name_scope('likelihood')
    @gpflow.params_as_tensors
    def _build_likelihood(self):
        # This is for tunign controller's parameters

        #TODO: m0 and S0 could come from the environment
        m0 = np.zeros([1,self.state_dim])
        S0 = np.diag(np.ones(self.state_dim) * 0.01)

        reward = self.predict(m0, S0, self.horizon)[2]
        return reward

    def optimize(self):
        '''
        Optimizes both GP's and controller's hypeparamemeters.
        '''
        import time
        start = time.time()
        self.mgpr.optimize()
        end = time.time()
        print("GPs' optimization:", end - start, "seconds")
        start = time.time()
        iK, beta = self.mgpr.get_factorizations()
        self.iK = iK
        self.beta = beta
        optimizer = gpflow.train.ScipyOptimizer()
        optimizer.minimize(self, maxiter=100, disp=True)
        end = time.time()
        print("Controller's optimization:", end - start, "seconds")

    @gpflow.autoflow((float_type,[None, None]))
    def compute_action(self, x_m):
        return self.controller.compute_action(x_m, tf.zeros([self.state_dim, self.state_dim], float_type))[0]

    @gpflow.autoflow((float_type,[None, None]), (float_type,[None, None]))
    @gpflow.params_as_tensors
    def predict_wrapper(self, m_x, s_x):
        return self.predict(m_x, s_x, 30)

    @gpflow.autoflow((float_type,[None, None]), (float_type,[None, None]))
    @gpflow.params_as_tensors
    def grad_predict_wrapper(self, m_x, s_x):
        out = self.predict(m_x, s_x, 30)
        return tf.gradients(out, [self.controller.W, self.controller.b])

    def predict(self, m_x, s_x, n):
        loop_vars = [
            tf.constant(0, tf.int32),
            m_x,
            s_x,
            tf.constant([[0]], float_type)
        ]

        _, m_x, s_x, reward = tf.while_loop(
            # Termination condition
            lambda j, m_x, s_x, reward: j < n,
            # Body function
            lambda j, m_x, s_x, reward: (
                j + 1,
                *self.propagate(m_x, s_x),
                tf.add(reward, self.reward.compute_reward(m_x, s_x)[0])
            ), loop_vars
        )

        return m_x, s_x, reward

    def propagate(self, m_x, s_x):
        m_u, s_u, c_xu = self.controller.compute_action(m_x, s_x)

        m = tf.concat([m_x, m_u], axis=1)
        s1 = tf.concat([s_x, s_x@c_xu], axis=1)
        s2 = tf.concat([tf.transpose(s_x@c_xu), s_u], axis=1)
        s = tf.concat([s1, s2], axis=0)

        # Comment/uncomment the following lines to see the difference 
        # M_dx, S_dx, C_dx = self.mgpr.predict_on_noisy_inputs(m, s)
        M_dx, S_dx, C_dx = self.mgpr.predict_given_factorizations(m, s, self.iK, self.beta)
        M_x = M_dx + m_x
        #TODO: cleanup the following line
        S_x = S_dx + s_x + s1@C_dx + tf.matmul(C_dx, s1, transpose_a=True, transpose_b=True)

        # While-loop requires the shapes of the outputs to be fixed
        M_x.set_shape([1, self.state_dim]); S_x.set_shape([self.state_dim, self.state_dim])
        return M_x, S_x
from pilco.models import PILCO
import tensorflow as tf
import gpflow
import time
import numpy as np
import pandas as pd

from pilco.models.mgpr import MGPR
from pilco.models.smgpr import SMGPR
from pilco import controllers
from pilco import rewards
from gpflow import params_as_tensors, Parameterized

float_type = gpflow.settings.dtypes.float_type

# Awkward way to side-step gpflow issue - under investigation
class Mu_wrap(Parameterized):
    def __init__(self, mu=-1):
        Parameterized.__init__(self)
        self.mu = gpflow.params.Parameter(mu, dtype=float_type)
        self.mu.trainable = False

    @params_as_tensors
    def get_mu(self):
        return self.mu


class SafePILCO(gpflow.models.Model):
    def __init__(self, X, Y, num_induced_points=None, horizon=30, controller=None,
            reward_add=None, reward_mult=None, m_init=None, S_init=None, name=None, mu=5.0):
        # super(SafePILCO, self).__init__(X, Y, num_induced_points=num_induced_points, horizon=horizon, controller=controller,
        #         reward=reward_add, m_init=m_init, S_init=S_init)
        super(SafePILCO, self).__init__()
        if reward_mult is None:
            raise exception("have to define multiplicative reward")

        self.mu = Mu_wrap(mu=mu)

        self.reward_mult = reward_mult
        if not num_induced_points:
            self.mgpr = MGPR(X, Y)
        else:
            self.mgpr = SMGPR(X, Y, num_induced_points)
        self.state_dim = Y.shape[1]
        self.control_dim = X.shape[1] - Y.shape[1]
        self.horizon = horizon

        if controller is None:
            self.controller = controllers.LinearController(self.state_dim, self.control_dim)
        else:
            self.controller = controller

        if reward_add is None:
            self.reward_add = rewards.ExponentialReward(self.state_dim)
        else:
            self.reward_add = reward_add

        if m_init is None or S_init is None:
            # If the user has not provided an initial state for the rollouts,
            # then define it as the first state in the dataset.
            self.m_init = X[0:1, 0:self.state_dim]
            self.S_init = np.diag(np.ones(self.state_dim) * 0.1)
        else:
            self.m_init = m_init
            self.S_init = S_init
        self.optimizer = None

    def propagate(self, m_x, s_x):
        m_u, s_u, c_xu = self.controller.compute_action(m_x, s_x)

        m = tf.concat([m_x, m_u], axis=1)
        s1 = tf.concat([s_x, s_x@c_xu], axis=1)
        s2 = tf.concat([tf.transpose(s_x@c_xu), s_u], axis=1)
        s = tf.concat([s1, s2], axis=0)

        M_dx, S_dx, C_dx = self.mgpr.predict_on_noisy_inputs(m, s)
        M_x = M_dx + m_x
        #TODO: cleanup the following line
        S_x = S_dx + s_x + s1@C_dx + tf.matmul(C_dx, s1, transpose_a=True, transpose_b=True)

        # While-loop requires the shapes of the outputs to be fixed
        M_x.set_shape([1, self.state_dim]); S_x.set_shape([self.state_dim, self.state_dim])
        return M_x, S_x

    #@params_as_tensors
    def predict(self, m_x, s_x, n):
        loop_vars = [
            tf.constant(0, tf.int32),
            m_x,
            s_x,
            tf.constant([[0]], float_type),
            tf.constant([[1.0]], float_type)
        ]

        _, m_x, s_x, reward_add, reward_mult = tf.while_loop(
            # Termination condition
            lambda j, m_x, s_x, reward_add, reward_mult: j < n,
            # Body function
            lambda j, m_x, s_x, reward_add, reward_mult: (
                j + 1,
                *self.propagate(m_x, s_x),
                tf.add(reward_add, self.reward_add.compute_reward(m_x, s_x)[0]),
                tf.multiply(reward_mult, 1.0-self.reward_mult.compute_reward(m_x, s_x)[0])
            ), loop_vars
        )
        reward_total = reward_add + self.mu.get_mu() * (1.0 - reward_mult)
        return m_x, s_x, reward_total

    @gpflow.autoflow((float_type,[None, None]))
    def compute_action(self, x_m):
        return self.controller.compute_action(x_m, tf.zeros([self.state_dim, self.state_dim], float_type))[0]

    @gpflow.name_scope('likelihood')
    def _build_likelihood(self):
        # This is for tuning controller's parameters
        reward = self.predict(self.m_init, self.S_init, self.horizon)[2]
        return reward


    @gpflow.autoflow()
    def compute_reward(self):
        return self._build_likelihood()

    def optimize_models(self, maxiter=200, restarts=1):
        '''
        Optimize GP models
        '''
        self.mgpr.optimize(restarts=restarts)
        # Print the resulting model parameters
        # ToDo: only do this if verbosity is large enough
        lengthscales = {}; variances = {}; noises = {};
        i = 0
        for model in self.mgpr.models:
            lengthscales['GP' + str(i)] = model.kern.lengthscales.value
            variances['GP' + str(i)] = np.array([model.kern.variance.value])
            noises['GP' + str(i)] = np.array([model.likelihood.variance.value])
            i += 1
        print('-----Learned models------')
        pd.set_option('precision', 3)
        print('---Lengthscales---')
        print(pd.DataFrame(data=lengthscales))
        print('---Variances---')
        print(pd.DataFrame(data=variances))
        print('---Noises---')
        print(pd.DataFrame(data=noises))


    def optimize_policy(self, maxiter=50, restarts=1):
        '''
        Optimize controller's parameter's
        '''
        start = time.time()
        if not self.optimizer:
            self.optimizer = gpflow.train.ScipyOptimizer(method="L-BFGS-B")
            self.optimizer.minimize(self, maxiter=maxiter)
        else:
            session = self.optimizer._model.enquire_session(None)
            self.optimizer.minimize(self, maxiter=maxiter, session=session)
        end = time.time()
        print("Controller's optimization: done in %.1f seconds with reward=%.3f." % (end - start, self.compute_reward()))
        restarts -= 1

        session = self.optimizer._model.enquire_session(None)
        best_parameters = self.read_values(session=session)
        best_reward = self.compute_reward()
        for restart in range(restarts):
            print("Before randomisation ", self.compute_reward())
            self.controller.randomize()
            print("After randomisation", self.compute_reward())
            start = time.time()
            self.optimizer.minimize(self, maxiter=maxiter, session=session)
            end = time.time()
            reward = self.compute_reward()
            print("Controller's optimization: done in %.1f seconds with reward=%.3f." % (end - start, self.compute_reward()))
            if reward > best_reward:
                best_parameters = self.read_values(session=session)
                best_reward = reward

        self.assign(best_parameters)
        end = time.time()

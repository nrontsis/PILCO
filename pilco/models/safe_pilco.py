from pilco.models import PILCO
import tensorflow as tf
import gpflow
import time
float_type = gpflow.settings.dtypes.float_type

class SafePILCO(PILCO):
    def __init__(self, X, Y, num_induced_points=None, horizon=30, controller=None,
            reward_add=None, reward_mult=None, m_init=None, S_init=None, name=None, mu=5.0):
        super(SafePILCO, self).__init__(X, Y, num_induced_points=num_induced_points, horizon=horizon, controller=controller,
                reward=reward_add, m_init=m_init, S_init=S_init)
        if reward_mult is None:
            raise exception("have to define multiplicative reward")
        self.mu = gpflow.params.Parameter(mu, dtype=float_type)
        self.mu.trainable = False
        self.reward_mult = reward_mult


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
                tf.add(reward_add, self.reward.compute_reward(m_x, s_x)[0]),
                tf.multiply(reward_mult, 1.0-self.reward_mult.compute_reward(m_x, s_x)[0])
            ), loop_vars
        )
        reward_total = reward_add + self.mu.value * (1.0 - reward_mult)
        return m_x, s_x, reward_total

    @gpflow.name_scope('likelihood')
    def _build_likelihood(self):
        # This is for tuning controller's parameters
        reward = self.predict(self.m_init, self.S_init, self.horizon)[2]
        return reward

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
from gpflow import Module

float_type = gpflow.config.default_float()
from gpflow import set_trainable

class SafePILCO(PILCO):
    def __init__(self, data, num_induced_points=None, horizon=30, controller=None,
            reward_add=None, reward_mult=None, m_init=None, S_init=None, name=None, mu=5.0):
        super(SafePILCO, self).__init__(data, num_induced_points=num_induced_points, horizon=horizon, controller=controller,
                reward=reward_add, m_init=m_init, S_init=S_init)
        if reward_mult is None:
            raise exception("have to define multiplicative reward")

        self.mu = gpflow.Parameter(mu, trainable=False)

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
        reward_total = reward_add + self.mu * (1.0 - reward_mult)
        return m_x, s_x, reward_total

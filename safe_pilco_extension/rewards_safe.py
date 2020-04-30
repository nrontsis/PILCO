import pilco
import tensorflow as tf
import numpy as np
from scipy.stats import mvn
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

import gpflow
from gpflow import Module
float_type = gpflow.config.default_float()
from gpflow import set_trainable

class RiskOfCollision(Module):
    def __init__(self, state_dim, low, high):
        Module.__init__(self)
        self.state_dim = state_dim
        self.low = tf.cast(low, float_type)
        self.high = tf.cast(high, float_type)

    def compute_reward(self, m, s):
        infl_diag_S = 2*tf.linalg.diag_part(s)
        dist1 = tfd.Normal(loc=m[0,0], scale=infl_diag_S[0])
        dist2 = tfd.Normal(loc=m[0,2], scale=infl_diag_S[2])
        risk = (dist1.cdf(self.high[0]) - dist1.cdf(self.low[0])) * (dist2.cdf(self.high[1]) - dist2.cdf(self.low[1]))
        return risk, 0.0001 * tf.ones(1, dtype=float_type)

class SingleConstraint(Module):
    def __init__(self, dim, high=None, low=None, inside=True):
        Module.__init__(self)
        if high is None:
            self.high = False
        else:
            self.high = high
        if low is None:
            self.low = False
        else:
            self.low = low
        if high is None and low is None:
            raise Exception("At least one of bounds (high,low) has to be defined")
        self.dim = tf.constant(dim, tf.int32)
        if inside:
            self.inside = True
        else:
            self.inside = False


    def compute_reward(self, m, s):
        # Risk refers to the space between the low and high value -> 1
        # otherwise self.in = 0
        if not self.high:
            dist = tfd.Normal(loc=m[0, self.dim], scale=s[self.dim, self.dim])
            risk = 1 - dist.cdf(self.low)
        elif not self.low:
            dist = tfd.Normal(loc=m[0, self.dim], scale=s[self.dim, self.dim])
            risk = dist.cdf(self.high)
        else:
            dist = tfd.Normal(loc=m[0, self.dim], scale=s[self.dim, self.dim])
            risk = dist.cdf(self.high) - dist.cdf(self.low)
        if not self.inside:
            risk = 1 - risk
        return risk, 0.0001 * tf.ones(1, dtype=float_type)

class ObjectiveFunction(Module):
    def __init__(self, reward_f, risk_f, mu=1.0):
        Module.__init__(self)
        self.reward_f = reward_f
        self.risk_f = risk_f
        self.mu = Parameter(mu, dtype=float_type, trainable=False)

    def compute_reward(self, m, s):
        reward, var = self.reward_f.compute_reward(m, s)
        risk, _ = self.risk_f.compute_reward(m, s)
        return reward - self.mu * risk, var

import pilco
# from pilco.rewards import ExponentialReward
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

        # Risk refers to the space between the low and high value -> 1
        #otherwise self.in = 0

    def compute_reward(self, m, s):
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


# if __name__=='__main__':
#     a = RiskOfCollision(2, [-10.0, -10.0], [10.0, 10.0])
#     m = np.array([2.0, 1, 3.5, 1])[None,:]
#     S = 2 * np.eye(4)
#
#     s = tf.Session()
#     print(s.run(a.compute_reward(m, S)))
#     r = ExponentialReward(4, W=np.eye(4), t=np.ones(4)[:,None])
#     ob = ObjectiveFunction(r, a)
#
#     print(reward_wrapper(r, m, S))
#     print(reward_wrapper(ob, m, S))
#     ob.mu.assign(3.0)
#     print(reward_wrapper(ob, m , S))
#
#     w1 = np.diag([0.2, 0.001, 0.2, 0.001])
#     t1 = np.array([3.0, 1.0, 3.0, 1.0])
#     R1 = ExponentialReward(state_dim=4, t=t1, W=w1)
#
#     w2 = np.diag([0.8, 0.0001, 0.8, 0.0001])
#     t2 = np.array([0.0,10.0,0.0,10.0])
#     R2 = ExponentialReward(4, W=w1, t=t1)
#
#     O = ObjectiveFunction(R1, R2, mu=5.0)
#     #For Swimmer
#     max_ang = 1.7453
#     C1 = SingleConstraint(2, low=-max_ang, high=max_ang, inside=False)
#     C2 = SingleConstraint(3, low=-max_ang, high=max_ang, inside=False)
#     from pilco.rewards import CombinedRewards
#     all = CombinedRewards(4, rewards=[R2, C1, C2], coefs=[1.0, -1.0, 2.0])
#     print(reward_wrapper(all, m, S))

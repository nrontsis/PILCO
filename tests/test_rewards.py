from pilco.rewards import ExponentialReward
import numpy as np
import os
from gpflow import autoflow
from gpflow import settings
import oct2py
octave = oct2py.Oct2Py()
dir_path = os.path.dirname(os.path.realpath("__file__")) + "/tests/Matlab Code"
octave.addpath(dir_path)

float_type = settings.dtypes.float_type

from pilco.utils import reward_wrapper

def test_reward():
    '''
    Test reward function by comparing to reward.m
    '''
    k = 2  # state dim
    m = np.random.rand(1, k)
    s = np.random.rand(k, k)
    s = s.dot(s.T)

    reward = ExponentialReward(k)
    W = reward.W.value
    t = reward.t.value

    M, S = reward_wrapper(reward, m, s)

    M_mat, _, _, S_mat = octave.reward(m.T, s, t.T, W, nout=4)

    np.testing.assert_allclose(M, M_mat)
    np.testing.assert_allclose(S, S_mat)


if __name__ == '__main__':
    test_reward()

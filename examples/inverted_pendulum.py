import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf
from gpflow import set_trainable
# from tensorflow import logging
np.random.seed(0)

from pilco.utils import rollout, policy

env = gym.make('InvertedPendulum-v2')
# Initial random rollouts to generate a dataset
X,Y, _, _ = rollout(env=env, pilco=None, random=True, timesteps=40, render=True)
for i in range(1,5):
    X_, Y_, _, _ = rollout(env=env, pilco=None, random=True,  timesteps=40, render=True)
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))


state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim
controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=10)
# controller = LinearController(state_dim=state_dim, control_dim=control_dim)

pilco = PILCO((X, Y), controller=controller, horizon=40)
# Example of user provided reward function, setting a custom target state
# R = ExponentialReward(state_dim=state_dim, t=np.array([0.1,0,0,0]))
# pilco = PILCO(X, Y, controller=controller, horizon=40, reward=R)

for rollouts in range(3):
    pilco.optimize_models()
    pilco.optimize_policy()
    import pdb; pdb.set_trace()
    X_new, Y_new, _, _ = rollout(env=env, pilco=pilco, timesteps=100, render=True)
    # Update dataset
    X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_data((X, Y))

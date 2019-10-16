import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf
from tensorflow import logging
np.random.seed(0)

from pilco.utils import rollout, policy

with tf.Session(graph=tf.Graph()) as sess:
    env = gym.make('InvertedPendulum-v2')
    # Initial random rollouts to generate a dataset
    X,Y = rollout(env=env, pilco=None, random=True, timesteps=40)
    for i in range(1,3):
        X_, Y_ = rollout(env=env, pilco=None, random=True,  timesteps=40)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))


    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=5)
    #controller = LinearController(state_dim=state_dim, control_dim=control_dim)

    pilco = PILCO(X, Y, controller=controller, horizon=40)
    # Example of user provided reward function, setting a custom target state
    # R = ExponentialReward(state_dim=state_dim, t=np.array([0.1,0,0,0]))
    # pilco = PILCO(X, Y, controller=controller, horizon=40, reward=R)

    # Example of fixing a parameter, optional, for a linear controller only
    #pilco.controller.b = np.array([[0.0]])
    #pilco.controller.b.trainable = False

    for rollouts in range(3):
        pilco.optimize_models()
        pilco.optimize_policy()
        import pdb; pdb.set_trace()
        X_new, Y_new = rollout(env=env, pilco=pilco, timesteps=100)
        print("No of ops:", len(tf.get_default_graph().get_operations()))
        # Update dataset
        X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
        pilco.mgpr.set_XY(X, Y)

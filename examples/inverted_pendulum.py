import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf
from tensorflow import logging
np.random.seed(0)


def rollout(policy, timesteps):
    X = []; Y = []
    env.reset()
    x, _, _, _ = env.step(0)
    for timestep in range(timesteps):
        env.render()
        u = policy(x)
        x_new, _, done, _ = env.step(u)
        if done: break
        X.append(np.hstack((x, u)))
        Y.append(x_new - x)
        x = x_new
    return np.stack(X), np.stack(Y)

def random_policy(x):
    return env.action_space.sample()

def pilco_policy(x):
    return pilco.compute_action(x[None, :])[0, :]

#with tf.Session(graph=tf.Graph()):
env = gym.make('InvertedPendulum-v2')
# Initial random rollouts to generate a dataset
X,Y = rollout(policy=random_policy, timesteps=10)
for i in range(1,3):
    X_, Y_ = rollout(policy=random_policy, timesteps=10)
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
    pilco.optimize()
    pilco.mgpr.try_restart(restarts=1)
    pilco.restart_controller(restarts=1)
    import pdb; pdb.set_trace()
    X_new, Y_new = rollout(policy=pilco_policy, timesteps=20)
    print("No of ops:", len(tf.get_default_graph().get_operations()))
    # Update dataset
    X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_XY(X, Y)

import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import LinearController
np.random.seed(0)

env = gym.make('InvertedPendulum-v2')

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


# Initial random rollouts to generate a dataset
def random_policy(x):
    return env.action_space.sample()
X1, Y1 = rollout(policy=random_policy, timesteps=40)
X2, Y2 = rollout(policy=random_policy, timesteps=40)
X = np.vstack((X1, X2))
Y = np.vstack((Y1, Y2))

# Define PILCO on three rollouts
pilco = PILCO(X, Y, horizon=20)
# Example of fixing a parameter
pilco.controller.b = np.array([[0.0]])
pilco.controller.b.trainable = False

def pilco_policy(x):
    return pilco.compute_action(x[None, :])[0, :]

for rollouts in range(2):
    pilco.optimize()
    import pdb; pdb.set_trace()
    X_new, Y_new = rollout(policy=pilco_policy, timesteps=100)
    # Update dataset
    X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_XY(X, Y)

import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RBF_Controller
np.random.seed(0)

env = gym.make('InvertedPendulum-v2')

def rollout(policy, timesteps):
    X = []; Y = []
    env.reset()
    x, _, _, _ = env.step(0)
    for timestep in range(timesteps):
        # env.render()
        u = policy(x)
        x_new, _, _, _ = env.step(u)

        X.append(np.hstack((x, u)))
        Y.append(x_new - x)
        x = x_new
    return np.stack(X), np.stack(Y)


# Initial random rollouts to generate a dataset
def random_policy(x):
    return env.action_space.sample()/5
X1, Y1 = rollout(policy=random_policy, timesteps=200)
X2, Y2 = rollout(policy=random_policy, timesteps=200)
X = np.vstack((X1, X2))
Y = np.vstack((Y1, Y2))

# Define PILCO on three rollouts
pilco = PILCO(X, Y, horizon=50)
# Example of fixing a parameter
pilco.controller.b = np.array([[0.0]])
pilco.controller.b.trainable = False

k = Y.shape[1]
m_x = np.random.rand(1, k)
s_x = np.random.rand(k, k)
s_x = s_x.T.dot(s_x)
import time
start = time.time()
pilco.predict_wrapper(m_x, s_x)
end = time.time()
print("Value:", end - start)
start = time.time()
pilco.grad_predict_wrapper(m_x, s_x)
end = time.time()
print("Grad:", end - start)
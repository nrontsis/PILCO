import gym
from ..models.pilco import PILCO
import numpy as np
import time
env = gym.make('InvertedPendulum-v2')

np.random.seed(0)
# Initial rollouts
X = []; Y = []
for rollouts in range(5):
    env.reset()
    x, _, _, _ = env.step(0)
    random_action = 0
    for i in range(30):
        env.render()
        random_action += env.action_space.sample()/5
        u = (-1.0)**(rollouts)*2*np.sin(i/10) + random_action
        X.append(np.hstack((x, u)))
        x_, _, _, _ = env.step(u)
        Y.append(x_ - x)
        x = x_

# Closed loop rollouts
for rollouts in range(10):
    start = time.time()
    pilco = PILCO(np.stack(X), np.stack(Y))
    pilco.optimize()
    end = time.time()
    print("Rollout #", rollouts, "done in ", end - start, "seconds.")
    import pdb; pdb.set_trace()
    env.reset()
    x, _, _, _ = env.step(0)
    for i in range(30):
        env.render()
        u = pilco.compute_action(x[None, :])
        X.append(np.hstack((x, u[0, :])))
        x_, _, _, _ = env.step(u)
        Y.append(x_ - x)
        x = x_

import gym
from ..models.pilco import PILCO
import numpy as np
np.random.seed(0)

env = gym.make('InvertedPendulum-v2')

X = []; Y = []
for rollouts in range(5):
    env.reset()
    x, _, _, _ = env.step(0)
    for i in range(50):
        env.render()
        u = env.action_space.sample()/5
        x_, _, _, _ = env.step(u)

        X.append(np.hstack((x, u)))
        Y.append(x_ - x)
        x = x_

pilco = PILCO(np.stack(X), np.stack(Y))
pilco.controller.t.trainable = False
pilco.controller.b.trainable = False
pilco.optimize()

import pdb; pdb.set_trace()
env.reset()
x, _, _, _ = env.step(0)
for i in range(50):
    env.render()
    u = pilco.compute_action(x[None, :])
    x_, _, _, _ = env.step(u)

    X.append(np.hstack((x, u[0, :])))
    Y.append(x_ - x)
    x = x_
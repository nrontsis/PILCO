import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
np.random.seed(0)

env = gym.make('InvertedPendulum-v2')

def rollout(policy, timesteps):
    X = []; Y = []
    env.reset()
    x, _, _, _ = env.step(0)
    for timestep in range(timesteps):
        env.render()
        u = policy(x)
        x_new, _, _, _ = env.step(u)

        X.append(np.hstack((x, u)))
        Y.append(x_new - x)
        x = x_new
    return np.stack(X), np.stack(Y)

def random_policy(x):
    return env.action_space.sample()/5

def pilco_policy(x):
    return pilco.compute_action(x[None, :])[0, :]

# Initial random rollouts to generate a dataset
X,Y = rollout(policy=random_policy, timesteps=40)
for i in range(1,5):
    X_, Y_ = rollout(policy=random_policy, timesteps=40)
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))


state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim
controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=10)
#controller = LinearController(state_dim=state_dim, control_dim=control_dim)
# Example of fixing a parameter
#pilco.controller.b = np.array([[0.0]])
#pilco.controller.b.trainable = False

pilco = PILCO(X, Y, controller=controller, horizon=10)

for rollouts in range(5):
    pilco.optimize()
    import pdb; pdb.set_trace()
    X_new, Y_new = rollout(policy=pilco_policy, timesteps=100)
    # Update dataset
    X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_XY(X, Y)

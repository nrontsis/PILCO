import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf
np.random.seed(0)
from pilco.utils import policy, rollout, Normalised_Env


SUBS = 5
T = 25
env = gym.make('MountainCarContinuous-v0')
# Initial random rollouts to generate a dataset
X1,Y1, _, _ = rollout(env=env, pilco=None, random=True, timesteps=T, SUBS=SUBS, render=True)
for i in range(1,5):
    X1_, Y1_,_,_ = rollout(env=env, pilco=None, random=True,  timesteps=T, SUBS=SUBS, render=True)
    X1 = np.vstack((X1, X1_))
    Y1 = np.vstack((Y1, Y1_))
env.close()

env = Normalised_Env('MountainCarContinuous-v0', np.mean(X1[:,:2],0), np.std(X1[:,:2], 0))
X = np.zeros(X1.shape)
X[:, :2] = np.divide(X1[:, :2] - np.mean(X1[:,:2],0), np.std(X1[:,:2], 0))
X[:, 2] = X1[:,-1] # control inputs are not normalised
Y = np.divide(Y1 , np.std(X1[:,:2], 0))

state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim
m_init =  np.transpose(X[0,:-1,None])
S_init =  0.5 * np.eye(state_dim)
controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=25)

R = ExponentialReward(state_dim=state_dim,
                      t=np.divide([0.5,0.0] - env.m, env.std),
                      W=np.diag([0.5,0.1])
                     )
pilco = PILCO((X, Y), controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)

best_r = 0
all_Rs = np.zeros((X.shape[0], 1))
for i in range(len(all_Rs)):
    all_Rs[i,0] = R.compute_reward(X[i,None,:-1], 0.001 * np.eye(state_dim))[0]

ep_rewards = np.zeros((len(X)//T,1))

for i in range(len(ep_rewards)):
    ep_rewards[i] = sum(all_Rs[i * T: i*T + T])

for model in pilco.mgpr.models:
    model.likelihood.variance.assign(0.05)
    set_trainable(model.likelihood.variance, False)

r_new = np.zeros((T, 1))
for rollouts in range(5):
    pilco.optimize_models()
    pilco.optimize_policy(maxiter=100, restarts=3)
    import pdb; pdb.set_trace()
    X_new, Y_new,_,_ = rollout(env=env, pilco=pilco, timesteps=T, SUBS=SUBS, render=True)

    for i in range(len(X_new)):
            r_new[:, 0] = R.compute_reward(X_new[i,None,:-1], 0.001 * np.eye(state_dim))[0]
    total_r = sum(r_new)
    _, _, r = pilco.predict(m_init, S_init, T)

    print("Total ", total_r, " Predicted: ", r)
    X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new));
    all_Rs = np.vstack((all_Rs, r_new)); ep_rewards = np.vstack((ep_rewards, np.reshape(total_r,(1,1))))
    pilco.mgpr.set_data((X, Y))

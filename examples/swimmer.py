import os
import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward, LinearReward, CombinedRewards
import tensorflow as tf
from pilco.utils import rollout, policy

seed = 0
np.random.seed(seed)
name = "swimmer_new" + str(seed)
env = gym.make('Swimmer-v2').env
state_dim = 8
control_dim = 2
SUBS = 5
maxiter = 80
max_action = 1.0
m_init = np.reshape(np.zeros(state_dim), (1,state_dim))  # initial state mean
S_init = 0.005 * np.eye(state_dim)
J = 10
N = 15
T = 15
bf = 40
T_sim = 50

# Reward function that dicourages the joints from hitting their max angles
max_ang = 95 / 180 * np.pi
rewards = []
rewards.append(LinearReward(state_dim, [0, 0, 0, 1.0, 0, 0, 0, 0]))
rewards.append(ExponentialReward(
    state_dim,
    W=np.diag(np.array([0, 0, 10, 0, 0, 0, 0, 0]) + 1e-6),
    t=[0, 0, max_ang, 0, 0, 0, 0, 0])
)
rewards.append(ExponentialReward(
    state_dim,
    W=np.diag(np.array([0, 0, 10, 0, 0, 0, 0, 0]) + 1e-6),
    t=[0, 0, -max_ang, 0, 0, 0, 0, 0])
)
rewards.append(ExponentialReward(
    state_dim,
    W=np.diag(np.array([0, 10, 0, 0, 0, 0, 0, 0]) + 1e-6),
    t=[0, max_ang, 0, 0, 0, 0, 0, 0])
)
rewards.append(ExponentialReward(
    state_dim,
    W=np.diag(np.array([0, 10, 0, 0, 0, 0, 0, 0]) + 1e-6),
    t=[0, -max_ang, 0, 0, 0, 0, 0, 0])
)
combined_reward = CombinedRewards(state_dim, rewards, coefs=[1.0, -1.0, -1.0, -1.0, -1.0])
# Initial random rollouts to generate a dataset
X,Y, _, _ = rollout(env, None, timesteps=T, random=True, SUBS=SUBS, verbose=True, render=False)
for i in range(1,J):
    X_, Y_, _, _ = rollout(env, None, timesteps=T, random=True, SUBS=SUBS, verbose=True, render=False)
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))

state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim
controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)

pilco = PILCO((X, Y), controller=controller, horizon=T, reward=combined_reward, m_init=m_init, S_init=S_init)

logging = False # To save results in .csv files turn this flag to True
eval_runs = 10
evaluation_returns_full = np.zeros((N, eval_runs))
evaluation_returns_sampled = np.zeros((N, eval_runs))
eval_max_timesteps = 1000//SUBS
X_eval=False
for rollouts in range(N):
    print("**** ITERATION no", rollouts, " ****")
    pilco.optimize_models(maxiter=maxiter, restarts=2)
    pilco.optimize_policy(maxiter=maxiter, restarts=2)

    X_new, Y_new, _, _ = rollout(env, pilco, timesteps=T_sim, verbose=True, SUBS=SUBS, render=True)

    cur_rew = 0
    for t in range(0,len(X_new)):
        cur_rew += Rew.compute_reward(X_new[t, 0:state_dim, None].transpose(), 0.0001 * np.eye(state_dim))[0]
        if t == T: print('On this episode, on the planning horizon, PILCO reward was: ', cur_rew)
    print('On this episode PILCO reward was ', cur_rew)

    gym_steps = 1000
    T_eval = gym_steps // SUBS
    # Update dataset
    X = np.vstack((X, X_new[:T,:])); Y = np.vstack((Y, Y_new[:T,:]))
    pilco.mgpr.set_data((X, Y))
    if logging:
        if eval_max_timesteps is None:
            eval_max_timesteps = sim_timesteps
        for k in range(0, eval_runs):
            [X_eval_, _,
            evaluation_returns_sampled[rollouts, k],
            evaluation_returns_full[rollouts, k]] = rollout(env, pilco,
                                                           timesteps=eval_max_timesteps,
                                                           verbose=False, SUBS=SUBS,
                                                           render=False)
            if not X_eval:
                X_eval = X_eval_.copy()
            else:
                X_eval = np.vstack((X_eval, X_eval_))
        np.savetxt("X_" + name + seed + ".csv", X, delimiter=',')
        np.savetxt("X_eval_" + name + seed + ".csv", X_eval, delimiter=',')
        np.savetxt("evaluation_returns_sampled_"  + name + seed + ".csv", evaluation_returns_sampled, delimiter=',')
        np.savetxt("evaluation_returns_full_" + name + seed+ ".csv", evaluation_returns_full, delimiter=',')

    # To save a video of a run
    # env2 = SwimmerWrapper(monitor=True)
    # rollout(env2, pilco, policy=policy, timesteps=T+50, verbose=True, SUBS=SUBS)
    # env2.env.close()

from master_alg import pilco_run, rollout
from pilco.utils import Normalised_Env
import numpy as np
import gym

number_of_random_seeds = 10

# Inverted Pendulum
env = gym.make('InvertedPendulum-v2')
for i in range(5, number_of_random_seeds):
    name = 'inv_pend_seed' + str(i) + '_'
    pilco_run(env, 4, 2, logging=True, eval_runs=10,
              restarts=2, eval_max_timesteps=100, seed=i, name=name)

# Mountain Car
SUBS=5
T = 25
for i in range(5, number_of_random_seeds):
    env = gym.make('MountainCarContinuous-v0')
    name = 'mountain_car_seed' + str(i) + '_'
    # Normalise before calling pilco_run
    # Initial random rollouts to generate a dataset
    X1, Y1, _, _ = rollout(env=env, pilco=None, random=True, timesteps=T, SUBS=SUBS)
    for i in range(1,5):
        X1_, Y1_, _, _ = rollout(env=env, pilco=None, random=True,  timesteps=T, SUBS=SUBS)
        X1 = np.vstack((X1, X1_))
        Y1 = np.vstack((Y1, Y1_))
    #env.close()
    env = Normalised_Env('MountainCarContinuous-v0', np.mean(X1[:,:2],0), np.std(X1[:,:2], 0))
    controller = {'type':'rbf', 'basis_functions':25}
    reward = {'type':'exp',
              't':np.divide([0.5,0.0] - env.m, env.std),
              'W':np.diag([0.5, 0.1])}
    pilco_run(env, 5, 5,
              SUBS=SUBS,
              restarts=3,
              maxiter=100,
              cont=controller,
              rew=reward,
              sim_timesteps=T,
              plan_timesteps=T,
              #m_init=np.reshape(np.divide(X1[0,:-1]-env.m, env.std), (1,2)),
              #S_init=np.eye(2),
              logging=True,
              eval_runs=5,
              eval_max_timesteps=T,
              name=name,
              seed=i)

# # Pendulum Swing Up
from pendulum_swing_up import myPendulum
env = myPendulum()
for i in range(number_of_random_seeds):
    name = 'inv_pend_' + str(i) + '_'
    controller = {'type':'rbf', 'basis_functions':30, 'max_action':2.0}
    reward = {'type':'exp', 't':np.array([1.0, 0.0, 0.0]), 'W':np.diag([2.0, 2.0, 0.3])}
    T = 40
    name='pend_swing_up' + str(i) + '_'
    pilco_run(env, 8, 4,
              SUBS=3,
              maxiter=50,
              restarts=2,
              m_init = np.reshape([-1.0, 0, 0.0], (1,3)),
              S_init = np.diag([0.01, 0.05, 0.01]),
              cont=controller,
              rew=reward,
              sim_timesteps=T,
              plan_timesteps=T,
              logging=True,
              eval_runs=5,
              eval_max_timesteps=T,
              fixed_noise=0.001,
              name=name,
              seed=i)

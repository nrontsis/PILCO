import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf
# from tensorflow import logging

from pilco.utils import policy


def rollout(env, pilco, timesteps, verbose=True, random=False, SUBS=1, render=False):
        X = []; Y = [];
        x = env.reset()
        ep_return_full = 0
        ep_return_sampled = 0
        for timestep in range(timesteps):
            if render: env.render()
            u = policy(env, pilco, x, random)
            for i in range(SUBS):
                x_new, r, done, _ = env.step(u)
                ep_return_full += r
                if done: break
                if render: env.render()
            if verbose:
                print("Action: ", u)
                print("State : ", x_new)
                print("Return so far: ", ep_return_full)
            X.append(np.hstack((x, u)))
            Y.append(x_new - x)
            ep_return_sampled += r
            x = x_new
            if done: break
        return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full


def pilco_run(env, N, J,
              safe=False,
              name='',
              seed=0,
              cont=None,
              rew=None,
              SUBS=1,
              sim_timesteps=50,
              plan_timesteps=30,
              restarts=1,
              maxiter=100,
              m_init=None, S_init=None,
              fixed_noise=None,
              logging=False, eval_runs=5, eval_max_timesteps=None,
              variable_episode_length=False):
    np.random.seed(seed)
    with tf.Session(graph=tf.Graph()) as sess:
        X, Y, _, _ = rollout(env=env, pilco=None, timesteps=sim_timesteps, random=True, SUBS=SUBS)
        for i in range(1,J):
            X_, Y_, _, _ = rollout(env=env, pilco=None,timesteps=sim_timesteps, random=True, SUBS=SUBS)
            X = np.vstack((X, X_))
            Y = np.vstack((Y, Y_))
        state_dim = Y.shape[1]
        control_dim = X.shape[1] - state_dim

        if cont is None:
            controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=5)
        elif cont['type']=='rbf':
            controller = RbfController(state_dim=state_dim, control_dim=control_dim,
                                       num_basis_functions=cont['basis_functions'], max_action=cont.get('max_action', 1.0))
        elif cont['type']=='linear':
            controller = LinearController(state_dim=state_dim, control_dim=control_dim,
                                          max_action=cont.get('max_action', 1.0))
        else:
            ValueError('Invalid Controller')

        if rew is None:
            reward=None
        elif rew['type']=='exp':
            reward = ExponentialReward(state_dim=state_dim, t=rew['t'], W=rew['W'])
        else:
            ValueError('This function only handles Exponential rewards for now')

        pilco = PILCO(X, Y, controller=controller, reward=reward, horizon=plan_timesteps, m_init=m_init, S_init=S_init)

        if fixed_noise is not None:
            for model in pilco.mgpr.models:
                model.likelihood.variance = fixed_noise
                model.likelihood.variance.trainable = False

        evaluation_returns_full = np.zeros((N, eval_runs))
        evaluation_returns_sampled = np.zeros((N, eval_runs))
        if name=='':
            from datetime import datetime
            current_time = datetime.now()
            name = current_time.strftime("%d_%m_%Y_%H_%M_%S")
        for rollouts in range(N):
            print("**** ITERATION no", rollouts, " ****")
            pilco.optimize_models()
            pilco.optimize_policy(maxiter=maxiter, restarts=restarts)

            X_new, Y_new, _, _ = rollout(env, pilco, timesteps=sim_timesteps, SUBS=SUBS, verbose=True)

            cur_rew = 0
            X = np.vstack((X, X_new[:plan_timesteps,:])); Y = np.vstack((Y, Y_new[:plan_timesteps,:]))
            pilco.mgpr.set_XY(X, Y)
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
                    if rollouts==0 and k==0:
                        X_eval = X_eval_.copy()
                    else:
                        X_eval = np.vstack((X_eval, X_eval_))
                np.savetxt("res/X_" + name + ".csv", X, delimiter=',')
                np.savetxt("res/X_eval_" + name + ".csv", X_eval, delimiter=',')
                np.savetxt("res/evaluation_returns_sampled_"  + name + ".csv", evaluation_returns_sampled, delimiter=',')
                np.savetxt("res/evaluation_returns_full_" + name + ".csv", evaluation_returns_full, delimiter=',')

if __name__=='__main__':
    env = gym.make('InvertedPendulum-v2')
    for i in range(10):
        name = 'inv_pend_seed' + str(i) + '_'
        pilco_run(env, 4, 2, logging=True, eval_runs=5, eval_max_timesteps=100, name=name, seed=i)

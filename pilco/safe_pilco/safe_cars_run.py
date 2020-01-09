import numpy as np
import tensorflow as tf
import gym
from gpflow import autoflow
from gpflow import settings
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward, LinearReward
from linear_cars_env import LinearCars
from rewards_safe import RiskOfCollision, ObjectiveFunction, reward_wrapper
from pilco.utils import predict_trajectory_wrapper
from safe_pilco import SafePILCO
from pilco.utils import rollout
from pilco.utils import policy
from tensorflow import logging

int_type = settings.dtypes.int_type
float_type = settings.dtypes.float_type


class Normalised_Env():
    def __init__(self, m, std):
        self.env = LinearCars()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.m = m
        self.std = std

    def state_trans(self, x):
        return np.divide(x-self.m, self.std)

    def step(self, action):
        ob, r, done, _ = self.env.step(action)
        return self.state_trans(ob), r, done, {}

    def reset(self):
        ob =  self.env.reset()
        return self.state_trans(ob)

    def render(self):
        self.env.render()


def safe_cars(name='', seed=0, logging=False):
    T = 25
    th = 0.05
    np.random.seed(seed)
    name = name
    J = 5
    N = 8
    eval_runs = 5
    with tf.Session(graph=tf.Graph()) as sess:
        env = LinearCars()
        # Initial random rollouts to generate a dataset
        X1, Y1, _, _ = rollout(env, pilco=None, timesteps=T, verbose=True, random=True, render=False)
        for i in range(1,5):
            X1_, Y1_, _, _ = rollout(env, pilco=None, timesteps=T, verbose=True, random=True, render=False)
            X1 = np.vstack((X1, X1_))
            Y1 = np.vstack((Y1, Y1_))

        env = Normalised_Env(np.mean(X1[:,:4],0), np.std(X1[:,:4], 0))
        X, Y, _, _ = rollout(env, pilco=None, timesteps=T, verbose=True, random=True, render=False)
        for i in range(1,J):
            X_, Y_, _, _ = rollout(env, pilco=None, timesteps=T, verbose=True, random=True, render=False)
            X = np.vstack((X, X_))
            Y = np.vstack((Y, Y_))

        state_dim = Y.shape[1]
        control_dim = X.shape[1] - state_dim

        m_init = np.transpose(X[0,:-1,None])
        S_init = 0.1 * np.eye(state_dim)

        controller = RbfController(state_dim=state_dim, control_dim=control_dim,
                                   num_basis_functions=40, max_action=0.2)

        #w1 = np.diag([1.5, 0.001, 0.001, 0.001])
        #t1 = np.divide(np.array([3.0, 1.0, 3.0, 1.0]) - env.m, env.std)
        #R1 = ExponentialReward(state_dim=state_dim, t=t1, W=w1)
        # R1 = LinearReward(state_dim=state_dim, W=np.array([0.1, 0.0, 0.0, 0.0]))
        R1 = LinearReward(state_dim=state_dim, W=np.array([1.0 * env.std[0], 0., 0., 0,]))

        bound_x1 = 1 / env.std[0]
        bound_x2 = 1 / env.std[2]
        B = RiskOfCollision(2, [-bound_x1-env.m[0]/env.std[0], -bound_x2 - env.m[2]/env.std[2]],
                               [bound_x1 - env.m[0]/env.std[0], bound_x2 - env.m[2]/env.std[2]])

        pilco = SafePILCO(X, Y, controller=controller, mu=-300.0, reward_add=R1, reward_mult=B, horizon=T, m_init=m_init, S_init=S_init)

        # define tolerance
        new_data = True
        # init = tf.global_variables_initializer()
        evaluation_returns_full = np.zeros((N, eval_runs))
        evaluation_returns_sampled = np.zeros((N, eval_runs))
        X_eval = []
        for rollouts in range(N):
            print("***ITERATION**** ", rollouts)
            if new_data:
                pilco.optimize_models(maxiter=100, restarts=2)
                new_data = False
            pilco.optimize_policy(maxiter=20, restarts=5)
            # check safety
            m_p = np.zeros((T, state_dim))
            S_p = np.zeros((T, state_dim, state_dim))
            predicted_risks = np.zeros(T)
            predicted_rewards = np.zeros(T)

            for h in range(T):
                m_h, S_h, _ = predict_trajectory_wrapper(pilco, m_init, S_init, h)
                m_p[h,:], S_p[h,:,:] = m_h[:], S_h[:,:]
                predicted_risks[h], _ = reward_wrapper(B, m_h, S_h)
                predicted_rewards[h], _ = reward_wrapper(R1, m_h, S_h)
            overall_risk = 1 - np.prod(1.0-predicted_risks)

            print("Predicted episode's return: ", sum(predicted_rewards))
            print("Overall risk ", overall_risk)
            print("Mu is ", pilco.mu.mu.value)
            print("bound1 ", bound_x1, " bound1 ", bound_x2)
            print("No of ops:", len(tf.get_default_graph().get_operations()))

            if overall_risk < th:
                X_new, Y_new, _, _ = rollout(env, pilco=pilco, timesteps=T, verbose=True, render=False)
                new_data = True
                X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
                pilco.mgpr.set_XY(X, Y)
                if overall_risk < (th/4):
                    pilco.mu.mu.assign(0.75 * pilco.mu.mu.value)
                if logging:
                    for k in range(0, eval_runs):
                        [X_eval_, _,
                        evaluation_returns_sampled[rollouts, k],
                        evaluation_returns_full[rollouts, k]] = rollout(env, pilco,
                                                                       timesteps=T,
                                                                       verbose=False, SUBS=1,
                                                                       render=False)
                        if len(X_eval)==0:
                            X_eval = X_eval_.copy()
                        else:
                            X_eval = np.vstack((X_eval, X_eval_))
                    np.savetxt("res/X_" + name + ".csv", X, delimiter=',')
                    np.savetxt("res/X_eval_" + name + ".csv", X_eval, delimiter=',')
                    np.savetxt("res/evaluation_returns_sampled_"  + name + ".csv", evaluation_returns_sampled, delimiter=',')
                    np.savetxt("res/evaluation_returns_full_" + name + ".csv", evaluation_returns_full, delimiter=',')
            else:
                X_new, Y_new,_,_ = rollout(env, pilco=pilco, timesteps=T, verbose=True, render=False)
                print(m_p[:,0] - X_new[:,0])
                print(m_p[:,2] - X_new[:,2])
                print("*********CHANGING***********")
                pilco.anchor(sess)
                _, _, r = predict_trajectory_wrapper(pilco, m_init, S_init, T)
                print(r)
                # to verify this actually changes, run the reward wrapper before and after on the same trajectory
                pilco.mu.mu.assign(1.5 * pilco.mu.mu.value, session=sess)
                _, _, r = predict_trajectory_wrapper(pilco, m_init, S_init, T)
                print(r)

if __name__=='__main__':
    safe_cars()

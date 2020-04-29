import numpy as np
import tensorflow as tf
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward, LinearReward

from linear_cars_env import LinearCars

from rewards_safe import RiskOfCollision, ObjectiveFunction
from safe_pilco import SafePILCO
from pilco.utils import rollout, policy

from gpflow import config
from gpflow import set_trainable
int_type = config.default_int()
float_type = config.default_float()


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


def safe_cars(seed=0):
    T = 25
    th = 0.10
    np.random.seed(seed)
    name = name
    J = 5
    N = 5
    eval_runs = 5
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

    pilco = SafePILCO((X, Y), controller=controller, mu=-300.0, reward_add=R1, reward_mult=B, horizon=T, m_init=m_init, S_init=S_init)

    for model in pilco.mgpr.models:
        model.likelihood.variance.assign(0.001)
        set_trainable(model.likelihood.variance, False)

    # define tolerance
    new_data = True
    # init = tf.global_variables_initializer()
    evaluation_returns_full = np.zeros((N, eval_runs))
    evaluation_returns_sampled = np.zeros((N, eval_runs))
    X_eval = []
    for rollouts in range(N):
        print("***ITERATION**** ", rollouts)
        if new_data:
            pilco.optimize_models(maxiter=100)
            new_data = False
        pilco.optimize_policy(maxiter=20, restarts=2)
        # check safety
        m_p = np.zeros((T, state_dim))
        S_p = np.zeros((T, state_dim, state_dim))
        predicted_risks = np.zeros(T)
        predicted_rewards = np.zeros(T)

        for h in range(T):
            m_h, S_h, _ = pilco.predict(m_init, S_init, h)
            m_p[h,:], S_p[h,:,:] = m_h[:], S_h[:,:]
            predicted_risks[h], _ = B.compute_reward(m_h, S_h)
            predicted_rewards[h], _ = R1.compute_reward(m_h, S_h)
        overall_risk = 1 - np.prod(1.0-predicted_risks)

        print("Predicted episode's return: ", sum(predicted_rewards))
        print("Overall risk ", overall_risk)
        print("Mu is ", pilco.mu.numpy())
        print("bound1 ", bound_x1, " bound1 ", bound_x2)

        if overall_risk < th:
            X_new, Y_new, _, _ = rollout(env, pilco=pilco, timesteps=T, verbose=True, render=False)
            new_data = True
            X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
            pilco.mgpr.set_data((X, Y))
            if overall_risk < (th/4):
                pilco.mu.assign(0.75 * pilco.mu.numpy())

        else:
            X_new, Y_new,_,_ = rollout(env, pilco=pilco, timesteps=T, verbose=True, render=False)
            print(m_p[:,0] - X_new[:,0])
            print(m_p[:,2] - X_new[:,2])
            print("*********CHANGING***********")
            _, _, r = pilco.predict(m_init, S_init, T)
            print(r)
            # to verify this actually changes, run the reward wrapper before and after on the same trajectory
            pilco.mu.assign(1.5 * pilco.mu.numpy())
            _, _, r = pilco.predict(m_init, S_init, T)
            print(r)

if __name__=='__main__':
    safe_cars()

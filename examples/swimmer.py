import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward, LinearReward, CombinedRewards
import tensorflow as tf
from tensorflow import logging
from pilco.utils import rollout, policy
np.random.seed(0)

# Uses a wrapper for the Swimmer
# First one to use a combined reward function, that includes penalties
# Uses the video capture function

class SwimmerWrapper():
    def __init__(self):
        self.env = gym.make('Swimmer-v2').env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def state_trans(self, s):
        return np.hstack([[self.x],s])

    def step(self, action):
        ob, r, done, _ = self.env.step(action)
        self.x += r / 10.0
        return self.state_trans(ob), r, done, {}

    def reset(self):
        ob =  self.env.reset()
        self.x = 0.0
        return self.state_trans(ob)

    def render(self):
        self.env.render()


with tf.Session() as sess:
    env = SwimmerWrapper()
    state_dim = 9
    control_dim = 2
    SUBS = 2
    maxiter = 40
    max_action = 1.0
    m_init = np.reshape(np.zeros(state_dim), (1,state_dim))  # initial state mean
    S_init = 0.05 * np.eye(state_dim)
    J = 15
    N = 15
    T = 20
    bf = 20
    T_sim=100

    # Reward function that dicourages the joints from hitting their max angles
    weights_l = np.zeros(state_dim)
    weights_l[0] = 0.1
    max_ang = 100 / 180 * np.pi
    t1 = np.zeros(state_dim)
    t1[2] = max_ang
    w1 = 1e-6 * np.eye(state_dim)
    w1[2,2] = 10
    t2 = np.zeros(state_dim)
    t2[3] = max_ang
    w2 = 1e-6 * np.eye(state_dim)
    w2[3,3] = 10
    t3 = np.zeros(state_dim); t3[2] = -max_ang
    t4 = np.zeros(state_dim); t4[3] = -max_ang
    R2 = LinearReward(state_dim, weights_l)
    R3 = ExponentialReward(state_dim, W=w1, t=t1)
    R4 = ExponentialReward(state_dim, W=w2, t=t2)
    R5 = ExponentialReward(state_dim, W=w1, t=t3)
    R6 = ExponentialReward(state_dim, W=w2, t=t4)
    R = CombinedRewards(state_dim, [R2, R3, R4, R5, R6], coefs=[1.0, -1.0, -1.0, -1.0, -1.0])

    # Initial random rollouts to generate a dataset
    X,Y = rollout(env, None, timesteps=T, random=True, SUBS=SUBS)
    for i in range(1,J):
        X_, Y_ = rollout(env, None, timesteps=T, random=True, SUBS=SUBS, verbose=True)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))

    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)

    pilco = PILCO(X, Y, controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)
    for model in pilco.mgpr.models:
        model.likelihood.variance = 0.001
        model.likelihood.variance.trainable = False

    for rollouts in range(N):
        print("**** ITERATION no", rollouts, " ****")
        pilco.optimize_models(maxiter=maxiter)
        pilco.optimize_policy(maxiter=maxiter, restarts=2)

        X_new, Y_new = rollout(env, pilco, timesteps=T_sim, verbose=True, SUBS=SUBS)

        # Update dataset
        X = np.vstack((X, X_new[:T,:])); Y = np.vstack((Y, Y_new[:T,:]))
        pilco.mgpr.set_XY(X, Y)

    # Saving a video of a run
    # env2 = SwimmerWrapper(monitor=True)
    # rollout(env2, pilco, policy=policy, timesteps=T+50, verbose=True, SUBS=SUBS)
    # env2.env.close()

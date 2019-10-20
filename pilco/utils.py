import numpy as np
from gpflow import autoflow
from gpflow import settings
import pilco
import gym
float_type = settings.dtypes.float_type


def rollout(env, pilco, timesteps, verbose=True, random=False, SUBS=1, render=True):
    X = []; Y = []
    x = env.reset()
    for timestep in range(timesteps):
        if render: env.render()
        u = policy(env, pilco, x, random)
        for i in range(SUBS):
            x_new, _, done, _ = env.step(u)
            if done: break
            if render: env.render()
        if verbose:
            print("Action: ", u)
            print("State : ", x_new)
        X.append(np.hstack((x, u)))
        Y.append(x_new - x)
        x = x_new
        if done: break
    return np.stack(X), np.stack(Y)


def policy(env, pilco, x, random):
    if random:
        return env.action_space.sample()
    else:
        return pilco.compute_action(x[None, :])[0, :]


@autoflow((float_type,[None, None]), (float_type,[None, None]))
def predict_one_step_wrapper(mgpr, m, s):
    return mgpr.predict_on_noisy_inputs(m, s)


@autoflow((float_type,[None, None]), (float_type,[None, None]), (np.int32, []))
def predict_trajectory_wrapper(pilco, m, s, horizon):
    return pilco.predict(m, s, horizon)


@autoflow((float_type,[None, None]), (float_type,[None, None]))
def compute_action_wrapper(pilco, m, s):
    return pilco.controller.compute_action(m, s)


@autoflow((float_type, [None, None]), (float_type, [None, None]))
def reward_wrapper(reward, m, s):
    return reward.compute_reward(m, s)

class Normalised_Env():
    def __init__(self, env_id, m, std):
        self.env = gym.make(env_id).env
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

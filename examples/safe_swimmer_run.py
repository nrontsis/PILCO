import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward, LinearReward, CombinedRewards
from safe_pilco_extension.rewards_safe import SingleConstraint
import tensorflow as tf
from utils import rollout, policy
from gpflow import set_trainable

# Uses a wrapper for the Swimmer
# First one to use a combined reward function, that includes penalties
# Uses the video capture function

# np.random.seed(int(sys.argv[2]))
# name = "safe_swimmer_final" + sys.argv[2]

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

def safe_swimmer_run(seed=0, logging=False):
    env = SwimmerWrapper()
    state_dim = 9
    control_dim = 2
    SUBS = 2
    maxiter = 60
    max_action = 1.0
    m_init = np.reshape(np.zeros(state_dim), (1,state_dim))  # initial state mean
    S_init = 0.05 * np.eye(state_dim)
    J = 1
    N = 12
    T = 25
    bf = 30
    T_sim=100

    # Reward function that dicourages the joints from hitting their max angles
    weights_l = np.zeros(state_dim)
    weights_l[0] = 0.5
    max_ang = (100 / 180 * np.pi) * 0.95
    R1 = LinearReward(state_dim, weights_l)

    C1 = SingleConstraint(1, low=-max_ang, high=max_ang, inside=False)
    C2 = SingleConstraint(2, low=-max_ang, high=max_ang, inside=False)
    C3 = SingleConstraint(3, low=-max_ang, high=max_ang, inside=False)
    R = CombinedRewards(state_dim, [R1, C1, C2, C3], coefs=[1.0, -10.0, -10.0, -10.0])

    th=0.2
    # Initial random rollouts to generate a dataset
    X,Y, _, _ = rollout(env, None, timesteps=T, random=True, SUBS=SUBS, verbose=True)
    for i in range(1,J):
        X_, Y_ , _, _= rollout(env, None, timesteps=T, random=True, SUBS=SUBS, verbose=True)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))

    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)

    pilco = PILCO((X, Y), controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)
    for model in pilco.mgpr.models:
        model.likelihood.variance.assign(0.001)
        set_trainable(model.likelihood.variance, False)

    new_data = True
    eval_runs = T_sim
    evaluation_returns_full = np.zeros((N, eval_runs))
    evaluation_returns_sampled = np.zeros((N, eval_runs))
    X_eval = []
    for rollouts in range(N):
        print("**** ITERATION no", rollouts, " ****")
        if new_data: pilco.optimize_models(maxiter=100); new_data = False
        pilco.optimize_policy(maxiter=1, restarts=2)

        m_p = np.zeros((T, state_dim))
        S_p = np.zeros((T, state_dim, state_dim))
        predicted_risk1 = np.zeros(T)
        predicted_risk2 = np.zeros(T)
        predicted_risk3 = np.zeros(T)
        for h in range(T):
            m_h, S_h, _ = pilco.predict(m_init, S_init, h)
            m_p[h,:], S_p[h,:,:] = m_h[:], S_h[:,:]
            predicted_risk1[h], _ = C1.compute_reward(m_h, S_h)
            predicted_risk2[h], _ = C2.compute_reward(m_h, S_h)
            predicted_risk3[h], _ = C3.compute_reward(m_h, S_h)
        estimate_risk1 = 1 - np.prod(1.0-predicted_risk1)
        estimate_risk2 = 1 - np.prod(1.0-predicted_risk2)
        estimate_risk3 = 1 - np.prod(1.0-predicted_risk3)
        overall_risk = 1 - (1 - estimate_risk1) * (1 - estimate_risk2) * (1 - estimate_risk3)
        if overall_risk < th:
            X_new, Y_new, _, _ = rollout(env, pilco, timesteps=T_sim, verbose=True, SUBS=SUBS)
            new_data = True
            # Update dataset
            X = np.vstack((X, X_new[:T,:])); Y = np.vstack((Y, Y_new[:T,:]))
            pilco.mgpr.set_data((X, Y))
            if estimate_risk1 < th/10:
                R.coefs.assign(R.coefs.value() * [1.0, 0.75, 1.0, 1.0])
            if estimate_risk2 < th/10:
                R.coefs.assign(R.coefs.value() * [1.0, 1.0, 0.75, 1.0])
            if estimate_risk3 < th/10:
                R.coefs.assign(R.coefs.value() * [1.0, 1.0, 1.0, 0.75])
        else:
            print("*********CHANGING***********")
            if estimate_risk1 > th/3:
                R.coefs.assign(R.coefs.value() * [1.0, 1.5, 1.0, 1.0])
            if estimate_risk2 > th/3:
                R.coefs.assign(R.coefs.value() * [1.0, 1.0, 1.5, 1.0])
            if estimate_risk3 > th/3:
                R.coefs.assign(R.coefs.value() * [1.0, 1.0, 1.0, 1.5])
            _, _, r = pilco.predict(m_init, S_init, T)



if __name__=='__main__':
    safe_swimmer_run()

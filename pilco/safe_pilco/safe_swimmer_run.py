import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward, LinearReward, CombinedRewards
from rewards_safe import SingleConstraint
import tensorflow as tf
from tensorflow import logging
from pilco.utils import rollout, policy, predict_trajectory_wrapper, reward_wrapper

np.random.seed(0)
name = "safe_swimmer_final"
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

# config = tf.ConfigProto()
# gpu_id = sys.argv[1]
# config.gpu_options.visible_device_list = gpu_id
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
with tf.Session() as sess:
    env = SwimmerWrapper()
    state_dim = 9
    control_dim = 2
    SUBS = 2
    maxiter = 60
    max_action = 1.0
    m_init = np.reshape(np.zeros(state_dim), (1,state_dim))  # initial state mean
    S_init = 0.05 * np.eye(state_dim)
    J = 10
    N = 12
    T = 25
    bf = 30
    T_sim=100

    # Reward function that dicourages the joints from hitting their max angles
    weights_l = np.zeros(state_dim)
    weights_l[0] = 0.5
    max_ang = (100 / 180 * np.pi) * 0.95
    # t1 = np.zeros(state_dim)
    # t1[2] = max_ang
    # w1 = 1e-6 * np.eye(state_dim)
    # w1[2,2] = 10
    # t2 = np.zeros(state_dim)
    # t2[3] = max_ang
    # w2 = 1e-6 * np.eye(state_dim)
    # w2[3,3] = 10
    # t3 = np.zeros(state_dim); t3[2] = -max_ang
    # t4 = np.zeros(state_dim); t4[3] = -max_ang
    R1 = LinearReward(state_dim, weights_l)
    # R3 = ExponentialReward(state_dim, W=w1, t=t1)
    # R4 = ExponentialReward(state_dim, W=w2, t=t2)
    # R5 = ExponentialReward(state_dim, W=w1, t=t3)
    # R6 = ExponentialReward(state_dim, W=w2, t=t4)
    #R = CombinedRewards(state_dim, [R1, R3, R4, R5, R6], coefs=[1.0, -1.0, -1.0, -1.0, -1.0])

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

    pilco = PILCO(X, Y, controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)
    #for model in pilco.mgpr.models:
        # model.likelihood.variance = 0.001
        # model.likelihood.variance.trainable = False

    new_data = True
    eval_runs = T_sim
    evaluation_returns_full = np.zeros((N, eval_runs))
    evaluation_returns_sampled = np.zeros((N, eval_runs))
    X_eval = []
    for rollouts in range(N):
        print("**** ITERATION no", rollouts, " ****")
        if new_data: pilco.optimize_models(maxiter=100); new_data = False
        pilco.optimize_policy(maxiter=maxiter, restarts=2)

        m_p = np.zeros((T, state_dim))
        S_p = np.zeros((T, state_dim, state_dim))
        predicted_risk1 = np.zeros(T)
        predicted_risk2 = np.zeros(T)
        predicted_risk3 = np.zeros(T)
        for h in range(T):
            m_h, S_h, _ = predict_trajectory_wrapper(pilco, m_init, S_init, h)
            m_p[h,:], S_p[h,:,:] = m_h[:], S_h[:,:]
            predicted_risk1[h], _ = reward_wrapper(C1, m_h, S_h)
            predicted_risk2[h], _ = reward_wrapper(C2, m_h, S_h)
            predicted_risk3[h], _ = reward_wrapper(C3, m_h, S_h)
        estimate_risk1 = 1 - np.prod(1.0-predicted_risk1)
        estimate_risk2 = 1 - np.prod(1.0-predicted_risk2)
        estimate_risk3 = 1 - np.prod(1.0-predicted_risk3)
        overall_risk = 1 - (1 - estimate_risk1) * (1 - estimate_risk2) * (1 - estimate_risk3)
        # print(predicted_risk1)
        # print(estimate_risk1)
        # print(estimate_risk2)
        # print(estimate_risk3)
        # print("No of ops:", len(tf.get_default_graph().get_operations()))
        #import pdb; pdb.set_trace()
        if overall_risk < th:
            X_new, Y_new, _, _ = rollout(env, pilco, timesteps=T_sim, verbose=True, SUBS=SUBS)
            new_data = True
            # Update dataset
            X = np.vstack((X, X_new[:T,:])); Y = np.vstack((Y, Y_new[:T,:]))
            pilco.mgpr.set_XY(X, Y)
            if estimate_risk1 < th/10:
                R.coefs.assign(R.coefs.value * [1.0, 0.75, 1.0, 1.0])
            if estimate_risk2 < th/10:
                R.coefs.assign(R.coefs.value * [1.0, 1.0, 0.75, 1.0])
            if estimate_risk3 < th/10:
                R.coefs.assign(R.coefs.value * [1.0, 1.0, 1.0, 0.75])
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
            print("*********CHANGING***********")
            # X_2, Y_2, _, _ = rollout(env, pilco, timesteps=T_sim, verbose=True, SUBS=SUBS)
            # print(m_p)
            # print(S_p)
            # _, _, r = predict_trajectory_wrapper(pilco, m_init, S_init, T)
            # print("Before ", r)
            # print(R.coefs.value)
            if estimate_risk1 > th/3:
                R.coefs.assign(R.coefs.value * [1.0, 1.5, 1.0, 1.0])
            if estimate_risk2 > th/3:
                R.coefs.assign(R.coefs.value * [1.0, 1.0, 1.5, 1.0])
            if estimate_risk3 > th/3:
                R.coefs.assign(R.coefs.value * [1.0, 1.0, 1.0, 1.5])
            _, _, r = predict_trajectory_wrapper(pilco, m_init, S_init, T)
            # print("After ", r)
            # print(R.coefs.value)



    # Saving a video of a run
    # env2 = SwimmerWrapper(monitor=True)
    # rollout(env2, pilco, policy=policy, timesteps=T+50, verbose=True, SUBS=SUBS)
    # env2.env.close()

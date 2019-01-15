import numpy as np
import gym
from gpflow import autoflow
from gpflow import settings
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf
from tensorflow import logging
int_type = settings.dtypes.int_type
float_type = settings.dtypes.float_type

@autoflow((float_type, [None, None]), (float_type, [None, None]), (int_type, []))
def pil_predict_wrapper(pilco, m_x, s_n, n):
    return pilco.predict(m_x, s_n, n)


@autoflow((float_type, [None, None]), (float_type, [None, None]))
def reward_wrapper(reward, m, s):
    return reward.compute_reward(m, s)


class DoublePendWrapper():
    def __init__(self):
        self.env = gym.make('InvertedDoublePendulum-v2')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def state_trans(self, s):
        a1 = np.arctan2(s[1], s[3])
        a2 = np.arctan2(s[2], s[4])
        s_new = np.hstack([s[0], a1, a2, s[5:-3]])
        return s_new

    def step(self, action):
        ob, r, done, _ = self.env.step(action)
        if np.abs(ob[0])> 0.98 or np.abs(ob[-3]) > 0.1 or  np.abs(ob[-2]) > 0.1 or np.abs(ob[-1]) > 0.1:
            done = True
        return self.state_trans(ob), r, done, {}

    def reset(self):
        ob =  self.env.reset()
        return self.state_trans(ob)

    def render(self):
        self.env.render()

class SwimmerWrapper():
    def __init__(self):
        self.env = gym.make('Swimmer-v2')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def state_trans(self, s):
        return np.hstack([[self.x],s])

    def step(self, action):
        ob, r, done, _ = self.env.step(action)
        self.x += r
        return self.state_trans(ob), r, done, {}

    def reset(self):
        ob =  self.env.reset()
        self.x = 0.0
        return self.state_trans(ob)

    def render(self):
        self.env.render()


def rollout(env, pilco, policy, timesteps, verbose=False, random=False, SUBS=1):
    X = []; Y = []
    x = env.reset()
    for timestep in range(timesteps):
        if timestep > 0:
            if done: break
        # env.render()
        u = policy(env, pilco, x, random)
        for i in range(SUBS):
            x_new, _, done, _ = env.step(u)
            if done: break
            # env.render()
            #x_new += 0.001 * (np.random.rand()-0.5)
        if verbose:
            print("Action: ", u)
            print("State : ",  x_new)
        # if done: break
        X.append(np.hstack((x, u)))
        Y.append(x_new - x)
        x = x_new
    return np.stack(X), np.stack(Y)

# def random_policy(env, x):
#     return env.action_space.sample()
#
# def pilco_policy(pilco, x):
#     return pilco.compute_action(x[None, :])[0, :]

def policy(env, pilco, x, random):
    if random:
        return env.action_space.sample()
    else:
        return pilco.compute_action(x[None, :])[0, :]


def make_env(env_id, **kwargs):
    seed = kwargs.get('seed', 0)
    np.random.seed(seed)
    linear = False
    restarts = False
    n_ind=None
    if env_id == 'InvertedPendulum-v2':
        env = gym.make(env_id)
        # Default_values
        state_dim = 4
        control_dim = 1
        # SOLVED
        SUBS = 1            # subsampling factor (if 2 every action is repeated twice)
        bf = 5              # number of basis functions for rbf controller
        maxiter = 50        # number of iterations for the controller optimisation
        max_action = 1.0
        target = np.array(np.zeros(state_dim))           # goal state, passed to the reward function
        weights = np.eye(state_dim)            # weights of the reward function
        m_init = np.reshape(np.zeros(state_dim), (1,state_dim))  # initial state mean
        S_init = 0.1 * np.eye(state_dim)           # initial state variance
        T = 60              # horizon length in timesteps
        J = 5               # number of initial rollouts with random actions
        N = 10              # number of iterations
    elif env_id == 'MountainCarContinuous-v0':
        env = gym.make(env_id)
        # NOT SOLVED, stuck in greedy behaviour
        SUBS=5
        bf = 10
        maxiter = 50
        max_action = 1.0
        target = np.array([0.45, 0])
        weights = np.diag([4.0, 0.0001])
        m_init = np.reshape([-0.5, 0], (1,2))
        S_init = np.diag([0.2, 0.001])
        T = 25
        J = 3
        N = 10
    elif env_id == 'Pendulum-v0':
        env = gym.make(env_id)
        # NEEDS a different initialisation than the one in gym (change the reset() method),
        # to (m_init, S_init)
        SUBS=3
        bf = 30
        maxiter=3
        max_action=2.0
        target = np.array([1.0, 0.0, 0.0])
        weights = np.diag([2.0, 2.0, 0.3])
        m_init = np.reshape([-1.0, 0, 0.0], (1,3))
        S_init = np.diag([0.01, 0.05, 0.01])
        T = 40
        J = 4
        N = 8
        restarts = True
    elif env_id == 'CartPole-v0':
        env = gym.make('env_id')
        # Takes discrete actions, crashes.
        SUBS = 2
        bf = 20
        maxiter=50
        max_action=2.0 # actions for these environments are discrete
        target = np.array([0.0, 0.0, 1.0, 0.0])
        weights = np.diag([1.0, 1.0, 1.0, 1.0])
        m_init = np.reshape([0.0, 0.0, 0.0, 0.0], (1,4))
        S_init = np.diag([0.05, 0.05, 0.05, 0.05])
        T = 40
        J = 2
        N = 12
    elif env_id == 'InvertedDoublePendulum-v2':
        env = gym.make(env_id)
        SUBS = 1
        bf = 60
        maxiter=100
        state_dim = 11
        control_dim = 1
        max_action=1.0 # actions for these environments are discrete
        target = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        weights = 1.5 * np.eye(state_dim)
        weights[5,5]= 0.3
        weights[6,6]= 0.3
        weights[7,7]= 0.3
        m_init = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])[None, :]
        S_init = 0.01 * np.eye(state_dim)
        S_init[6,6] = 1 # ???
        S_init[7,7] = 1
        T = 30
        J = 20
        N = 20
        restarts = True
    elif env_id == 'InvertedDoublePendulumWrapped':
        env = DoublePendWrapper()
        SUBS = 1
        bf = 40
        maxiter=80
        state_dim = 6
        control_dim = 1
        max_action=1.0 # actions for these environments are discrete
        target = np.zeros(state_dim)
        weights = 3.0 * np.eye(state_dim)
        weights[0,0] = 0.5
        weights[3,3] = 0.5
        m_init = np.zeros(state_dim)[None, :]
        S_init = 0.01 * np.eye(state_dim)
        T = 40
        J = 1
        N = 12
        T_sim = 130
        restarts=True
    elif env_id == 'SwimmerWrapped':
        env = SwimmerWrapper()
        state_dim = 9
        control_dim = 2
        SUBS = 2
        maxiter = 40
        max_action = 1.0
        m_init = np.reshape(np.zeros(state_dim), (1,state_dim))  # initial state mean
        S_init = 0.5 * np.eye(state_dim)
        target = np.array(np.zeros(state_dim))           # goal state, passed to the reward function
        target[0] = 20
        weights = 1e-6 * np.eye(state_dim)
        weights[0,0] = 0.02
        J = 4
        N = 8
        T = 30
        bf = 30
        linear = False
        restarts = True
    elif env_id == 'Quadcopter':
        from quadcopter_env import Quadcopter
        env = Quadcopter()
        SUBS = 1
        bf = 100
        maxiter = 50
        state_dim = 15
        control_dim = 4
        max_action = 1.0
        weights = 0.00001 * np.eye(state_dim)
        weights[0,0] = 1; weights[1,1] = 1; weights[2,2] = 1
        m_init = np.array([ 0., 0., 10., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.])[None, :]
        target = np.zeros(state_dim)
        target[:] = m_init[:]
        target[0] += 0.5; target[1] += 0.0; target[2] += 1.0
        S_init = 0.01 * np.eye(state_dim)
        T = 30
        J = 5
        N = 10
        restarts = True

    else:
        print("Didn't recognise environment id")

    # read values, using the default value for the specific env when no value is provided
    parameters = {}
    parameters['SUBS'] = kwargs.get('SUBS', SUBS)
    parameters['bf'] = kwargs.get('bf', bf)
    parameters['maxiter'] = kwargs.get('maxiter', maxiter)
    parameters['max_action'] = kwargs.get('max_action', max_action)
    parameters['target'] = kwargs.get('target', target)
    parameters['weights'] = kwargs.get('weights', weights)
    parameters['m_init'] = kwargs.get('m_init', m_init)
    parameters['S_init'] = kwargs.get('S_init', S_init)
    parameters['T'] = kwargs.get('T', T)
    parameters['J'] = kwargs.get('J', J)
    parameters['N'] = kwargs.get('N', N)
    parameters['restarts'] = kwargs.get('restarts', restarts)
    parameters['n_ind'] = kwargs.get('n_ind', n_ind)
    parameters['linear'] = kwargs.get('linear', linear)
    try:
        parameters['T_sim'] = T_sim
    except:
        parameters['T_sim'] = T
    parameters['seed'] = seed
    return env, parameters

def run(env_id, **kwargs ):
    config = tf.ConfigProto()
    gpu_id = kwargs.get('gpu_id', "1")
    config.gpu_options.visible_device_list = gpu_id
    config.gpu_options.per_process_gpu_memory_fraction = 0.80
    sess = tf.Session(graph=tf.Graph(), config=config)
    with tf.Session(config=config) as sess:
        # Make env
        env, parameters = make_env(env_id, **kwargs)
        SUBS, bf, maxiter, max_action, target, weights, m_init, S_init, T, J, N, restarts, linear, T_sim, n_ind = \
        parameters['SUBS'], parameters['bf'],                  \
        parameters['maxiter'], parameters['max_action'],       \
        parameters['target'], parameters['weights'],           \
        parameters['m_init'], parameters['S_init'],            \
        parameters['T'], parameters['J'],                      \
        parameters['N'], parameters['restarts'],               \
        parameters['linear'], parameters['T_sim'], parameters['n_ind']

        # Initial random rollouts to generate a dataset
        X,Y = rollout(env, None, policy=policy, timesteps=T, random=True, SUBS=SUBS)
        for i in range(1,J):
            X_, Y_ = rollout(env, None, policy=policy, timesteps=T, random=True, SUBS=SUBS, verbose=True)
            X = np.vstack((X, X_))
            Y = np.vstack((Y, Y_))

        state_dim = Y.shape[1]
        control_dim = X.shape[1] - state_dim

        if linear:
            controller = LinearController(state_dim=state_dim, control_dim=control_dim, max_action=max_action)
        else:
            controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)

        R = ExponentialReward(state_dim=state_dim, t=target, W=weights)

        pilco = PILCO(X, Y, controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init, num_induced_points=n_ind)

        # Example of fixing a parameter, optional, for a linear controller only
        #pilco.controller.b = np.array([[0.0]])
        #pilco.controller.b.trainable = False

        # Some initialisations
        x_pred = np.zeros((T, state_dim))
        s_pred = np.zeros((T, state_dim, state_dim))
        rr = np.zeros(T)
        lens = []
        for rollouts in range(N):
            print("**** ITERATION no", rollouts, " ****")
            pilco.optimize(maxiter=maxiter)
            if restarts:
                pilco.mgpr.try_restart(sess, restarts=1)
                pilco.restart_controller(sess, restarts=1)
            print("No of ops:", len(tf.get_default_graph().get_operations()))

            # Predict the trajectory, to check model's accuracy
            # for i in range(1,T):
            #     x_pred[i,:], s_pred[i,:,:], rr[i] = pil_predict_wrapper(pilco, m_init, S_init, i)

            X_new, Y_new = rollout(env, pilco, policy=policy, timesteps=T_sim, verbose=True, SUBS=SUBS)
            cur_rew = 0
            for t in range(0,len(X_new)):
                cur_rew += reward_wrapper(R, X_new[t, 0:state_dim, None].transpose(), 0.0001 * np.eye(state_dim))[0]
            print('On this episode reward was ', cur_rew)
            # Update dataset
            X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
            pilco.mgpr.set_XY(X, Y)
            lens.append(len(X_new))
            if len(X_new) > 120: break

        parameters['lens'] = lens
        return X, parameters

if __name__ == '__main__':
    np.random.seed(0)
    run('SwimmerWrapped')
    #run('Quadcopter')

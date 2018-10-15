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

np.random.seed(0)

@autoflow((float_type, [None, None]), (float_type, [None, None]), (int_type, []))
def pil_predict_wrapper(pilco, m_x, s_n, n):
    return pilco.predict(m_x, s_n, n)

def rollout(policy, timesteps, verbose=False):
    X = []; Y = []
    x = env.reset()
    for timestep in range(timesteps):
        env.render()
        u = policy(x)
        for i in range(SUBS):
            x_new, _, done, _ = env.step(u)
            env.render()
            # x_new += [0.001*np.random.randn(), 0.001*np.random.randn(), 0.001*np.random.randn()]
        if verbose:
            print("Action: ", u)
            print("State : ",  x_new)
        if done: break
        X.append(np.hstack((x, u)))
        Y.append(x_new - x)
        x = x_new
    return np.stack(X), np.stack(Y)

def random_policy(x):
    return env.action_space.sample()

def pilco_policy(x):
    return pilco.compute_action(x[None, :])[0, :]


# env = gym.make('InvertedPendulum-v2')
# state_dim = 4
# control_dim = 1
# # SOLVED
# SUBS = 1            # subsampling factor (if 2 every action is repeated twice)
# bf = 5              # number of basis functions for rbf controller
# maxiter = 50        # number of iterations for the controller optimisation
# max_action = 1.0
# target = np.array(np.zeros(state_dim))           # goal state, passed to the reward function
# weights = np.diag(np.ones(state_dim))            # weights of the reward function
# m_init = np.reshape(np.zeros(state_dim), (1,state_dim))  # initial state mean
# S_init = 0.1 * np.diag(np.ones(state_dim))            # initial state variance
# T = 60              # horizon length in timesteps
# J = 5               # number of initial rollouts with random actions
# N = 10              # number of iterations

# env = gym.make('MountainCarContinuous-v0')
# # NOT SOLVED, stuck in greedy behaviour
# SUBS=5
# bf = 10
# maxiter = 50
# max_action = 1.0
# target = np.array([0.45, 0])
# weights = np.diag([4.0, 0.0001])
# m_init = np.reshape([-0.5, 0], (1,2))
# S_init = np.diag([0.2, 0.001])
# T = 25
# J = 3
# N = 10

with tf.Session(graph=tf.Graph()):
    env = gym.make('Pendulum-v0')
    # NEEDS a different initialisation than the one in gym (change the reset() method),
    # to (m_init, S_init)
    SUBS=3
    bf = 20
    maxiter=50
    max_action=2.0
    target = np.array([1.0, 0.0, 0.0])
    weights = np.diag([2.0, 2.0, 0.3])
    m_init = np.reshape([-1.0, 0, 0.0], (1,3))
    S_init = np.diag([0.01, 0.05, 0.01])
    T = 40
    J = 4
    N = 12

    # env = gym.make('Reacher-v2')
    # SUBS=3
    # bf = 20
    # maxiter=50
    # max_action=2.0
    # target = np.array([1.0, 0.0, 0.0])
    # weights = np.diag([2.0, 2.0, 0.3])
    # m_init = np.reshape([-1.0, 0, 0.0], (1,3))
    # S_init = np.diag([0.01, 0.05, 0.01])
    # T = 30

    # Initial random rollouts to generate a dataset
    X,Y = rollout(policy=random_policy, timesteps=T)
    for i in range(1,J):
        X_, Y_ = rollout(policy=random_policy, timesteps=T)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))


    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
    # controller = LinearController(state_dim=state_dim, control_dim=control_dim, max_action=max_action)

    R = ExponentialReward(state_dim=state_dim, t=target, W=weights)

    pilco = PILCO(X, Y, controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)

    # Example of fixing a parameter, optional, for a linear controller only
    #pilco.controller.b = np.array([[0.0]])
    #pilco.controller.b.trainable = False

    # Some initialisations
    x_pred = np.zeros((T, state_dim))
    s_pred = np.zeros((T, state_dim, state_dim))
    rr = np.zeros(T)

    for rollouts in range(N):
        print("**** ITERATION no", rollouts, " ****")
        pilco.optimize(maxiter=1)
        print("No of ops:", len(tf.get_default_graph().get_operations()))
        import pdb; pdb.set_trace()
        pilco.optimize(maxiter=1)
        print("No of ops:", len(tf.get_default_graph().get_operations()))
        import pdb; pdb.set_trace()

        # Predict the trajectory, to check model's accuracy
        # for i in range(0,T):
        #    x_pred[i,:], s_pred[i,:,:], rr[i] = pil_predict_wrapper(pilco, m_init, S_init, i)
        X_new, Y_new = rollout(policy=pilco_policy, timesteps=T, verbose=True)

        # import pdb; pdb.set_trace()

        # Update dataset
        X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
        pilco.mgpr.set_XY(X, Y)


        # RESTARTS model and controller, to avoid local minima
        # if rollouts % 3 == 0:
        #     c2 = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
        #     p2 = PILCO(X, Y, controller=c2, horizon=T, reward=R)
        #     p2.optimize(maxiter=maxiter)
            #_ ,_ , rr2 = pil_predict_wrapper(p2, T)
            # if the predicted reward is higher,replaces the previous model/controller
            # if rr2 > rr[T-1]:
            #     pilco = p2

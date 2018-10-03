import numpy as np
import gym
from gpflow import autoflow
from gpflow import settings
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
int_type = settings.dtypes.int_type
float_type = settings.dtypes.float_type

np.random.seed(0)


@autoflow((float_type, [None, None]), (float_type, [None, None]), (int_type, []))
def pil_predict_wrapper(pilco, m_x, s_n, n):
    return pilco.predict(m_x, s_n, n)

@autoflow((float_type,[None, None]), (float_type,[None, None]))
def predict_wrapper(mgpr, m, s):
    return mgpr.predict_on_noisy_inputs(m, s)

# env = gym.make('InvertedPendulum-v2')

# env = gym.make('MountainCarContinuous-v0')
# SUBS=3
# bf = 10
# max_action=1.0
# target = np.array([0.45, 0])
# weights = np.diag([2.0, 0.0001])
# m_init = np.reshape([-0.5, 0], (1,2))
# S_init = np.diag([0.1, 0.001])


env = gym.make('Pendulum-v0')
SUBS=3
bf = 20
maxiter=50
max_action=2.0
target = np.array([1.0, 0.0, 0.0])
weights = np.diag([2.0, 2.0, 0.3])
m_init = np.reshape([-1.0, 0, 4], (1,3))
S_init = np.diag([0.005, 0.03, 0.01])
T = 30

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

# Initial random rollouts to generate a dataset
X,Y = rollout(policy=random_policy, timesteps=T)
for i in range(1,4):
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

x_pred = np.zeros((T, 3))
s_pred = np.zeros((T, 3, 3))
rr = np.zeros(T)
for i in range(0,T):
    x_pred[i,:], s_pred[i,:,:], rr[i] = pil_predict_wrapper(pilco, m_init, S_init, i)

test = predict_wrapper(pilco.mgpr, np.reshape([-1.0, 0.0, 4.0, 0.0], (1,4)), np.diag([0.005, 0.03, 0.01, 0.001]))

for rollouts in range(8):
    pilco.optimize(maxiter=maxiter)
    for i in range(1,T):
        x_pred[i,:], s_pred[i,:,:], rr[i] = pil_predict_wrapper(pilco, m_init, S_init, i)
    X_new, Y_new = rollout(policy=pilco_policy, timesteps=T, verbose=True)
    # import pdb; pdb.set_trace()
    # Update dataset
    X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_XY(X, Y)
    #if rollouts<5:
    #    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
        # controller = LinearController(state_dim=state_dim, control_dim=control_dim)
    #    pilco = PILCO(X, Y, controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)

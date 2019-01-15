import numpy as np
from gpflow import autoflow
from gpflow import settings
from pilco.models.pilco import PILCO
float_type = settings.dtypes.float_type

@autoflow((float_type,[None, None]), (float_type,[None, None]), (np.int32, []))
def predict_wrapper(pilco, m, s, horizon):
    return pilco.predict(m, s, horizon)

@autoflow((float_type,[None, None]), (float_type,[None, None]))
def predict_gpr_wrapper(mgpr, m, s):
    return mgpr.predict_on_noisy_inputs(m, s)

@autoflow((float_type,[None, None]), (float_type,[None, None]))
def compute_action_wrapper(controller, m, s, squash=False):
    return controller.compute_action(m, s, squash)

@autoflow((float_type, [None, None]), (float_type, [None, None]))
def reward_wrapper(reward, m, s):
    return reward.compute_reward(m, s)

@autoflow()
def get_induced_points(smgpr):
    return smgpr.Z

def save_pilco(path, X, Y, pilco, sparse=False):
    np.savetxt(path + 'X.csv', X, delimiter=',')
    np.savetxt(path + 'Y.csv', Y, delimiter=',')
    if sparse:
        with open(path+ 'n_ind.txt', 'w') as f:
            f.write('%d' % pilco.mgpr.num_induced_points)
            f.close()
    np.save(path + 'pilco_values.npy', pilco.read_values())
    for i,m in enumerate(pilco.mgpr.models):
        np.save(path + "model_" + str(i) + ".npy", m.read_values())

def load_pilco(path, sparse=False):
    X = np.loadtxt(path + 'X.csv', delimiter=',')
    Y = np.loadtxt(path + 'Y.csv', delimiter=',')
    if not sparse:
        pilco = PILCO(X, Y)
    else:
        with open(path+ 'n_ind.txt', 'r') as f:
            n_ind = int(f.readline())
            f.close()
        pilco = PILCO(X, Y, num_induced_points=n_ind)
    params = np.load(path + "pilco_values.npy").item()
    pilco.assign(params)
    for i,m in enumerate(pilco.mgpr.models):
        values = np.load(path + "model_" + str(i) + ".npy").item()
        m.assign(values)
    return pilco

from pilco.models import MGPR
from pilco.models.pilco import PILCO
import numpy as np
import os
from gpflow import autoflow
from gpflow import settings
import oct2py
import logging
octave = oct2py.Oct2Py(logger=oct2py.get_log())
octave.logger = oct2py.get_log('new_log')
octave.logger.setLevel(logging.INFO)
dir_path = os.path.dirname(os.path.realpath("__file__")) + "/tests/Matlab Code"
octave.addpath(dir_path)

float_type = settings.dtypes.float_type

@autoflow((float_type,[None, None]), (float_type,[None, None]), (np.int32, []))
def predict_wrapper(pilco, m, s, horizon):
    return pilco.predict(m, s, horizon)

@autoflow((float_type,[None, None]), (float_type,[None, None]))
def compute_action_wrapper(pilco, m, s):
    return pilco.controller.compute_action(m, s)

def test_cascade():
    np.random.seed(0)
    d = 2  # State dimenstion
    k = 1  # Controller's output dimension
    horizon = 10

    # Training Dataset
    X0 = np.random.rand(100, d + k)
    A = np.random.rand(d + k, d)
    Y0 = np.sin(X0).dot(A) + 1e-3*(np.random.rand(100, d) - 0.5)  #  Just something smooth
    pilco = PILCO(X0, Y0)
    # pilco.optimize()

    # Generate input
    m = np.random.rand(1, d)  # But MATLAB defines it as m'
    s = np.random.rand(d, d)
    s = s.dot(s.T)  # Make s positive semidefinite

    M, S, reward = predict_wrapper(pilco, m, s, horizon)

    # convert data to the struct expected by the MATLAB implementation
    policy = oct2py.io.Struct()
    policy.p = oct2py.io.Struct()
    policy.p.w = pilco.controller.W.value
    policy.p.b = pilco.controller.b.value.T
    policy.maxU = 1e5*np.ones(k)

    # convert data to the struct expected by the MATLAB implementation
    lengthscales = np.stack([model.kern.lengthscales.value for model in pilco.mgpr.models])
    variance = np.stack([model.kern.variance.value for model in pilco.mgpr.models])
    noise = np.stack([model.likelihood.variance.value for model in pilco.mgpr.models])

    hyp = np.log(np.hstack(
        (lengthscales,
         np.sqrt(variance[:, None]),
         np.sqrt(noise[:, None]))
    )).T

    dynmodel = oct2py.io.Struct()
    dynmodel.hyp = hyp
    dynmodel.inputs = X0
    dynmodel.targets = Y0

    plant = oct2py.io.Struct()
    plant.angi = np.zeros(0)
    plant.angi = np.zeros(0)
    plant.poli = np.arange(d) + 1
    plant.dyni = np.arange(d) + 1
    plant.difi = np.arange(d) + 1

    # Call function in octave
    M_mat, S_mat = octave.pred(policy, plant, dynmodel, m.T, s, horizon, nout=2, verbose=True)
    # Extract only last element of the horizon
    M_mat = M_mat[:,-1]
    S_mat = S_mat[:,:,-1]

    np.testing.assert_allclose(M[0], M_mat.T, rtol=1e-4)
    np.testing.assert_allclose(S, S_mat, rtol=1e-4)


if __name__ == '__main__':
    test_cascade()

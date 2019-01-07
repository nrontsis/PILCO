from pilco.models import MGPR
from pilco.models.pilco import PILCO
import numpy as np
import os
from gpflow import autoflow
from gpflow import settings
float_type = settings.dtypes.float_type

from pilco.utils import save_pilco, load_pilco, predict_wrapper

def test_load_save():
    np.random.seed(0)
    d = 2  # State dimenstion
    k = 1  # Controller's output dimension
    horizon = 10
    e = np.array([[10.0]])   # Max control input. Set too low can lead to Cholesky failures.

    # Training Dataset
    X0 = np.random.rand(100, d + k)
    A = np.random.rand(d + k, d)
    Y0 = np.sin(X0).dot(A) + 1e-3*(np.random.rand(100, d) - 0.5)  #  Just something smooth
    pilco = PILCO(X0, Y0)
    pilco.optimize()

    save_pilco("", X0, Y0, pilco)

    pilco2 = load_pilco("")

    m = np.random.rand(1, d)  # But MATLAB defines it as m'
    s = np.random.rand(d, d)
    s = s.dot(s.T)  # Make s positive semidefinite

    M1, S1, r1 = predict_wrapper(pilco, m, s, 5)
    M2, S2, r2 = predict_wrapper(pilco2, m, s, 5)
    np.testing.assert_allclose(M1, M2, rtol=1e-6)
    np.testing.assert_allclose(S1, S2, rtol=1e-6)
    np.testing.assert_allclose(r1, r2, rtol=1e-6)


if __name__=='__main__':
    test_load_save()

from model.mgpr import MGPR
import numpy as np
import matlab.engine
import os
from gpflow import autoflow
from gpflow import settings

float_type = settings.dtypes.float_type

eng = matlab.engine.start_matlab()
dir_path = os.path.dirname(os.path.realpath(__file__))
eng.cd(dir_path, nargout=0)

@autoflow((float_type,[None, None]), (float_type,[None, None]))
def predict_wrapper(mgpr, m, s):
    return mgpr.predict_on_noisy_inputs(m, s)

def test_predictions():
    np.random.seed(0)
    d = 3  # Input dimension
    k = 2  # Number of outputs

    # Training Dataset
    X0 = np.random.rand(100, d)
    A = np.random.rand(d, k)
    Y0 = np.sin(X0).dot(A) + 1e-3*(np.random.rand(100, k) - 0.5)  #  Just something smooth
    mgpr = MGPR(X0, Y0)

    mgpr.optimize()

    # Generate input
    m = np.random.rand(1, d)  # But MATLAB defines it as m'
    s = np.random.rand(d, d)
    s = s.dot(s.T)  # Make s positive semidefinite

    M, S, V = predict_wrapper(mgpr, m, s)

    # Change the dataset and predict again.
    X0 = 5*np.random.rand(100, d)
    for i in range(k):
        mgpr.models[i].X = X0

    M, S, V = predict_wrapper(mgpr, m, s)

    # convert data to MATLAB
    lengthscales_ = np.stack([model.kern.lengthscales.value for model in mgpr.models])

    variance_ = np.stack([model.kern.variance.value for model in mgpr.models])

    noise_ = np.stack([model.likelihood.variance.value for model in mgpr.models])

    hyp = np.log(np.hstack(
        (lengthscales_,
         np.sqrt(variance_[:, None]),
         np.sqrt(noise_[:, None]))
    )).T
    hyp_mat = matlab.double(hyp.tolist())
    X0_mat = matlab.double(X0.tolist())
    Y0_mat = matlab.double(Y0.tolist())
    m_mat = matlab.double(m.T.tolist())
    s_mat = matlab.double(s.tolist())

    # Call gp0 in matlab
    M_mat, S_mat, V_mat = eng.gp0(hyp_mat, X0_mat, Y0_mat, m_mat, s_mat, nargout=3)
    # Convert outputs to numpy arrays
    M_mat = np.asarray(M_mat)
    S_mat = np.asarray(S_mat)
    V_mat = np.asarray(V_mat)

    assert M.shape == M_mat.T.shape
    assert S.shape == S_mat.shape
    assert V.shape == V_mat.shape
    np.testing.assert_allclose(M, M_mat.T, rtol=1e-5)
    np.testing.assert_allclose(S, S_mat, rtol=1e-5)
    np.testing.assert_allclose(V, V_mat, rtol=1e-5)


if __name__ == '__main__':
    test_predictions()

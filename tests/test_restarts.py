from pilco.models.pilco import PILCO
from pilco.controllers import RbfController
import numpy as np
import tensorflow as tf

# THIS TEST DOESN'T PROVE CORRECTNESS, but if it fails the relevant method IS WRONG
def test_restarts():
    np.random.seed(0)
    d = 2  # State dimenstion
    k = 1  # Controller's output dimension
    horizon = 5
    with tf.Session(graph=tf.Graph()) as sess:
        # Training Dataset
        X0 = np.random.rand(20, d + k)
        A = np.random.rand(d + k, d)
        Y0 = np.sin(X0).dot(A) + 1e-3*(np.random.rand(20, d) - 0.5)  #  Just something smooth

        pilco = PILCO(X0, Y0)
        pilco.optimize(maxiter=5, disp=False)

        old_likelihoods = [m.compute_log_likelihood() for m in pilco.mgpr.models]
        pilco.mgpr.try_restart(sess, restarts=1, verbose=False)
        likelihoods = [m.compute_log_likelihood() for m in pilco.mgpr.models]

        old_reward = pilco.compute_return()
        pilco.restart_controller(sess,restarts=1, verbose=False, maxiter=5, disp=False)
        reward = pilco.compute_return()

        m_init = np.reshape(np.zeros(d), (1,d))
        S_init = 0.1 * np.eye(d)
        controller = RbfController(state_dim=d, control_dim=k, num_basis_functions=5)
        pilco2 = PILCO(X0, Y0, num_induced_points=5, m_init=m_init, S_init=S_init, controller=controller)
        pilco2.optimize(maxiter=5, disp=False)
        pilco2.optimize(maxiter=5, disp=False)

        old_likelihoods2 = [m.compute_log_likelihood() for m in pilco2.mgpr.models]
        pilco2.mgpr.try_restart(sess, restarts=1, verbose=False)
        likelihoods2 = [m.compute_log_likelihood() for m in pilco2.mgpr.models]

        old_reward2 = pilco2.compute_return()
        pilco2.restart_controller(sess,restarts=1, verbose=False, maxiter=5, disp=False)
        reward2 = pilco2.compute_return()

        for i in range(len(likelihoods)):
            assert( np.isclose(likelihoods[i], old_likelihoods[i], rtol=1e-6) or (likelihoods[i] > old_likelihoods[i]))
            assert( np.isclose(likelihoods2[i], old_likelihoods2[i], rtol=1e-6) or (likelihoods2[i] > old_likelihoods2[i]))
        assert(reward >= old_reward)
        assert(reward2 >= old_reward2)

if __name__=='__main__':
    test_restarts()

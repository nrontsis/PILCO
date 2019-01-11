from pilco.models.pilco import PILCO
import numpy as np
import tensorflow as tf

# THIS TEST DOESN'T PROVE CORRECTNESS, but if it fails the relevant method IS WRONG
def test_restarts():
    np.random.seed(0)
    d = 2  # State dimenstion
    k = 1  # Controller's output dimension
    horizon = 10
    with tf.Session(graph=tf.Graph()) as sess:
        # Training Dataset
        X0 = np.random.rand(100, d + k)
        A = np.random.rand(d + k, d)
        Y0 = np.sin(X0).dot(A) + 1e-3*(np.random.rand(100, d) - 0.5)  #  Just something smooth

        pilco = PILCO(X0, Y0)
        pilco.optimize()

        old_likelihoods = [m.compute_log_likelihood() for m in pilco.mgpr.models]
        pilco.mgpr.try_restart(sess, restarts=1, verbose=False)
        likelihoods = [m.compute_log_likelihood() for m in pilco.mgpr.models]

        old_reward = pilco.compute_return()
        pilco.restart_controller(sess,restarts=1, verbose=False)
        reward = pilco.compute_return()

        assert(likelihoods >= old_likelihoods)
        assert(reward >= old_reward)

if __name__=='__main__':
    test_restarts()

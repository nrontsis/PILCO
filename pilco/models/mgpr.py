import tensorflow as tf
from tensorflow_probability import distributions as tfd
import gpflow
import numpy as np
float_type = gpflow.config.default_float()
f64 = gpflow.utilities.to_default_float

def randomize(model):
    mean = 1; sigma = 0.01

    model.kernel.lengthscales.assign(
        mean + sigma*np.random.normal(size=model.kernel.lengthscales.shape))
    model.kernel.variance.assign(
        mean + sigma*np.random.normal(size=model.kernel.variance.shape))
    if model.likelihood.variance.trainable:
        model.likelihood.variance.assign(
            mean + sigma*np.random.normal())

class MGPR(gpflow.Module):
    def __init__(self, data, name=None):
        super(MGPR, self).__init__(name)

        self.num_outputs = data[1].shape[1]
        self.num_dims = data[0].shape[1]
        self.num_datapoints = data[0].shape[0]

        self.create_models(data)
        self.optimizers = []

    def create_models(self, data):
        self.models = []
        for i in range(self.num_outputs):
            kern = gpflow.kernels.SquaredExponential(lengthscales=tf.ones([data[0].shape[1],], dtype=float_type))
            #TODO: Maybe fix noise for better conditioning
            kern.lengthscales.prior = tfd.Gamma(f64(1.1),f64(1/10.0)) # priors have to be included before
            kern.variance.prior = tfd.Gamma(f64(1.5),f64(1/2.0))    # before the model gets compiled
            self.models.append(gpflow.models.GPR((data[0], data[1][:, i:i+1]), kernel=kern))
            self.models[-1].likelihood.prior = tfd.Gamma(f64(1.2),f64(1/0.05))

    def set_data(self, data):
        for i in range(len(self.models)):
            if isinstance(self.models[i].data[0], gpflow.Parameter):
                self.models[i].X.assign(data[0])
                self.models[i].Y.assign(data[1][:, i:i+1])
                self.models[i].data = [self.models[i].X, self.models[i].Y]
            else:
                self.models[i].data = (data[0], data[1][:, i:i+1])

    def optimize(self, restarts=1):
        if len(self.optimizers) == 0:  # This is the first call to optimize();
            for model in self.models:
                # Create an gpflow.train.ScipyOptimizer object for every model embedded in mgpr
                optimizer = gpflow.optimizers.Scipy()
                optimizer.minimize(model.training_loss, model.trainable_variables)
                self.optimizers.append(optimizer)
        else:
            for model, optimizer in zip(self.models, self.optimizers):
                #session = optimizer._model.enquire_session(None)
                optimizer.minimize(model.training_loss, model.trainable_variables)

        for model, optimizer in zip(self.models, self.optimizers):
            # session = optimizer._model.enquire_session(None)
            best_params = {
                "lengthscales" : model.kernel.lengthscales.value(),
                "k_variance" : model.kernel.variance.value(),
                "l_variance" : model.likelihood.variance.value()}
            #best_likelihood = model.log_marginal_likelihood()
            best_loss = model.training_loss()
            for restart in range(restarts):
                randomize(model)
                optimizer.minimize(model.training_loss, model.trainable_variables)
                #likelihood = model.log_marginal_likelihood()
                loss = model.training_loss()
                if loss < best_loss:
                    best_params["k_lengthscales"] = model.kernel.lengthscales.value()
                    best_params["k_variance"] = model.kernel.variance.value()
                    best_params["l_variance"] = model.likelihood.variance.value()
                    #best_likelihood = likelihood
                    best_loss = model.training_loss()
            model.kernel.lengthscales.assign(best_params["lengthscales"])
            model.kernel.variance.assign(best_params["k_variance"])
            model.likelihood.variance.assign(best_params["l_variance"])

    def predict_on_noisy_inputs(self, m, s):
        iK, beta = self.calculate_factorizations()
        return self.predict_given_factorizations(m, s, iK, beta)

    def calculate_factorizations(self):
        K = self.K(self.X)
        batched_eye = tf.eye(tf.shape(self.X)[0], batch_shape=[self.num_outputs], dtype=float_type)
        L = tf.linalg.cholesky(K + self.noise[:, None, None]*batched_eye)
        iK = tf.linalg.cholesky_solve(L, batched_eye, name='chol1_calc_fact')
        Y_ = tf.transpose(self.Y)[:, :, None]
        # Why do we transpose Y? Maybe we need to change the definition of self.Y() or beta?
        beta = tf.linalg.cholesky_solve(L, Y_, name="chol2_calc_fact")[:, :, 0]
        return iK, beta

    def predict_given_factorizations(self, m, s, iK, beta):
        """
        Approximate GP regression at noisy inputs via moment matching
        IN: mean (m) (row vector) and (s) variance of the state
        OUT: mean (M) (row vector), variance (S) of the action
             and inv(s)*input-ouputcovariance
        """

        s = tf.tile(s[None, None, :, :], [self.num_outputs, self.num_outputs, 1, 1])
        inp = tf.tile(self.centralized_input(m)[None, :, :], [self.num_outputs, 1, 1])

        # Calculate M and V: mean and inv(s) times input-output covariance
        iL = tf.linalg.diag(1/self.lengthscales)
        iN = inp @ iL
        B = iL @ s[0, ...] @ iL + tf.eye(self.num_dims, dtype=float_type)

        # Redefine iN as in^T and t --> t^T
        # B is symmetric so its the same
        t = tf.linalg.matrix_transpose(
                tf.linalg.solve(B, tf.linalg.matrix_transpose(iN), adjoint=True, name='predict_gf_t_calc'),
            )

        lb = tf.exp(-tf.reduce_sum(iN * t, -1)/2) * beta
        tiL = t @ iL
        c = self.variance / tf.sqrt(tf.linalg.det(B))

        M = (tf.reduce_sum(lb, -1) * c)[:, None]
        V = tf.matmul(tiL, lb[:, :, None], adjoint_a=True)[..., 0] * c[:, None]

        # Calculate S: Predictive Covariance
        R = s @ tf.linalg.diag(
                1/tf.square(self.lengthscales[None, :, :]) +
                1/tf.square(self.lengthscales[:, None, :])
            ) + tf.eye(self.num_dims, dtype=float_type)

        # TODO: change this block according to the PR of tensorflow. Maybe move it into a function?
        X = inp[None, :, :, :]/tf.square(self.lengthscales[:, None, None, :])
        X2 = -inp[:, None, :, :]/tf.square(self.lengthscales[None, :, None, :])
        Q = tf.linalg.solve(R, s, name='Q_solve')/2
        Xs = tf.reduce_sum(X @ Q * X, -1)
        X2s = tf.reduce_sum(X2 @ Q * X2, -1)
        maha = -2 * tf.matmul(X @ Q, X2, adjoint_b=True) + \
            Xs[:, :, :, None] + X2s[:, :, None, :]
        #
        k = tf.math.log(self.variance)[:, None] - \
            tf.reduce_sum(tf.square(iN), -1)/2
        L = tf.exp(k[:, None, :, None] + k[None, :, None, :] + maha)
        S = (tf.tile(beta[:, None, None, :], [1, self.num_outputs, 1, 1])
                @ L @
                tf.tile(beta[None, :, :, None], [self.num_outputs, 1, 1, 1])
            )[:, :, 0, 0]

        diagL = tf.transpose(tf.linalg.diag_part(tf.transpose(L)))
        S = S - tf.linalg.diag(tf.reduce_sum(tf.multiply(iK, diagL), [1, 2]))
        S = S / tf.sqrt(tf.linalg.det(R))
        S = S + tf.linalg.diag(self.variance)
        S = S - M @ tf.transpose(M)

        return tf.transpose(M), S, tf.transpose(V)

    def centralized_input(self, m):
        return self.X - m

    def K(self, X1, X2=None):
        return tf.stack(
            [model.kernel.K(X1, X2) for model in self.models]
        )

    @property
    def Y(self):
        return tf.concat(
            [model.data[1] for model in self.models],
            axis = 1
        )

    @property
    def X(self):
        return self.models[0].data[0]

    @property
    def lengthscales(self):
        return tf.stack(
            [model.kernel.lengthscales.value() for model in self.models]
        )

    @property
    def variance(self):
        return tf.stack(
            [model.kernel.variance.value() for model in self.models]
        )

    @property
    def noise(self):
        return tf.stack(
            [model.likelihood.variance.value() for model in self.models]
        )

    @property
    def data(self):
        return (self.X, self.Y)

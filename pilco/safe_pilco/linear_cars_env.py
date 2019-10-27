import numpy as np
from gym import spaces
from gym.core import Env

class LinearCars(Env):
    def __init__(self):
        self.action_space = spaces.Box(low=-0.4, high=0.4, shape=(1,))
        self.observation_space = spaces.Box(low=-100, high=100, shape=(4,))
        self.M = 1 # car mass [kg]
        self.b = 0.001    # friction coef [N/m/s]
        self.Dt = 0.50 # timestep [s]

        self.A = np.array([[0, self.Dt, 0, 0],
                          [0, -self.b*self.Dt/self.M, 0, 0],
                          [0, 0, 0, self.Dt],
                          [0, 0, 0, 0]])

        self.B = np.array([0,self.Dt/self.M, 0, 0]).reshape((4,1))

        self.initial_state = np.array([-6.0, 1.0, -5.0, 1.0]).reshape((4,1))

    def step(self, action):
        self.state += self.A @ self.state + self.B * action
                      #0.1 * np.random.normal(scale=[[1e-3], [1e-3], [1e-3], [0.001]], size=(4,1))

        if self.state[0] < 0:
            reward = -1
        else:
            reward = 1
        return np.reshape(self.state.copy(), (4,)), reward, False, None

    def reset(self):
        self.state = self.initial_state + 0.03 * np.random.normal(size=(4,1))
        return np.reshape(self.state.copy(), (4,))

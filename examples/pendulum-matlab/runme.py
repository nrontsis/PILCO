# Call inside python as
# exec(open("runme.py").read())
# to avoid MATLAB closing if the scirpt ends/crashes.
import matlab.engine
import os
import urllib.request
import zipfile
import numpy as np
import gym
import time

if not os.path.isdir("pilcov0.9"):
    print("Matlab implementation not found in current path.")
    print("Attempting to download now")
    urllib.request.urlretrieve("http://mlg.eng.cam.ac.uk/pilco/release/pilcoV0.9.zip", "pilcoV0.9.zip")
    zip_ref = zipfile.ZipFile("pilcoV0.9.zip", 'r')
    zip_ref.extractall("./")
    zip_ref.close()
    print("Done!")


def convert_to_matlab(x):
    dtheta = x[2]
    cos_theta = x[0]
    sin_theta = x[1]

    theta = np.arctan2(sin_theta, cos_theta)
    return np.array([dtheta, theta, sin_theta, cos_theta])

env = gym.make('Pendulum-v0')

def rollout(policy, timesteps):
    X = []; Y = []
    env.reset()
    x = convert_to_matlab(env.step([0])[0])
    for timestep in range(timesteps):
        env.render()
        u = policy(np.array([x[0], x[2], x[3]]))
        x_new, _, done, _ = env.step(u)
        x_new = convert_to_matlab(x_new) # x_new -> dtheta, theta, sin(theta), cos(theta)
        if done: break
        X.append(np.hstack((x, u)))
        Y.append(x_new[0:2]) # Y -> dtheta, theta
        x = x_new
    return np.stack(X), np.stack(Y)

def random_policy(x):
    return env.action_space.sample()

eng = matlab.engine.start_matlab("-desktop")
# dir_path = os.path.dirname(os.path.realpath(__file__)) + "/matlab-environment"
dir_path = "matlab-environment"
eng.cd(dir_path, nargout=0)

def matlab_policy(x):
    n = x.shape[0]
    s = np.zeros((n,n))
    u = eng.policy_wrapper(matlab.double(x[:, None].tolist()), matlab.double(s.tolist()), nargout=1)
    return np.array([u])

# Initial random rollouts to generate a dataset
X,Y = rollout(policy=random_policy, timesteps=40)
for i in range(1,3):
    X_, Y_ = rollout(policy=random_policy, timesteps=40)
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))

eng.settings_pendulum(nargout=0)
for rollouts in range(10):
    print("Rollout #", rollouts + 1)
    eng.workspace['j'] = rollouts + 1
    eng.workspace['x'] = matlab.double(X.tolist())
    eng.workspace['y'] = matlab.double(Y.tolist())
    eng.trainDynModel(nargout=0)
    start = time.time()
    eng.learnPolicy(nargout=0)
    end = time.time()
    print("Learning of policy done in ", end - start, " seconds.")
    if rollouts > 8:
        import pdb; pdb.set_trace()
    X_new, Y_new = rollout(policy=matlab_policy, timesteps=100)
    # Update dataset
    X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
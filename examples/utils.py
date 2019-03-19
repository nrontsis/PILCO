import numpy as np

def rollout(env, pilco, timesteps, verbose=True, random=False, SUBS=1):
    X = []; Y = []
    x = env.reset()
    for timestep in range(timesteps):
        env.render()
        u = policy(env, pilco, x, random)
        for i in range(SUBS):
            x_new, _, done, _ = env.step(u)
            if done: break
            env.render()
        if verbose:
            print("Action: ", u)
            print("State : ",  x_new)
        X.append(np.hstack((x, u)))
        Y.append(x_new - x)
        x = x_new
        if done: break
    return np.stack(X), np.stack(Y)

def policy(env, pilco, x, random):
    if random:
        return env.action_space.sample()
    else:
        return pilco.compute_action(x[None, :])[0, :]

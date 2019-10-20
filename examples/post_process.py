import numpy as np
import sys
from matplotlib import pyplot as plt

X = []
X_eval =[]
all_returns_sampled = []
all_returns_full = []

paths = ["res/inverted_pendulum_with_restart/", "res/mountain_car_final/", "res/"]   # 'res/inverted_pendulum_with_restart/'
names = ["_inv_pend_", "_mountain_car_", "_pend_swing_up_"]
path = paths[2]
name = names[2]

for i in range(10):
    X.append(np.loadtxt(path + 'X' + name + "seed" + str(i) + '_.csv', delimiter=','))
    X_eval.append(np.loadtxt(path + 'X_eval' + name + 'seed' + str(i) + '_.csv', delimiter=','))
    all_returns_sampled.append(np.loadtxt(path + 'evaluation_returns_sampled'+ name + "seed" + str(i) + '_.csv', delimiter=','))
    all_returns_full.append(np.loadtxt(path + 'evaluation_returns_full' + name + "seed" + str(i) + '_.csv', delimiter=','))

# Lists = [seed, [iteration_number, evaluation_runs] 10 x 5 x 4
evals_run = []
for seed in all_returns_full:
    # average out evaluation runs
    evals_run.append(np.mean(seed, 1))

evals_run = np.array(evals_run)
# average out random seed
means_per_iteration = np.mean(evals_run, 0)
std_per_iteration = np.std(evals_run, 0)

plt.plot(means_per_iteration)
plt.fill_between(range(len(means_per_iteration)), means_per_iteration + std_per_iteration, means_per_iteration - std_per_iteration, alpha=0.3)
#plt.ylim([0,120])
plt.savefig(path+name)
plt.show()



#
# path = sys.argv[1]
#
# '''
# ["Exponential_Penalties/swimmer_new_X", "Exponential_Penalties/swimmer_exp_new_lowPX",
# "Exponential_Penalties/swimmer_finalX", "Exponential_Penalties/swimmer_final_lowPX",
# "Safety_Check/safe_final_X", "PILCO/gym_swimmer_highSubs_highMaxiter_longEval_S5X"]
# '''


# for i in range(1,11):
    # X_ = np.loadtxt("Exponential_Penalties/safe_swimmer_X" + str(i) + ".csv", delimiter=',')
    # X_ = np.loadtxt("Safety_Check/swimmer_run_X" + str(i) + ".csv", delimiter=',')
    # X_ = np.loadtxt("Pilco/pilco_swimmerX" + str(i) + ".csv", delimiter=',')
    # X_ = np.loadtxt("Pilco/safe_swimmer_lowP_X" + str(i) + ".csv", delimiter=',')
    # X_ = np.loadtxt("Pilco/highJ_lowTh_run_X" + str(i) + ".csv", delimiter=',')
    # X_ = np.loadtxt("Pilco/trainable_noise2_X" + str(i) + ".csv", delimiter=',')
    # X_ = np.loadtxt("Safety_Check/trainable_noise_highS_X" + str(i) + ".csv", delimiter=',')
    # X_ = np.loadtxt("Pilco/pilco_exp_newX" + str(i) + ".csv", delimiter=',')
    # X_ = np.loadtxt("Exponential_Penalties/swimmer_exp_new_lowPX" + str(i) + ".csv", delimiter=',')
    # X_ = np.loadtxt("Exponential_Penalties/swimmer_new_X" + str(i) + ".csv", delimiter=',')
    # X_ = np.loadtxt("PILCO/gym_swimmer_highSubs_highMaxiter_longEval_S5X" + str(i) + ".csv", delimiter=',')
    # X_ = np.loadtxt(path + str(i) + ".csv", delimiter=',')

import main
import numpy as np 
import matplotlib.pyplot as plt

fs = 18
degree =4

beta_fix = 4
N_set = [9, 16, 25, 36, 49, 64, 81, 100, 400, 900, 1600, 2500, 3600, 4900, 6400, 8100, 10000, 22500, 40000]
N_set = [9, 16, 25, 36, 49, 64, 81, 100, 10000]
sigma_set =[0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4]
sigma_set = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5]

tau_ave = np.zeros((np.size(sigma_set), np.size(N_set)))
T = 10
dt = 0.01
t = np.arange(0, T, dt)


for i in range(np.size(N_set)):
    for j in range(np.size(sigma_set)):

        des = '../data/grid' + str(degree) + '/size' + str(N_set[i]) + '/beta' + str(beta_fix) + '/strength=' + str(sigma_set[j]) + '_T=' + str(T) + '/'
        tau_file = des + 'lifetime.csv'
        tau = np.array(pd.read_csv(tau_file).iloc[:,: ])
        tau_ave[j, i] = np.mean(tau)
    plt.plot(sigma_set, tau_ave[:, i], label=f'N={N_set[i]}')

plt.xlabel('noise $\\sigma$', fontsize=fs)
plt.ylabel('lifetime $\\tau$', fontsize=fs)
plt.legend()
title = f'$\\beta=${beta_fix}'
plt.show()

import main
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

fs = 18
N_set = [100, 400, 900, 6400]
beta_fix = 4
sigma_set = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]
sigma_set = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5]
T = 100
degree = 4
x_h = np.zeros((np.size(N_set), np.size(sigma_set)))
for N, i in zip(N_set, np.arange(np.size(N_set))):
    for sigma, j in zip(sigma_set, np.arange(np.size(sigma_set))):

        des = '../data/grid' + str(degree) + '/size' + str(N) + '/beta' + str(beta_fix) + '/strength=' + str(sigma) + '_T=' + str(T) + '/'

        x_h_file = des + 'x_h.csv'
        x_h[i, j] = np.array(pd.read_csv(x_h_file, header=None).iloc[0,0])

A = main.network_ensemble_grid(9, 3, 4, 4)
_, x_h_theory = main.stable_state(A, 4)
x_h_theory = np.mean(x_h_theory)
for i in range(np.size(N_set)):

    plt.plot(sigma_set, x_h[i], label=f'$N={N_set[i]}$')

plt.plot(sigma_set, x_h_theory * np.ones(np.size(sigma_set)), label='stable state without noise')
plt.xlabel('noise $\\sigma$', fontsize=fs)
plt.ylabel('$x_H$', fontsize=fs)
plt.title('Higher state from simulation', fontsize=fs)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()

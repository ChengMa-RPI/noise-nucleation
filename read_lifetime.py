import main
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

def tau_ave_realization(des, bins):
    """TODO: Docstring for tau_ave_realization.

    :rho: TODO
    :returns: TODO

    """
    rho_last = np.array(pd.read_csv(des + 'rho.csv', header=None).iloc[:, -1])
    lifetime = np.array(pd.read_csv(des + 'lifetime.csv', header=None).iloc[:,: ])
    if np.sum(rho_last < 1/2 )/np.sum(rho_last) < 1e-3:
        tau = np.mean(lifetime)
        return tau
    else:
        
        num = np.zeros(np.size(bins))
        for i in range(np.size(bins)):
            num[i] = np.sum(lifetime<bins[i])
        p = num/np.size(lifetime) 
        start = int(np.ceil(np.min(lifetime)))
        if 1- p[-1] < 1e-2:
            end = next(x for x, y in enumerate(p) if y > 1-1e-2)
        else:
            end = np.size(bins)

        z = np.polyfit(bins[start:end], np.log(1-p[start:end]), 1)

        tau = start - 1/z[0]
    return tau

def tau_from_rho(des):
    """TODO: Docstring for tau_from_rho.

    :des: TODO
    :returns: TODO

    """
    rho = np.array(pd.read_csv(des + 'rho.csv').iloc[:,: ])
    rho_ave = np.mean(rho, 0)
    if rho_ave[-1] > 1/2:
        tau_ave = next(x for x, y in enumerate(rho_ave) if y > 1/2) * dt
    else:
        tau_ave =  np.size(rho_ave)*dt
    return tau_ave 

fs = 18
degree =4

beta_fix = 4
N_set = [9, 16, 25, 36, 49, 64, 81, 100, 400, 900, 1600, 2500, 3600, 4900, 6400, 8100, 10000, 22500, 40000]
N_set = [9, 16, 25, 36, 49, 64, 81, 100, 10000]
sigma_set =[0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4]
sigma_set = [ 0.5, 0.6, 0.7, 0.8, 0.9, 1]
sigma_set = [0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]
N_set = [25, 100, 400]
T_set = [100]  * 7
tau_ave1 = np.zeros((np.size(sigma_set), np.size(N_set)))
tau_ave2 = np.zeros((np.size(sigma_set), np.size(N_set)))
dt = 0.01
for i in range(np.size(N_set)):
    for j, sigma, T in zip(np.arange(np.size(sigma_set)), sigma_set, T_set):
        t = np.arange(0, T, dt)
        bins = np.arange(0, T, 1)

        des = '../data/grid' + str(degree) + '/size' + str(N_set[i]) + '/beta' + str(beta_fix) + '/strength=' + str(sigma) + '_T=' + str(T) + '/'
        tau_ave1[j, i] = tau_ave_realization(des, bins)
        # tau_ave2[j, i] = tau_from_rho(des)
    plt.loglog(sigma_set, tau_ave1[:, i], '*--', label=f'N={N_set[i]}')

plt.xlabel('noise $\\sigma$', fontsize=fs)
plt.ylabel('lifetime $\\langle \\tau \\rangle$', fontsize=fs)
plt.legend()
plt.title(f'$\\beta=${beta_fix}',fontsize=fs)
plt.show()

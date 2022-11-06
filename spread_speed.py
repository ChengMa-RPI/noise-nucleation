import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
import itertools
import matplotlib as mpl


mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'grey', 'tab:cyan']) * cycler(linestyle=['-', '-']))
mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'grey', 'tab:cyan']) * cycler(linestyle=['-']))


marker = itertools.cycle(('d', 'v', 'o', 'X', '*'))

dynamics = 'mutual'
degree = 4
c = 4 
N = 10000
sigma = 0
realization = 0
T_start = 0
T_end = 200
dt = 0.01
interval= 2000
t_plot = np.arange(T_start, T_end, dt* interval)

des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/'
des_evo = des + 'evolution/'
evolution_file = des_evo + f'realization{realization}_T_{T_start}_{T_end}.npy'
evolution = np.load(evolution_file)
evolution_interval = evolution[::interval]
x_ave = np.mean(evolution_interval, 1)
x_l = evolution_interval[0, 0]
M = int(np.sqrt(N))
center_index = int(N/2+np.sqrt(N)/2)
x_h = evolution_interval[0, center_index]
rho = (evolution_interval - x_l)/(x_h-x_l)  # global state
rho_matrix = rho.reshape(np.size(rho, 0), M, M)

center = int(M/2)
distance = np.arange(0, center, 1)
near_far = np.zeros((center, np.size(rho, 0)))
for i in range( center):
    near_far[i] = np.mean(np.hstack((rho_matrix[:, center, [center+i, center-i]], rho_matrix[:, [center+i, center-i], center])), 1)


def plot_rho_r(rho, r, t_list):
    """plot rho with r, the distance from the introduced droplet at different time. 

    :rho: TODO
    :: TODO
    :returns: TODO

    """
    for t, i in zip(t_list, range(np.size(t_list))):
        plt.semilogy(r, rho[:, i], 'o-', label=f'$t={int(t)}$')
    plt.xlabel('distance $r$', fontsize = fs)
    plt.ylabel('$\\rho$', fontsize= fs)
    plt.subplots_adjust(left=0.15, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(fontsize=legendsize)

fs = 20
ticksize = 14
legendsize=13

plot_rho_r(near_far[:, 1:8], distance, t_plot[1:8])

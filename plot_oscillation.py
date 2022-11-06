import main
import numpy as np 
import matplotlib.pyplot as plt
import os
import ast
import pandas as pd 
import matplotlib as mpl
from cycler import cycler
import seaborn as sns
import itertools
from read_lifetime import tau_fit, tau_average, tau_combine
from matplotlib.colors import LogNorm
import matplotlib.ticker 
import networkx as nx
from scipy.integrate import odeint

mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']) * cycler(linestyle=['-']))
plt.rc('font', family='arial', weight='bold')

fs = 22
ticksize = 16
legendsize= 16
alpha = 0.8
lw = 3
marksize = 8


"from all_dynamics.py"

def eutrophication_lattice(x, t, N, index, degree, A_interaction, c, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :degree: the degree of lattice  
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    a, r, R = arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    x_tile = np.broadcast_to(x, (N,N))  # copy vector x to N rows
    x_j = x_tile[index].reshape(N, degree) # select interaction term j with i
    dxdt = a - r * x + c * x**8 / (x**8 + 1) - 4 * R * x + R * np.sum(A_interaction * x_j, -1)
    return dxdt

def stable_state(A, degree, dynamics, c, low, high, arguments):
    """calculate stables states for a given interaction matrix and dynamics-main.mutual

    :A: Adjacency matrix
    :degree: degree of lattice 
    :returns: stable states for all nodes x_l, x_h

    """
    t = np.arange(0, 10000, 0.01)
    N = np.size(A, -1)
    index = np.where(A!=0)
    A_interaction = A[index].reshape(N, degree)
    if dynamics == 'eutrophication':
        dynamics_function = eutrophication_lattice
    xs_low = odeint(dynamics_function, np.ones(N) * low, t, args=(N, index, degree, A_interaction, c, arguments))[-1]
    xs_high = odeint(dynamics_function, np.ones(N) * high, t, args=(N, index, degree, A_interaction, c, arguments))[-1]
    return xs_low, xs_high

def network_ensemble_grid(N, num_col):
    G = nx.grid_graph(dim=[num_col,int(N/num_col)], periodic=True)
    A = np.array(nx.adjacency_matrix(G).todense())
    return A

"from read_rho_lifetime.py"
def heatmap(dynamics, degree, N, c, sigma, R, realization_index, plot_range, plot_interval, dt, linewidth=0, low=0.1, high=10):
    """plot and save figure for animation

    :des: the destination where data is saved and the figures are put
    :realization_index: which data is chosen
    :plot_range: the last moment to plot 
    :plot_interval: the interval to plot
    :dt: simulation interval
    :returns: None

    """
    des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + f'_R{R}/'
    A = network_ensemble_grid(9, int(np.sqrt(9)))
    xs_low, xs_high = stable_state(A, degree, dynamics, c, low, high, arguments)
    xmin = np.mean(xs_low)
    xmax = np.mean(xs_high)
    des_sub = des + 'heatmap/realization' + str(realization_index) + '/'
    if not os.path.exists(des_sub):
        os.makedirs(des_sub)
    des_file = des + f'evolution/realization{realization_index}_T_{plot_range[0]}_{plot_range[1]}.npy'
    data = np.load(des_file)
    rho = (data - xmin) / (xmax - xmin)
    for i in np.arange(plot_range[0], plot_range[1], plot_interval):
        data_snap = rho[int(i/dt)].reshape(int(np.sqrt(N)), int(np.sqrt(N)))

        fig = sns.heatmap(data_snap, vmin=0, vmax=1, linewidths=linewidth)
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=0.9 * fs)
        """
        data_snap = abs(data_snap)
        data_snap = np.log(data_snap)
        fig = sns.heatmap(data_snap, vmin=-4, vmax=0, linewidths=linewidth)
        """
        fig = fig.get_figure()
        # plt.subplots_adjust(left=0.02, right=0.98, wspace=0.25, hspace=0.25, bottom=0.02, top=0.98)
        plt.subplots_adjust(left=0.15, right=0.88, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
        plt.axis('off')
        #fig.patch.set_alpha(0.)
        # plt.title('time = ' + str(round(i, 2)) )
        #fig.savefig(des_sub + str(int(i/plot_interval)) + '.svg', format="svg")
        fig.savefig(des_sub + str(int(i/plot_interval)) + '.png')

        plt.close()
    return None


def plot_rho_ave(dynamics, arguments, degree, N, c, sigma, R, realization_list, tstart, tend, realization_ave, save_des, low=0.1, high=10):
    """TODO: Docstring for plot_rho.

    :arg1: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + f'_R{R}/'
    A = network_ensemble_grid(9, int(np.sqrt(9)))
    xs_low, xs_high = stable_state(A, degree, dynamics, c, low, high, arguments)
    xs_l = np.mean(xs_low)
    xs_h = np.mean(xs_high)
    data_list = []
    for realization in realization_list:
        des_file = des + f'evolution/realization{realization}_T_{tstart}_{tend}.npy'
        data = np.load(des_file)[::100]
        rho = (data - xs_l) / (xs_h - xs_l)
        data_ave = np.mean(rho, 1)
        data_list.append(data_ave)
    data_list = np.vstack((data_list)).transpose()
    if realization_ave:
        plt.plot(np.mean(data_list, 1), linewidth=2.5, color='#8da0cb', label=f'$N={N}$')
        plt.ylabel('$\\langle \\rho \\rangle$', fontsize=fs)
    else:
        plt.plot(data_list[:, 1:], linewidth=1, color='#66c2a5')
        plt.plot(data_list[:, 0], linewidth=2.5, color='#fc8d62')
        plt.ylabel('$\\rho$', fontsize=fs)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.locator_params(nbins=5)
    plt.xlabel('$t$', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False)
    #plt.savefig(save_des, format="svg") 
    return data_list

def plot_rho_individual(dynamics, arguments, degree, N, c, sigma, R, realization, tstart, tend, individual_ave, save_des, low=0.1, high=10):
    """TODO: Docstring for plot_rho.

    :arg1: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + f'_R{R}/'
    A = network_ensemble_grid(9, int(np.sqrt(9)))
    xs_low, xs_high = stable_state(A, degree, dynamics, c, low, high, arguments)
    xs_l = np.mean(xs_low)
    xs_h = np.mean(xs_high)
    des_file = des + f'evolution/realization{realization}_T_{tstart}_{tend}.npy'
    data = np.load(des_file)[::100]
    rho = (data - xs_l) / (xs_h - xs_l)
    if individual_ave:
        plt.plot(np.mean(rho, 1), linewidth=2.5, color='#e78ac3', label=f'$N={N}$')
        plt.ylabel('$\\langle \\rho \\rangle$', fontsize=fs)
    else:
        plt.plot(rho[:, 1:], linewidth=1, color='#66c2a5')
        plt.plot(rho[:, 0], linewidth=2.5, color='#fc8d62')
        plt.ylabel('$\\rho$', fontsize=fs)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.locator_params(nbins=5)
    plt.xlabel('$t$', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False)
    plt.savefig(save_des, format="svg") 

    return rho


a = 0.5
r = 1
R = 0.02
dynamics = 'eutrophication'
arguments = (a, r, R)
degree = 4
N = 100
c = 1.1
sigma = 0.2
R = 0.02
realization_list = np.arange(0,100,1)
realization =  0
tstart = 0
tend = 1000

realization_ave = 1
individual_ave = 1

save_des = f'../manuscript/response_letter/figure/beta={c}_N={N}_ave.svg'

#data = plot_rho_ave(dynamics, arguments, degree, N, c, sigma, R, realization_list, tstart, tend, realization_ave, save_des)

data = plot_rho_individual(dynamics, arguments, degree, N, c, sigma, R, realization, tstart, tend, individual_ave, save_des)

realization_index = 0
plot_range = [0, 1000]
plot_interval = 10
dt = 0.01
#heatmap(dynamics, degree, N, c, sigma, R, realization_index, plot_range, plot_interval, dt, linewidth=0, low=0.1, high=10)

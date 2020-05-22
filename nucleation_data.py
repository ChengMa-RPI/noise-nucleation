import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from cycler import cycler
import itertools
import matplotlib as mpl

mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'grey', 'tab:cyan']) * cycler(linestyle=['-', '-']))
mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:pink', 'grey', 'tab:cyan']) * cycler(linestyle=['-']))

fs = 20
ticksize = 15
legendsize = 14
def plot_nucleation(dynamics, degree, c, N, sigma, realization): 
    """TODO: Docstring for plot_nucleation.

    :N: TODO
    :sigma: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/'
    data = np.array(pd.read_csv(des + 'nucleation.csv', header=None).iloc[:, :])

    t = data[0]
    cluster, number_l, nucleation = data[1: 1+realization], data[1+realization:1+realization*2], data[1+realization*2:1+realization*3]

    cluster_mean = np.mean(cluster, 0)
    number_l_mean = np.mean(number_l, 0)
    nucleation_mean = np.mean(nucleation, 0)
    index = np.where(nucleation_mean<0)[0][0]
    number_l_plot = number_l_mean[: index]
    nucleation_plot = nucleation_mean[: index]
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('t', fontsize=fs)
    ax1.set_ylabel('nucleation cluster', fontsize = fs)
    ax1.plot(t[: index], nucleation_plot, color=color, label='total')
    ax1.plot(t[: index], nucleation_plot/number_l_plot * 1e4, '--', color=color, label='effective')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=ticksize)
    ax1.tick_params(axis='x', labelsize=ticksize)

    color = 'tab:blue'
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('nodes in $x_L$', fontsize = fs)  # we already handled the x-label with ax1
    ax2.plot(t[: index], number_l_plot, color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize = ticksize)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend(fontsize=legendsize)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.show()
    return None

def nucleation_sigma(dynamics, degree, c, N, sigma_set, realization, plot_type):
    """TODO: Docstring for nucleation_sigma.

    :dynamics: TODO
    :degree: TODO
    :c: TODO
    :N: TODO
    :sigma: TODO
    :: TODO
    :returns: TODO

    """
    for sigma in sigma_set:
        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/'
        data = np.array(pd.read_csv(des + 'nucleation.csv', header=None).iloc[:, :])

        t = data[0]
        cluster, number_l, nucleation = data[1: 1+realization], data[1+realization:1+realization*2], data[1+realization*2:1+realization*3]

        cluster_mean = np.mean(cluster, 0)
        number_l_mean = np.mean(number_l, 0)
        nucleation_mean = np.mean(nucleation, 0)
        index = np.where(nucleation_mean<0)[0][0]
        number_l_plot = number_l_mean[: index]
        nucleation_plot = nucleation_mean[: index]
        c_fit = 0.052
        # plt.plot(t[:index], nucleation_plot, label=f'$\\sigma=${sigma}')
        plt.plot(number_l_plot, nucleation_plot/np.exp(-c_fit/sigma**2), label=f'$\\sigma=${sigma}')
    plt.legend(fontsize = legendsize)
    plt.xlabel('$t$', fontsize =fs)
    plt.ylabel('nucleation cluster',fontsize=fs)
    plt.subplots_adjust(left=0.15, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)

    plt.show()

    

dynamics = 'mutual'
degree = 4 
c = 4
N = 10000
realization = 1000
sigma = 0.1
sigma_set = [0.09, 0.1, 0.11]
plot_type = 'nucleation'

# plot_nucleation(dynamics, degree, c, N, sigma, realization)
nucleation_sigma(dynamics, degree, c, N, sigma_set, realization, plot_type)

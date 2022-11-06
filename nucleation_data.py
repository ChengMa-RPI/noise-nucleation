import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from cycler import cycler
import itertools
import matplotlib as mpl

mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'grey', 'tab:cyan']) * cycler(linestyle=['-', '-']))
mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:pink', 'grey', 'tab:cyan', 'tab:purple', 'tab:olive']) * cycler(linestyle=['-']))

plt.rc('text', usetex=True)
plt.rc('font', family='arial', weight='bold')

fs = 35
ticksize = 25
legendsize = 22
lw = 3
alpha = 0.8
def plot_nucleation(dynamics, degree, c, N, sigma, initial_noise): 
    """TODO: Docstring for plot_nucleation.

    :N: TODO
    :sigma: TODO
    :returns: TODO

    """
    if initial_noise == 0:
        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/'
    elif type(initial_noise) == float:
        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_x_i' + str(initial_noise) + '/'
    elif initial_noise == 'metastable':
        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_' + initial_noise + '/'


    data = np.array(pd.read_csv(des + 'nucleation.csv', header=None).iloc[:, :])

    t = data[0]
    realization = int((np.size(data, 0)-1)/4)
    cluster, number_l, nucleation, low = data[1: 1+realization], data[1+realization:1+realization*2], data[1+realization*2:1+realization*3], data[1+realization*3:1+realization*4]

    cluster_mean = np.mean(cluster, 0)
    number_l_mean = np.mean(number_l, 0)
    nucleation_mean = np.mean(nucleation, 0)
    low_mean = np.mean(low, 0)
    "extract low and high stable state"
    des_ave = des + 'ave/'
    ave_data = np.load(des_ave + 'realization0_T_0_100.npy')
    low_mean = ave_data[0] + low_mean * (ave_data[-1] - ave_data[0])
    low_mean = np.mean(low, 0)

    index = np.min(np.where(number_l < N/100)[1])
    number_l_plot = number_l_mean[: index]
    nucleation_plot = nucleation_mean[: index]
    low_plot = low_mean[: index]
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('t', fontsize=fs)
    ax1.set_ylabel('nucleation rate', fontsize = fs)
    # ax1.plot(t[: index], nucleation_plot, color=color, label='total')
    ax1.plot(t[: index], nucleation_plot/number_l_plot * 1e4, 'o--', color=color, label='effective', ms= 10, lw=lw, alpha=alpha)
    # ax1.plot(t[: index], nucleation_plot/number_l_mean[1:index+1] * 1e4, '--', color=color, label='effective')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=ticksize)
    ax1.tick_params(axis='x', labelsize=ticksize)

    color = 'tab:blue'
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('average $\\rho_L$', fontsize = fs)  # we already handled the x-label with ax1
    ax2.plot(t[: index], low_plot, color=color, lw=lw, alpha=alpha)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.tick_params(axis='y', labelcolor=color, labelsize = ticksize)
    ax2.yaxis.get_offset_text().set_fontsize(ticksize-3)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # ax1.legend(fontsize=legendsize)
    plt.subplots_adjust(left=0.20, right=0.87, wspace=0.25, hspace=0.25, bottom=0.20, top=0.93)
    plt.savefig("../summery/F5c.svg", format="svg") 
 


    #plt.show()
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

def plot_x_l(dynamics, degree, c, N, sigma_set, initial_noise):
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
        if initial_noise == 0:
            des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/'
        else:
            des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_x_i' + str(initial_noise) + '/'
        data = np.array(pd.read_csv(des + 'nucleation.csv', header=None).iloc[:, :])
        t = data[0]
        realization = int((np.size(data, 0)-1)/4)
        cluster, number_l, nucleation, low = data[1: 1+realization], data[1+realization:1+realization*2], data[1+realization*2:1+realization*3], data[1+realization*3:1+realization*4]

        cluster_mean = np.mean(cluster, 0)
        number_l_mean = np.mean(number_l, 0)
        nucleation_mean = np.mean(nucleation, 0)
        low_mean = np.mean(low, 0)
        if np.sum(number_l == 0):
            if N == 1 or N == 9:
                index = np.unique(np.where(number_l[:, -1]==N)[0])
                low_mean = np.mean(low[index], 0)
                t = t[:80]
                low_mean = low_mean[:80]
            else:
                index = np.min(np.where(number_l ==0)[1])
                t = t[: index]
                low_mean = low_mean[:index]
        plt.plot(t[1:], low_mean[1:], label=f'$\\sigma=${sigma}',linewidth=2, alpha =0.8)
    plt.legend(fontsize = legendsize)
    plt.xlabel('$t$', fontsize =fs)
    plt.ylabel('low average',fontsize=fs)
    plt.subplots_adjust(left=0.15, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.96)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # plt.show()
    return low_mean[1:]

def plot_xstat(dynamics, degree, c, N, sigma, initial_noise):
    """TODO: Docstring for plot_xstat.

    :dynamics: TODO
    :: TODO
    :returns: TODO

    """
    if initial_noise == 0:
        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/'
    elif type(initial_noise) == float:
        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_x_i' + str(initial_noise) + '/'
    elif initial_noise == 'metastable':
        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_' + initial_noise + '/'



    data = np.array(pd.read_csv(des + 'xstat.csv', header=None).iloc[:, :])
    bins, hists = data[0], data[1:]
    bins_small = hists[:, bins<1.2]
    for i in range(0, 5, 1):
        #plt.loglog(bins, hists[i]/1000/np.sum(bins_small[i]), label=f'$t={i}$')
        plt.loglog(bins, hists[i]/1000, label=f'$t={i}$')
    plt.xscale('symlog') 
    plt.xlabel('$\\rho$', fontsize =fs)
    plt.ylabel('Number of nodes', fontsize =fs)
    plt.legend()
    plt.xticks((-0.1, 0, 0.05))
    #plt.xlim(-0.019, 0.07)
    # plt.ylim(1e1, 1e4)
    plt.subplots_adjust(left=0.15, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.96)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)

    # plt.show()
    return bins_small

dynamics = 'mutual'
degree = 4 
c = 4
N = 2500
N = 10000
sigma = 0.08
sigma_set = [0.01, 0.03, 0.04, 0.05, 0.08, 0.09, 0.1, 0.11, 0.12]
initial_noise = 'metastable'
initial_noise = 0
plot_type = 'nucleation'

plot_nucleation(dynamics, degree, c, N, sigma, initial_noise)
# nucleation_sigma(dynamics, degree, c, N, sigma_set, realization, plot_type)
# plot_x_l(dynamics, degree, c, N, sigma_set, initial_noise)
# plot_xstat(dynamics, degree, c, N, sigma, initial_noise)

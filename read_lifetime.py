import main
import os
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from matplotlib.colors import LogNorm
from cycler import cycler
import itertools
import matplotlib as mpl

mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'grey', 'tab:cyan']) * cycler(linestyle=['-']))
mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'grey', 'tab:cyan']) * cycler(linestyle=['-', '-']))


marker = itertools.cycle(('d', 'v', 'o', 'X', '*'))

def tau_fit(des, bins):
    """TODO: Docstring for tau_ave_realization.

    :rho: TODO
    :returns: TODO

    """
    lifetime = np.array(pd.read_csv(des + 'lifetime.csv', header=None).iloc[:,: ])
    # lifetime = np.array(pd.read_csv(des + 'mfpt.csv', header=None).iloc[:,: ])
    num = np.zeros(np.size(bins))
    for i in range(np.size(bins)):
        num[i] = np.sum(lifetime<bins[i])
    p = num/np.size(lifetime) 
    start = next(x for x, y in enumerate(p) if y > 0) 
    if 1- p[-1] < 5e-2:
        end = next(x for x, y in enumerate(p) if y > 1-5e-2)
    else:
        end = np.size(bins)
    z = np.polyfit(bins[start + 100:end], np.log(1-p[start +100:end]), 1, full=True)
    k, b = z[0]
    tn = -1/k
    tg = bins[start]
    # print(z[1]/np.size(bins[start:end]))
    tau = tg + tn
    return tau, tn, tg

def tau_average(des):
    """TODO: Docstring for tau_average.

    :des: TODO
    :returns: TODO

    """
    lifetime = np.array(pd.read_csv(des + 'lifetime.csv', header=None).iloc[:,: ])
    # lifetime = np.array(pd.read_csv(des + 'mfpt.csv', header=None).iloc[:,: ])
    tau = np.mean(lifetime)
    return tau 

def tau_fit_ave(des, bins):
    """TODO: Docstring for tau_fit_ave.

    :des: TODO
    :bins: TODO
    :returns: TODO

    """
    lifetime = np.array(pd.read_csv(des + 'lifetime.csv', header=None).iloc[:,: ])
    lifetime_max = lifetime.max()
    max_num = np.sum(lifetime == lifetime_max)
    if max_num > 1 and lifetime_max.is_integer():
        print('need fit')
    "fit method"
    num = np.zeros(np.size(bins))
    for i in range(np.size(bins)):
        num[i] = np.sum(lifetime<bins[i])
    p = num/np.size(lifetime) 
    start = next(x for x, y in enumerate(p) if y > 0) 
    if 1- p[-1] < 5e-2:
        end = next(x for x, y in enumerate(p) if y > 1-5e-2)
        if p[end] == 1:
            end = np.where(p<1)[0][-1]
    else:
        end = np.size(bins)
    z = np.polyfit(bins[start:end], np.log(1-p[start:end]), 1)

    tau1 = bins[start] - 1/z[0]
    "average method"
    tau2 = np.mean(lifetime)
    return tau1, bins[start], -1/z[0], tau2

def tau_combine(des, bins, criteria):
    """TODO: Docstring for tau_combine.

    :arg1: TODO
    :returns: TODO

    """
    lifetime = np.array(pd.read_csv(des + 'lifetime.csv', header=None).iloc[:,: ])
    lifetime_max = lifetime.max()
    max_num = np.sum(lifetime == lifetime_max)
    if ( 1 - (max_num-1)/np.size(lifetime) )  < criteria and lifetime_max.is_integer():
        tau, tn, tg = tau_fit(des, bins)
    else:
        tau = tau_average(des)
    return tau

def scaling(dynamics, degree, N, sigma, tau, scaling_method, fit_method):
    """TODO: Docstring for scaling.

    :des: TODO
    :: TODO
    :returns: TODO

    """
    fit_data = np.array(pd.read_csv('../data/' + dynamics + str(degree) +  '/tau_fit.csv', header=None).iloc[:,: ])
    N_all = fit_data[:, 0]
    c_all = fit_data[:, 1]
    if fit_method == 'various':
        fit_c = c_all[N_all == N][0]
    elif fit_method == 'uniform':
        fit_c = np.mean(c_all)

    print(fit_c)
    if scaling_method == 'scaling_all':
        x = np.exp(fit_c/3/sigma**2)/N**(1/2)
        y = tau * np.exp(-fit_c/3/sigma**2)
        x = np.exp(3*fit_c/8/sigma**2)/N**(1/2)
        y = tau * np.exp(-fit_c/4/sigma**2)
        plt.loglog(x, y, '--', marker = next(marker), markersize=8, label=f'N={N}')
        "show y = x**2 line"
        if N == 10000:
            x_standard = np.arange(2, 50, 1)
            plt.loglog(x_standard, x_standard**2, color = 'k', label='$y=x^2$')
    elif scaling_method == 'scaling_single':
        x = 1/sigma**2
        y = tau * N 
        plt.semilogy(x, y, '--', marker= next(marker), markersize=8, label=f'N={N}')
    elif scaling_method == 'scaling_nucleation':
        x = np.exp(-fit_c/sigma**2)
        y = tau
        ax = plt.gca()
        lin, = ax.plot(x, y, '--', lw=2, label=f'N={N}')
        mark, = ax.plot(x, y, linestyle ='None',  marker=next(marker), alpha=.8, ms=8)

        # plt.loglog(x, y, '--', marker = next(marker), alpha = 0.5, markersize=8, label=f'N={N}')
        if N == 10000:
            x_standard = np.arange(1e-8, 2e-7, 1e-7)
            plt.loglog(x_standard, 1/x_standard*1e-4, color = 'k', label='$y \\sim x^{-1}$')
            x_standard = np.arange(1e-4, 1e-2, 1e-3)
            plt.loglog(x_standard, 1/x_standard**(1/3)*1e-0, '--', color = 'k', label='$y \\sim x^{-1/3}$')
            plt.loglog(x_standard, 1/x_standard**(1/4)*4e-0, '--', color = 'r', label='$y \\sim x^{-1/4}$')
    elif scaling_method == 'scaling_nucleation_inverse':
        x = np.exp(fit_c/sigma**2)
        y = tau
        ax = plt.gca()
        lin, = ax.plot(x, y, '--', lw=2, label=f'N={N}')
        mark, = ax.plot(x, y, linestyle ='None',  marker=next(marker), alpha=.8, ms=8)
        # plt.loglog(x, y, '--', marker = next(marker), alpha=0.5, markersize=8, label=f'N={N}')
        if N == 10000:
            x_standard = np.arange(8e6, 1e8, 1e7)
            plt.loglog(x_standard, x_standard*1e-4, color = 'k', label='$y \\sim x^{1}$')
            x_standard = np.arange(1e2, 1e4, 1e3)
            plt.loglog(x_standard, x_standard**(1/3)*1e-0, '--', color = 'k', label='$y \\sim x^{1/3}$')
            plt.loglog(x_standard, x_standard**(1/4)*4, '--', color = 'r', label='$y \\sim x^{1/4}$')

    return x, y

def tau_all(dynamics, N_set, sigma_set, R_set, c, arguments, bins, criteria, fit, powerlaw, plot_type):
    """TODO: Docstring for tau_all.

    :N_set: TODO
    :sigma_set: TODO
    :R_set: TODO
    :returns: TODO

    """
    # fig = plt.figure()
    N_size = np.size(N_set)
    sigma_size = np.size(sigma_set)
    R_size = np.size(R_set)
    tau = np.zeros((N_size, sigma_size, R_size))
    for i, N in zip(range(N_size), N_set):
        for j, sigma in zip(range(sigma_size), sigma_set):
            for k, R in zip(range(R_size), R_set):
                if N == 1 and dynamics == 'mutual':
                    R = 0.2
                elif N == 1 and dynamics != 'mutual':
                    R = 0
                if dynamics != 'quadratic' and R == 0.2:
                    des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/'
                elif dynamics != 'quadratic' and R != 0.2:
                    des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_R' + str(R) + '/'
                elif dynamics == 'quadratic':
                    des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/x2=' + str(c) + '/strength=' + str(sigma) + '/' + f'A1={arguments[0]}_A2={arguments[1]}_R={R}/'
                if plot_type == 'tn':
                    _, tau[i, j, k], _ = tau_fit(des, bins)
                elif plot_type == 'tg':
                    _, _, tau[i, j, k] = tau_fit(des, bins)
                else:
                    tau[i, j, k] = tau_combine(des, bins, criteria)
    if plot_type == 'heatmap':

        if N_size > 1 and sigma_size >1 and R_size == 1:
            data_grid = tau[:, :, 0]
            xlabel = np.array(sigma_set)
            ylabel = np.array(N_set)
            ax = sns.heatmap(data_grid, cmap="YlGnBu", linewidth=0, norm=LogNorm(vmin=data_grid.min(), vmax=data_grid.max()), xticklabels=xlabel, yticklabels=ylabel, cbar_kws={"drawedges":False, 'label': 'Average lifetime $\\langle \\tau \\rangle$'} )
            ax.figure.axes[-1].yaxis.label.set_size(17)

            plt.xlabel('$\\sigma$', fontsize=fs)
            plt.ylabel('$N$', fontsize=fs)
            plt.subplots_adjust(left=0.13, right=0.98, wspace=0.25, hspace=0.25, bottom=0.13, top=0.98)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.show()
    
    else:

        if N_size == 1 and sigma_size > 1 and R_size > 1:
            tau_plot = tau[0]
            for i in range(R_size):
                if powerlaw == 'powerlaw':
                    plt.loglog(sigma_set, tau_plot[:, i], '*--', label=f'R={R_set[i]}')
                elif powerlaw == 'exponential':
                    if fit:
                        plt.semilogy(1/sigma_set**2, tau_plot[:, i], next(marker), markersize = 8)
                        z = np.polyfit(1/sigma_set**2, np.log(tau_plot[:, i]), 1, full=True)
                        k, b = z[0]
                        error = z[1][0]
                        print(k, b, error)
                        plt.semilogy(1/sigma_set**2, np.exp(k * 1/sigma_set**2) *np.exp(b), '--', label=f'R={R_set[i]}')
                    else:
                        plt.semilogy(1/sigma_set**2, tau_plot[:, i],'--', marker=next(marker), markersize=8, label=f'R={R_set[i]}')

            plt.xlabel('noise $\\sigma$', fontsize=fs)
            # plt.title(f'$c=${c}_$N=${N}',fontsize=fs)

        if sigma_size == 1 and N_size >=1 and R_size > 1:
            tau_plot = tau[:, 0, :]
            for i in range(N_size):
                plt.loglog(R_set, tau_plot[i, :], '*--', label=f'N={N_set[i]}')

            plt.xlabel('R', fontsize=fs)
            # plt.title(f'$c=${c}_$\\sigma=${sigma}',fontsize=fs)

        if R_size == 1  and N_size >=1 and sigma_size >= 1:
            tau_plot = tau[:, :, 0]
            sigma_set = np.array(sigma_set)
            N_set = np.array(N_set)
            if plot_type == 'lifetime' or plot_type =='tn':
                for i in range(N_size):
                    if powerlaw == 'powerlaw':
                        plt.loglog(sigma_set, tau_plot[i, :], '*-', label=f'N={N_set[i]}')
                        if fit:
                            k, b = np.polyfit(np.log(sigma_set), np.log(tau_plot[i, :]), 1)
                            plt.loglog(sigma_set, np.exp(b) * sigma_set**k, '--', color = 'k')

                    elif powerlaw == 'exponential_powerlaw':
                        plt.loglog(sigma_set, np.log(tau_plot[i, :]), '*-', label=f'N={N_set[i]}')
                        if fit:
                            k, b = np.polyfit(np.log(sigma_set), np.log(np.log(tau_plot[i, :])), 1)
                            plt.loglog(sigma_set, np.exp(b) * sigma_set**k, '--', color = 'k')

                    elif powerlaw == 'exponential':
                        if fit:
                            plt.semilogy(1/sigma_set**2, tau_plot[i, :], next(marker), markersize = 8)
                            z = np.polyfit(1/sigma_set**2, np.log(tau_plot[i, :]), 1, full=True)
                            k, b = z[0]
                            error = z[1]
                            print(k, b, error)
                            data_df = pd.DataFrame(np.array([N_set[i], k, b, error]).reshape(1, 4))
                            data_df.to_csv('../data/' + dynamics + str(degree) +  '/tau_fit.csv', mode='a', index=False, header=False)
                            plt.semilogy(1/sigma_set**2, np.exp(k * 1/sigma_set**2) *np.exp(b), '--', label=f'N={N_set[i]}')
                            # plt.semilogy(1/sigma_set**2, np.exp(k * 1/sigma_set**2) *np.exp(b), '--', label=f'R={R_set[i]}')
                        else:
                            plt.semilogy(1/sigma_set**2, tau_plot[i, :],'--', marker=next(marker), markersize=8, label=f'N={N_set[i]}')
                    elif powerlaw == 'scaling_all' or powerlaw == 'scaling_single' or powerlaw == 'scaling_nucleation' or powerlaw == 'scaling_nucleation_inverse':
                        fit_method = 'various'
                        scaling(dynamics, degree, N_set[i], sigma_set, tau_plot[i, :], powerlaw, fit_method)


            elif plot_type == 'tg':
                for i in range(np.size(sigma_set)):
                    plt.plot(np.sqrt(N_set), tau[:, i])

            # plt.title(f'$c=${c}',fontsize=fs)

        if powerlaw == 'exponential_powerlaw':
            plt.ylabel('lifetime $\\log(\\langle \\tau \\rangle)$', fontsize=fs)
            plt.xlabel('$\\sigma$', fontsize=fs)
        elif powerlaw == 'powerlaw':
            plt.ylabel('lifetime $\\langle \\tau \\rangle$', fontsize=fs)
            plt.xlabel('$\\sigma$', fontsize=fs)
        elif powerlaw == 'exponential':
            if plot_type == 'lifetime':
                plt.ylabel('lifetime $\\langle \\tau \\rangle$', fontsize=fs)
            elif plot_type == 'tn':
                plt.ylabel('Nucleation time $\\langle t_n \\rangle$', fontsize=fs)
            plt.xlabel('$1/\\sigma^2$', fontsize=fs)
        elif powerlaw == 'scaling_all':
            # plt.xlabel('$e^{c/3\\sigma^2}/\\sqrt{N} $', fontsize =fs)
            # plt.ylabel('$\\langle \\tau \\rangle e^{-c/3\\sigma^2} $', fontsize= fs )
            plt.xlabel('$e^{3c/8\\sigma^2}/\\sqrt{N} $', fontsize =fs)
            plt.ylabel('$\\langle \\tau \\rangle e^{-c/4\\sigma^2} $', fontsize= fs )
        elif powerlaw == 'scaling_single':
            plt.ylabel('$N \\langle \\tau \\rangle  $', fontsize =fs)
            plt.xlabel('$1/\\sigma^2$', fontsize=fs)
        elif powerlaw == 'scaling_nucleation':
            plt.ylabel('$\\langle \\tau \\rangle  $', fontsize =fs)
            plt.xlabel('$e^{-c/\\sigma^2}$', fontsize=fs)
        elif powerlaw == 'scaling_nucleation_inverse':
            plt.ylabel('$\\langle \\tau \\rangle  $', fontsize =fs)
            plt.xlabel('$e^{c/\\sigma^2}$', fontsize=fs)


        plt.subplots_adjust(left=0.15, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.legend(fontsize=legendsize)
        # plt.show()
    #return fig

fs = 20
ticksize = 15
legendsize=14
degree =4
beta_fix = 4
A1 = 0.01
A2 = 0.1
x1 = 1
x2 = 1.2
x3 = 5

dynamics_set = ['mutual', 'harvest', 'eutrophication', 'vegetation', 'quadratic']
c_set = [4, 1.8, 6, 2.7, x2]
index = 0
arguments = (A1, A2, x1, x3)
N_set = [9, 16, 25, 36, 49, 64, 81, 100, 400, 900, 1600, 2500, 3600, 4900, 6400, 8100, 10000, 22500, 40000]
sigma_set = [0.017, 0.018, 0.019, 0.02, 0.021, 0.025, 0.03, 0.1, 0.2, 0.3, 1]
sigma_set = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
N_set = [9, 16, 25, 36, 49, 64, 81, 100, 900, 2500]
N_set = [2500, 900, 100, 9]
N_set = [900]
N_set = [9, 100, 900, 2500]
N_set = [9, 16, 25, 36, 49, 64, 81, 100]
N_set = [9, 16, 36, 100]
R_set = [1, 2, 5, 8, 10, 20, 21, 22]
R_set = [0, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 8, 10, 20, 24]
R_set = [0, 0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26]
R_set = [0.2]
sigma_set = [6e-6, 7e-6, 8e-6, 9e-6, 1e-5]
sigma_set = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005]
sigma_set = [5e-7, 6e-7, 7e-7, 8e-7, 9e-7, 1e-6, 5e-6]
sigma_set = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1]
sigma_set = [0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2]
sigma_set = [0.055, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]
sigma_set = [0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014]
sigma_set = [0.06, 0.061, 0.062, 0.063, 0.064, 0.065, 0.07, 0.08, 0.09, 0.1]
sigma_set = [0.06, 0.061, 0.062, 0.063, 0.07]
sigma_set = [0.065, 0.066, 0.068, 0.07]
sigma_set = [0.055, 0.057, 0.059, 0.06, 0.07]
sigma_set = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
sigma_set = [0.02, 0.021, 0.022, 0.023]
criteria = 1
fit = 1
powerlaw = 'scaling_nucleation'
powerlaw = 'scaling_all'
powerlaw = 'exponential'
plot_type ='lifetime'
plot_range = [0, 10000]
bins = np.logspace(np.log10(plot_range[0] + 0.1), np.log10(plot_range[1]), plot_range[-1] + 1)
bins = np.arange(plot_range[0], plot_range[1], 1) 
# tau_all(dynamics_set[index], N_set, sigma_set, R_set, c_set[index], arguments, bins, criteria, fit, powerlaw, plot_type)
plt.show()

N_sets = [[9], [100], [900], [2500], [1], [2]]
N_sets = [[9], [100], [900], [2500], [10000]]
sigma_sets = [[0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.025], [0.02, 0.021, 0.022, 0.025]]
sigma_sets = [[0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.025], [0.018, 0.019, 0.02, 0.021, 0.022], [0.017, 0.018, 0.019, 0.02, 0.021], [0.017, 0.018, 0.019, 0.02]]

sigma_sets = [[], [], [], [], []]

sigma_sets = [[0.06, 0.061, 0.062, 0.063, 0.07], [0.06, 0.061, 0.062, 0.063, 0.064, 0.065, 0.07], [0.055, 0.057, 0.059, 0.06, 0.07], [0.055, 0.057, 0.059, 0.06, 0.07], [0.07]]
sigma_sets = [sigma_sets[i] + [0.08, 0.09, 0.1] for i in range(5)]

sigma_sets = [[0.007, 0.008, 0.009, 0.01, 0.02], [0.0065, 0.007, 0.008, 0.009, 0.01, 0.02], [0.006, 0.0065, 0.007, 0.008, 0.009, 0.01, 0.02], [0.006, 0.0065, 0.007, 0.008, 0.009, 0.01, 0.02]]

"mutual_both"
sigma_sets = [[0.06, 0.061, 0.062, 0.063], [0.06, 0.061, 0.062, 0.063, 0.064, 0.065], [0.055, 0.057, 0.059, 0.06], [0.055, 0.057, 0.059, 0.06], [0.054, 0.055, 0.056, 0.057, 0.06]]
sigma_sets = [sigma_sets[i] + [0.07, 0.08, 0.085, 0.09, 0.095, 0.1, 0.15] for i in range(5)]

"mutual_multi"
sigma_sets = [[0.08, 0.083, 0.085, 0.087, 0.09, 0.092, 0.095, 0.097, 0.1]]*3

"mutual_single"
sigma_sets = [[0.06, 0.061, 0.062, 0.063, 0.07], [0.06, 0.061, 0.062, 0.063, 0.064, 0.065, 0.07], [0.055, 0.057, 0.059, 0.06], [0.055, 0.057, 0.059, 0.06], [0.053, 0.054, 0.055, 0.056, 0.057]]

for N_set, sigma_set in zip(N_sets[:], sigma_sets[:]):
    tau_all(dynamics_set[index], N_set, sigma_set, R_set, c_set[index], arguments, bins, criteria, fit, powerlaw, plot_type)

'''
N_set = [100]
R_sets = [[0.01], [0.02], [0.1], [0.2], [1], [10]]
sigma_sets = [[0.006, 0.007, 0.008, 0.009], [0.006, 0.007, 0.008, 0.009], [0.012, 0.014, 0.016, 0.018], [0.018, 0.019, 0.02, 0.021, 0.022], [0.04, 0.043, 0.045, 0.047, 0.05], [0.06, 0.07, 0.08, 0.09]]
for R_set, sigma_set in zip(R_sets, sigma_sets):
    tau_all(dynamics_set[index], N_set, sigma_set, R_set, c_set[index], arguments, bins, criteria, fit, powerlaw, plot_type)
'''

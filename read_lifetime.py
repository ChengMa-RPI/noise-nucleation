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
import matplotlib.patches as mpatches

mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'grey', 'tab:cyan']) * cycler(linestyle=['-']))
mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:brown', 'tab:pink', 'grey', 'tab:cyan']) * cycler(linestyle=['-']))

mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'grey', 'tab:cyan']) * cycler(linestyle=['-']))
mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'grey', 'tab:cyan']) * cycler(linestyle=['-', '-']))
mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']) * cycler(linestyle=['-']))
mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']) * cycler(linestyle=['-', '-']))
mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']) * cycler(linestyle=['-']))

color = itertools.cycle(('#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'))
marker = itertools.cycle(('d', 'v', 'o', 'X', '*'))
plt.rc('text', usetex=True)
plt.rc('font', family='arial', weight='bold')

save_des = "../summery/F3a.svg"
fs = 50
ticksize = 40
legendsize= 30
alpha = 0.8
lw = 3
marksize = 15
fs = 22
ticksize = 15
legendsize= 15
alpha = 0.8
lw = 3
marksize = 10


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
    col= next(color)
    mark= next(marker)

    fit_data = np.array(pd.read_csv('../data/' + dynamics + str(degree) +  '/tau_fit.csv', header=None).iloc[:,: ])
    N_all = fit_data[:, 0]
    c_all = fit_data[:, 1]
    if fit_method == 'various':
        fit_c = c_all[N_all == N][0]
    elif fit_method == 'uniform':
        fit_c = np.mean(c_all)
    print(fit_c)
    if scaling_method == 'scaling_all':
        x = np.exp(3*fit_c/8/sigma**2)/N**(1/2)
        y = tau * np.exp(-fit_c/4/sigma**2)
        x = np.exp(fit_c/3/sigma**2)/N**(1/2)
        y = tau * np.exp(-fit_c/3/sigma**2)
        plt.loglog(x, y, '--', lw=lw, marker = mark, markersize=marksize, color=col, label=f'N={N}', alpha = alpha)
        plt.text(0.05, 7, 'multi cluster', fontsize=18)
        plt.text(2, 1e2, 'single cluster', rotation=50, fontsize=18)

        "mutual"
        """
        "vegetation_R002/harvest_R002"
        plt.text(0.02, 30, 'single cluster', fontsize=18)
        plt.text(1, 1e2, 'multi cluster', rotation=57, fontsize=18)
        """
        "eutrophication_R002"
        #plt.text(0.02, 7, 'single cluster', fontsize=fs)
        #plt.text(1, 1e2, 'multi cluster', rotation=57, fontsize=fs)
        "show y = x**2 line"
        if N == 10000:
            x_standard = np.arange(2, 50, 1)
            plt.loglog(x_standard, 2*x_standard**2, color = 'k', label='slope=2', lw=lw, alpha = alpha)
        plt.xlabel('$e^{3c/8\\sigma^2}/\\sqrt{N} $', fontsize =fs)
        plt.ylabel('$\\langle \\tau \\rangle e^{-c/4\\sigma^2} $', fontsize= fs )
        plt.xlabel('$e^{2c/5\\sigma^2}/\\sqrt{N} $', fontsize =fs)
        plt.ylabel('$\\langle \\tau \\rangle e^{-c/5\\sigma^2} $', fontsize= fs )
        plt.xlabel('$e^{c/3\\sigma^2}/\\sqrt{N} $', fontsize =fs)
        plt.ylabel('$\\langle \\tau \\rangle e^{-c/3\\sigma^2} $', fontsize= fs )

    elif scaling_method == 'scaling_single':
        x = 1/sigma**2
        y = tau * N 
        plt.semilogy(x, y, '--', marker= next(marker), markersize=8, label=f'N={N}')
        plt.ylabel('$N \\langle \\tau \\rangle  $', fontsize =fs)
        plt.xlabel('$1/\\sigma^2$', fontsize=fs)
    elif scaling_method == 'scaling_nucleation':
        x = np.exp(-fit_c/sigma**2)
        y = tau
        ax = plt.gca()
        lin, = ax.loglog(x, y, '--', lw=lw, color = col, alpha=alpha)
        mark_line, = ax.loglog(x, y, linestyle ='None',  marker=mark, alpha=alpha, ms=marksize, color=col)
        ax.plot([], [], '--', marker=mark, lw=lw, ms=marksize, color=col, label=f'N={N}', alpha=alpha)

        # plt.loglog(x, y, '--', marker = next(marker), alpha = 0.5, markersize=8, label=f'N={N}')
        if N == 10000:
            x_standard = np.arange(1e-8, 1e-6, 1e-7)
            plt.loglog(x_standard, 1/x_standard*7e-5, color = 'k', label='slope$=-1$', lw=lw, alpha = alpha)
            x_standard = np.arange(1e-5, 5e-3, 1e-3)
            plt.loglog(x_standard, 1/x_standard**(1/3)*9e-1, '--', color = 'k', label='slope$=-\\frac{1}{3}$', lw=lw, alpha = alpha)
            # plt.loglog(x_standard, 1/x_standard**(1/4)*2e-0, '--', color = 'r', label='$y \\sim x^{-1/4}$')

        plt.ylabel('$\\langle \\tau \\rangle  $', fontsize =fs)
        plt.xlabel('$e^{-c/\\sigma^2}$', fontsize=fs)
    elif scaling_method == 'scaling_nucleation_inverse':
        x = np.exp(fit_c/sigma**2)
        y = tau
        ax = plt.gca()
        lin, = ax.plot(x, y, '--', lw=lw, label=f'N={N}')
        mark, = ax.plot(x, y, linestyle ='None',  marker=next(marker), alpha=alpha, ms=marksize)
        # plt.loglog(x, y, '--', marker = next(marker), alpha=0.5, markersize=8, label=f'N={N}')
        if N == 10000:
            x_standard = np.arange(8e6, 1e8, 1e7)
            plt.loglog(x_standard, x_standard*1e-4, color = 'k', label='$y \\sim x^{1}$')
            x_standard = np.arange(1e2, 1e4, 1e3)
            plt.loglog(x_standard, x_standard**(1/3)*1e-0, '--', color = 'k', label='$y \\sim x^{1/3}$')
            plt.loglog(x_standard, x_standard**(1/4)*4, '--', color = 'r', label='$y \\sim x^{1/4}$')
        plt.ylabel('$\\langle \\tau \\rangle  $', fontsize =fs)
        plt.xlabel('$e^{c/\\sigma^2}$', fontsize=fs)

    return x, y

def separation_dot(dynamics, N1, sigma1, sigma2):
    """TODO: Docstring for seperation.

    :dynamics: TODO
    :N: TODO
    :sigma: TODO
    :tau: TODO
    :returns: TODO

    """
    fit_data = np.array(pd.read_csv('../data/' + dynamics + str(degree) +  '/tau_fit.csv', header=None).iloc[:,: ])
    N_all = fit_data[:, 0]
    c_all = fit_data[:, 1]
    fit_c = np.mean(c_all)
    N1_set = []
    N2_set = []
    sigma1_set = []
    sigma2_set = []
    for N, sigma in zip(N1, sigma1):
        N1_set.extend([N] * np.size(sigma))
        sigma1_set.extend(sigma)
    for N, sigma in zip(N1, sigma2):
        N2_set.extend([N] * np.size(sigma))
        sigma2_set.extend(sigma)
    sigma_array = np.linspace(np.min(sigma1_set), np.max(sigma2_set), 10)
    y = np.exp(2*fit_c/3/sigma_array**2)
    plt.semilogy(sigma_array, y, '--', alpha = alpha, linewidth=lw, color='k')
    plt.plot(sigma1_set, N1_set, '*', markersize=8, label='single cluster', color='tab:red')
    plt.plot(sigma2_set, N2_set, 'o', markersize=8, label='multi cluster', color='tab:blue')
    plt.xlabel('$\\sigma$', fontsize=fs)
    plt.ylabel('$N$', fontsize=fs)
    plt.subplots_adjust(left=0.15, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(fontsize=legendsize)
    plt.show()

def separation_fill(dynamics, sigma, gradual):
    fs_local = fs * 0.4
    ticksize_local = ticksize * 0.4
    legendsize_local = legendsize * 0.4
    lw_local = lw * 0.8
    """TODO: Docstring for seperation.

    :dynamics: TODO
    :N: TODO
    :sigma: TODO
    :tau: TODO
    :returns: TODO

    """
    fit_data = np.array(pd.read_csv('../data/' + dynamics + str(degree) +  '/tau_fit.csv', header=None).iloc[:,: ])
    N_all = fit_data[:, 0]
    c_all = fit_data[:, 1]
    fit_c = np.mean(c_all)
    boundary = np.exp(2*fit_c/3/sigma**2)
    N_min = np.min(boundary)
    N_max = np.max(boundary)
    if gradual == True:
        boundary = np.log(boundary)
        logN = np.linspace(np.log(N_min), np.log(N_max) ,100)
        normalN = np.exp(logN)
        distance_single = np.ones((np.size(sigma), 100)) * (-1)
        distance_multi = np.ones((np.size(sigma), 100)) * (-1)
        distance= np.ones((np.size(sigma), 100)) * (-1)
        start = 5
        end= -1
        for x, i in zip(sigma, range(np.size(sigma))):
            for y, j in zip(logN, range(100)):
                dx = (x-sigma[start:end])/ np.max(sigma[start:end])
                dy = (y-boundary[start:end])/ np.max(boundary[start:end])
                if np.sum(dy[dx<0]< 0) > 0:
                    distance[i, j] = -np.min(np.sqrt((dx)**2 + (dy)**2) )
                else:
                    distance[i, j] = np.min(np.sqrt((dx)**2 + (dy)**2) )

        cmap = plt.cm.Reds
        cmap = plt.cm.RdBu
        logdistance = distance.copy()
        
        constant = 1e-5
        constant = 1e-1
        logdistance[distance>=0]= np.log(distance[distance>=0]+ constant) + np.max(abs(np.log(distance[distance>=0]+constant)))

        logdistance[distance<0]= -np.log(-distance[distance<0]+constant) -np.max(abs(np.log(-distance[distance<0]+constant))) 
        plt.contourf(sigma , normalN, logdistance, levels=np.linspace(np.min(logdistance), np.max(logdistance), 100), cmap=cmap, alpha = 0.3*alpha)
        plt.semilogy(sigma, np.exp(boundary), '--', alpha = alpha, linewidth=lw_local, color='k')
        red_patch = mpatches.Patch(color='tab:blue')
        blue_patch = mpatches.Patch(color='tab:red')
        # plt.legend(handles=[red_patch, blue_patch], fontsize=legendsize, fancybox=True, framealpha=0.5)

    else:
        plt.semilogy(sigma, boundary, '--', alpha = alpha, linewidth=lw_local, color='k')
        #plt.fill_between(sigma, N_min, boundary, color='tab:red', alpha = 0.1, label='single cluster')
        #plt.fill_between(sigma, boundary, N_max, color='tab:blue',label='multi cluster')
    plt.xlabel('$\\sigma$', fontsize=fs_local)
    plt.ylabel('$N$', fontsize=fs_local)
    plt.subplots_adjust(left=0.20, right=0.98, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.xticks(fontsize=ticksize_local)
    plt.yticks(fontsize=ticksize_local)
    plt.locator_params(axis='x', nbins=6)
    plt.minorticks_off()
    plt.ylim(5, 2e5)
    plt.xlim(sigma[0], sigma[-5])
    plt.savefig(save_des, format="svg") 
    plt.show()

def separation(dynamics, sigma):
    """TODO: Docstring for seperation.

    :dynamics: TODO
    :N: TODO
    :sigma: TODO
    :tau: TODO
    :returns: TODO

    """
    fit_data = np.array(pd.read_csv('../data/' + dynamics + str(degree) +  '/tau_fit.csv', header=None).iloc[:,: ])
    N_all = fit_data[:, 0]
    c_all = fit_data[:, 1]
    fit_c = np.mean(c_all)
    boundary = np.exp(2*fit_c/3/sigma**2)
    N_min = np.min(boundary)
    N_max = np.max(boundary)

    fig, ax = plt.subplots()
    ax.semilogy(sigma, boundary, '--', alpha = alpha, linewidth=lw, color='k')
    plt.xlabel('$\\sigma$', fontsize=fs*0.6)
    plt.ylabel('$N$', fontsize=fs*0.6)
    plt.subplots_adjust(left=0.20, right=0.98, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.xticks(fontsize=ticksize*0.6)
    plt.yticks(fontsize=ticksize*0.6)
    ax.locator_params(axis='x', nbins=6)
    plt.minorticks_off()
    plt.xlim(sigma[0] *0.9, sigma[-1]*1.02)
    plt.savefig(save_des, format="svg") 
    plt.show()

def tau_all(dynamics, N_set, sigma_set, R_set, c, arguments, bins, criteria, fit, powerlaw, plot_type, initial_noise=0):
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
                    if initial_noise == 0:
                        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/'
                    elif initial_noise == 'metastable':
                        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_metastable/'
                        if not os.path.exists(des):
                            des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/'


                elif dynamics != 'quadratic' and R != 0.2:
                    if initial_noise == 0:
                        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma)  + '_R' + str(R)+ '/'
                    elif initial_noise == 'metastable':
                        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma)  + '_R' + str(R)+ '_metastable/'
                        if not os.path.exists(des):
                            des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma)  + '_R' + str(R)+ '/'

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
            "customize colorbar" 
            ax.figure.axes[-1].yaxis.label.set_size(legendsize)
            plt.gcf().axes[-1].tick_params(labelsize=ticksize-5)
            plt.gcf().axes[-1].minorticks_off()

            plt.xlabel('$\\sigma$', fontsize=fs)
            plt.ylabel('$N$', fontsize=fs)
            plt.subplots_adjust(left=0.18, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
            plt.xticks(fontsize=ticksize-3)
            plt.yticks(fontsize=ticksize-3)
            plt.savefig(save_des, format="svg") 
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

        if R_size == 1 and N_size > 1 and sigma_size > 1:
            if plot_type == 'tn':
                tau_plot = tau[:, :, 0]
                sigma_set = np.array(sigma_set)
                N_set = np.array(N_set)
                for i in range(sigma_size):
                    plt.loglog(N_set, tau_plot[:, i], next(marker), markersize = marksize, label=f'$\\sigma={sigma_set[i]}$')
                    if fit:
                        z = np.polyfit(np.log(N_set), np.log(tau_plot[:, i]), 1, full=True)
                        k, b = z[0]
                        error = z[1]
                        print(k, b, error)
                        plt.loglog(N_set, np.exp(b) * N_set**k, '--')

                plt.xlabel('$N$', fontsize = fs)
                plt.ylabel('Nucleation time $\\langle t_n \\rangle$', fontsize = fs)

        if R_size == 1  and N_size ==1 and sigma_size >= 1:
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
                            plt.semilogy(1/sigma_set**2, tau_plot[i, :], next(marker), markersize = marksize, label=f'N={N_set[i]}', color='tab:red')
                            z = np.polyfit(1/sigma_set**2, np.log(tau_plot[i, :]), 1, full=True)
                            k, b = z[0]
                            error = z[1]
                            print(k, b, error)
                            #data_df = pd.DataFrame(np.array([N_set[i], k, b, error]).reshape(1, 4))
                            #data_df.to_csv('../data/' + dynamics + str(degree) +  '/tau_fit.csv', mode='a', index=False, header=False)
                            plt.semilogy(1/sigma_set**2, np.exp(k * 1/sigma_set**2) *np.exp(b), '--', linewidth=lw, alpha=alpha)
                            #plt.semilogy(1/sigma_set**2, tau_plot[i, :], 'o', markersize = marksize, color='tab:blue')
                            #plt.semilogy(1/sigma_set**2, np.exp(k * 1/sigma_set**2) *np.exp(b), '--', label=f'N={N_set[i]}', linewidth=lw, alpha=alpha, color='tab:blue')
                            # plt.semilogy(1/sigma_set**2, np.exp(k * 1/sigma_set**2) *np.exp(b), '--', label=f'R={R_set[i]}')
                        else:
                            plt.semilogy(1/sigma_set**2, tau_plot[i, :],'--', marker=next(marker), markersize=marksize, label=f'N={N_set[i]}')
                    elif powerlaw == 'scaling_all' or powerlaw == 'scaling_single' or powerlaw == 'scaling_nucleation' or powerlaw == 'scaling_nucleation_inverse':
                        fit_method = 'various'
                        fit_method = 'uniform'
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
                    plt.ylabel('$\\langle \\tau \\rangle$', fontsize=fs)
                elif plot_type == 'tn':
                    plt.ylabel('Nucleation time $\\langle t_n \\rangle$', fontsize=fs)
                plt.xlabel('$1/\\sigma^2$', fontsize=fs)
            """
            elif powerlaw == 'scaling_all':

                plt.xlabel('$e^{3c/8\\sigma^2}/\\sqrt{N} $', fontsize =fs)
                plt.ylabel('$\\langle \\tau \\rangle e^{-c/4\\sigma^2} $', fontsize= fs )
                plt.xlabel('$e^{2c/5\\sigma^2}/\\sqrt{N} $', fontsize =fs)
                plt.ylabel('$\\langle \\tau \\rangle e^{-c/5\\sigma^2} $', fontsize= fs )
                plt.xlabel('$e^{c/3\\sigma^2}/\\sqrt{N} $', fontsize =fs)
                plt.ylabel('$\\langle \\tau \\rangle e^{-c/3\\sigma^2} $', fontsize= fs )
            elif powerlaw == 'scaling_single':
                plt.ylabel('$N \\langle \\tau \\rangle  $', fontsize =fs)
                plt.xlabel('$1/\\sigma^2$', fontsize=fs)
            elif powerlaw == 'scaling_nucleation':
                plt.ylabel('$\\langle \\tau \\rangle  $', fontsize =fs)
                plt.xlabel('$e^{-c/\\sigma^2}$', fontsize=fs)
            elif powerlaw == 'scaling_nucleation_inverse':
                plt.ylabel('$\\langle \\tau \\rangle  $', fontsize =fs)
                plt.xlabel('$e^{c/\\sigma^2}$', fontsize=fs)
            """


        plt.subplots_adjust(left=0.20, right=0.98, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.legend(frameon=False, fontsize = legendsize)
        #plt.savefig(save_des, format="svg") 

        #plt.show()
    #return fig

degree =4
beta_fix = 4
A1 = 0.01
A2 = 0.1
x1 = 1
x2 = 1.2
x3 = 5

dynamics_set = ['mutual', 'harvest', 'eutrophication', 'vegetation', 'quadratic']
c_set = [5, 1.8, 6, 2.6, x2]
index = 0
arguments = (A1, A2, x1, x3)
N_set = [9, 16, 25, 36, 49, 64, 81, 100, 400, 900, 1600, 2500, 3600, 4900, 6400, 8100, 10000, 22500, 40000]
sigma_set = [0.017, 0.018, 0.019, 0.02, 0.021, 0.025, 0.03, 0.1, 0.2, 0.3, 1]
sigma_set = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
N_set = [9, 16, 25, 36, 49, 64, 81, 100, 900, 2500]
N_set = [2500, 900, 100, 9]
N_set = [900]
N_set = [9, 16, 25, 36, 49, 64, 81, 100]
N_set = [9, 100, 900, 2500, 10000][::-1]
N_set = [9, 100, 900, 2500, 10000]
N_set = [1]
R_set = [1, 2, 5, 8, 10, 20, 21, 22]
R_set = [0, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 8, 10, 20, 24]
R_set = [0, 0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26]
R_set = [0.02]
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
sigma_set = [0.02, 0.021, 0.022, 0.023]
sigma_set = [0.0055, 0.0057, 0.006, 0.007, 0.008, 0.009]
sigma_set = [0.007, 0.008, 0.009, 0.01]
sigma_set = [0.055, 0.06, 0.07, 0.08, 0.09, 0.1]
sigma_set = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
criteria = 1
fit = 1
powerlaw = 'scaling_nucleation'
plot_type ='tn'
plot_type = 'heatmap'
plot_type ='lifetime'
powerlaw = 'scaling_all'
powerlaw = 'exponential'
initial_noise = 'metastable'
initial_noise = 0
plot_range = [0, 10000]
bins = np.arange(plot_range[0], plot_range[1], 1) 
bins = np.logspace(np.log10(plot_range[0] + 0.1), np.log10(plot_range[1]), plot_range[-1] + 1)
#tau_all(dynamics_set[index], N_set, sigma_set, R_set, c_set[index], arguments, bins, criteria, fit, powerlaw, plot_type, initial_noise)
#plt.show()

N_sets = [[9], [100], [900], [2500], [1], [2]]
N_sets = [[1], [100], [900], [2500], [10000]]
N_sets = [[9], [100], [900], [2500], [10000]]

sigma_sets = [[], [], [], [], []]





"mutual_multi"
sigma_sets = [[0.08, 0.083, 0.085, 0.087, 0.09, 0.092, 0.095, 0.097, 0.1]]*3


"eutrophication_multi_R0.2"
sigma_sets = [[], [], [], [], []]
sigma_sets = [sigma_sets[i] + [0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03] for i in range(5)]




"eutrophication_single_R0.2"
sigma_sets = [[0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.025], [0.018, 0.019, 0.02, 0.021, 0.022], [0.017, 0.018, 0.019, 0.02], [0.017, 0.018, 0.019, 0.02], [0.016, 0.018]]



"eutrophication_both_R0.2"
sigma_sets = [[0.017], [], [0.017], [0.017], [0.016]]
sigma_sets = [sigma_sets[i] + [0.018, 0.019, 0.02, 0.021, 0.022, 0.025, 0.03, 0.05] for i in range(5)]


"mutual_multi"
sigma_sets = [[0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5]]*5
sigma_sets = [[0.08, 0.1]]*5



"eutrophication_tn_R0.02"
sigma_sets = [[0.007, 0.008, 0.009], [0.0065, 0.007, 0.008, 0.009], [0.006, 0.0065, 0.007, 0.008, 0.009], [0.006, 0.0065, 0.007, 0.008, 0.009], [0.006, 0.0065, 0.007, 0.008, 0.009]]



"harvest_tn_R0.02"
sigma_sets = [[0.035, 0.037, 0.04], [0.035, 0.037, 0.04,], [0.032, 0.033, 0.035, 0.037, 0.04], [0.031, 0.032, 0.033, 0.035, 0.037, 0.04], [0.03, 0.031, 0.032, 0.033, 0.035, 0.037]]

sigma_sets = [sigma_sets[i] + [] for i in range(5)]

"vegetation_tn_R0.02"
sigma_sets = [[0.0055, 0.0057, 0.006], [0.0055, 0.0057, 0.006], [0.0051, 0.0053, 0.0055, 0.0057, 0.006], [0.005, 0.0051, 0.0053, 0.0055, 0.0057, 0.006], [0.0049, 0.005, 0.0051, 0.0053, 0.0055, 0.0057, 0.006]]





"harvest_both_R0.02"
sigma_sets = [[], [], [], [], []]
sigma_sets = [sigma_sets[i] + [0.07, 0.08, 0.09, 0.1] for i in range(5)]



"mutual_multi"
sigma_sets = [[0.08, 0.085, 0.09, 0.095, 0.1]]*5

"mutual_single"
sigma_sets = [[0.063, 0.065, 0.07], [0.06, 0.063, 0.065, 0.07], [0.055, 0.057, 0.059, 0.06, 0.065], [0.055, 0.057, 0.059, 0.06, 0.065], [0.054, 0.055, 0.056, 0.057]]

"harvest_single_R0.02"
sigma_sets = [[0.035, 0.037, 0.04, 0.045], [0.033, 0.035, 0.037, 0.04], [0.032, 0.033, 0.035, 0.037], [0.031, 0.032, 0.033, 0.035], [0.03, 0.031, 0.032, 0.033]]

"harvest_multi_R0.02"
sigma_sets = [[0.06, 0.07, 0.08]] * 5


"eutrophication_single_R0.02"
sigma_sets = [[0.007, 0.008, 0.009, 0.01], [0.0065, 0.007, 0.008, 0.009], [0.006, 0.0065, 0.007, 0.008], [0.006, 0.0065, 0.007], [0.0057, 0.006, 0.0065]]

"eutrophication_multi_R0.02"
sigma_sets = [[0.009,0.01]] * 5


"vegetation_single_R0.02"
sigma_sets = [[0.0057, 0.006, 0.0065, 0.007], [0.0055, 0.0057, 0.006, 0.0065], [0.0051, 0.0053, 0.0055, 0.0057], [0.005, 0.0051, 0.0053, 0.0055], [0.0048, 0.0049, 0.005, 0.0051]]

"vegetation_multi_R0.02"
sigma_sets = [[0.009,0.01]] * 5


"eutrophication_both_R0.02"
sigma_sets = [[0.007, 0.008, 0.009, 0.01, 0.02], [0.0065, 0.007, 0.008, 0.009, 0.01, 0.02], [0.006, 0.0065, 0.007, 0.008, 0.009, 0.01, 0.02], [0.006, 0.0065, 0.007, 0.008, 0.009, 0.01, 0.02], []]


"mutual_tn"
sigma_sets = [[0.06, 0.063, 0.065, 0.07], [0.06, 0.063, 0.065, 0.07], [0.055, 0.057, 0.06, 0.065, 0.07], [0.055, 0.057, 0.06, 0.065, 0.07], [0.055, 0.057, 0.06, 0.065, 0.07]]
sigma_sets = [sigma_sets[i] + [] for i in range(5)]


"harvest_both_R0.02"
sigma_sets = [[0.035, 0.037, 0.04, 0.045, 0.05, 0.06, 0.07], [0.033, 0.035, 0.037, 0.04, 0.045, 0.05, 0.06, 0.07], [0.032, 0.033, 0.035, 0.037, 0.04, 0.045, 0.05, 0.06, 0.07], [0.031, 0.032, 0.033, 0.035, 0.037, 0.04, 0.045, 0.05, 0.06, 0.07 ], [0.03, 0.031, 0.032, 0.033, 0.035, 0.037, 0.04, 0.045, 0.05, 0.06, 0.07]]
sigma_sets = [sigma_sets[i] + [0.08] for i in range(5)]

"eutrophication_both_R0.02"
sigma_sets = [[0.007, 0.008, 0.009, 0.01, 0.02], [0.0065, 0.007, 0.008, 0.009, 0.01, 0.02], [0.006, 0.0065, 0.007, 0.008, 0.009, 0.01, 0.02], [0.006, 0.0065, 0.007, 0.008, 0.009, 0.01, 0.02], [0.0057, 0.006, 0.0065, 0.007, 0.008, 0.009, 0.01, 0.02]]

"vegetation_both_R0.02"
sigma_sets = [[0.0057, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015], [0.0055, 0.0057, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015], [0.0051, 0.0053, 0.0055, 0.0057, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015], [0.005, 0.0051, 0.0053, 0.0055, 0.0057, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015], [0.0048, 0.0049, 0.005, 0.0051, 0.0053, 0.0055, 0.0057, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015]]

"mutual_both"
sigma_sets = [[0.06, 0.063], [0.06, 0.063], [0.055, 0.057, 0.06], [0.055, 0.057, 0.06], [0.053, 0.054, 0.055,  0.057, 0.06]]
sigma_sets = [sigma_sets[i] + [0.065, 0.07, 0.08, 0.085, 0.09, 0.095, 0.1] for i in range(5)]

for N_set, sigma_set in zip(N_sets[:], sigma_sets[:]):
    #tau_all(dynamics_set[index], N_set, sigma_set, R_set, c_set[index], arguments, bins, criteria, fit, powerlaw, plot_type, initial_noise)
    pass

"mutual_separation"
sigma1_set = [[0.06, 0.061, 0.062, 0.063, 0.07, 0.08, 0.085, 0.09], [0.06, 0.061, 0.062, 0.063, 0.064, 0.065, 0.07, 0.08], [0.055, 0.057, 0.059, 0.06], [0.055, 0.057, 0.06], [0.053, 0.055, 0.057, 0.06]]
sigma2_set = [[], [], [0.08, 0.085, 0.09, 0.095, 0.1], [0.08, 0.085, 0.09, 0.095, 0.1], [0.07, 0.08, 0.085, 0.09, 0.095, 0.1]]


"harvest_separation"
sigma1_set = [[0.035, 0.037, 0.04, 0.045, 0.05], [0.035, 0.037, 0.04, 0.045], [0.032, 0.035, 0.037], [0.031,0.033, 0.035], [0.03, 0.031, 0.032, 0.033]]
sigma2_set = [[], [0.06, 0.07, 0.08], [0.05, 0.06, 0.07, 0.08], [0.05, 0.06, 0.07, 0.08], [0.045, 0.05, 0.06, 0.07, 0.08]]

"eutrophication_separation"
sigma1_set = [[0.007, 0.008, 0.009, 0.01], [0.0065, 0.007, 0.008], [0.006, 0.0065, 0.007], [0.006, 0.0065], [0.0057, 0.006]]
sigma2_set = [[], [], [0.01], [0.009, 0.01], [0.008, 0.009, 0.01]]

"vegetation_separation"
sigma1_set = [[0.0055, 0.006, 0.007, 0.008], [0.0055, 0.0057, 0.006, 0.007], [0.005, 0.0051, 0.0053, 0.0055], [0.005, 0.0051, 0.0053, 0.0055], [0.0049, 0.005, 0.0051]]
sigma2_set = [[], [0.009, 0.01], [0.007, 0.008, 0.009, 0.01], [0.007, 0.008, 0.009, 0.01], [0.006, 0.007, 0.008, 0.009, 0.01]]
# separation(dynamics_set[index], N_sets, sigma1_set, sigma2_set)
sigma_set = np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.1])

index = 3
sigma_set = np.linspace(0.0045, 0.015, 100)

index = 2
sigma_set = np.linspace(0.005, 0.015, 100)

index = 1
sigma_set = np.linspace(0.015, 0.075, 100)

index = 0 
sigma_set = np.linspace(0.045, 0.10, 100)
#separation(dynamics_set[index], sigma_set)
#separation_fill(dynamics_set[index], sigma_set, 0)

'''
N_set = [100]
R_sets = [[0.01], [0.02], [0.1], [0.2], [1], [10]]
sigma_sets = [[0.006, 0.007, 0.008, 0.009], [0.006, 0.007, 0.008, 0.009], [0.012, 0.014, 0.016, 0.018], [0.018, 0.019, 0.02, 0.021, 0.022], [0.04, 0.043, 0.045, 0.047, 0.05], [0.06, 0.07, 0.08, 0.09]]
for R_set, sigma_set in zip(R_sets, sigma_sets):
    tau_all(dynamics_set[index], N_set, sigma_set, R_set, c_set[index], arguments, bins, criteria, fit, powerlaw, plot_type)
'''

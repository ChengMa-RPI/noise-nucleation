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

mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:red', 'tab:blue', 'tab:green', 'tab:brown', 'tab:pink', 'grey', 'tab:cyan']) * cycler(linestyle=['-']))
mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:blue', 'tab:red', 'tab:orange', 'tab:grey']) * cycler(linestyle=['-', '-']))
mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:blue', 'tab:orange', 'tab:brown', 'tab:pink', 'grey', 'tab:cyan', 'tab:green', 'tab:red']) * cycler(linestyle=['-']))
mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'grey', 'tab:cyan']) * cycler(linestyle=['-', '-']))
mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:green', 'tab:blue', 'tab:red', 'tab:orange', 'tab:brown', 'tab:pink', 'grey', 'tab:cyan']) * cycler(linestyle=['-', '-']))
mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:green', 'tab:blue', 'tab:red', 'tab:orange', 'tab:brown', 'tab:pink', 'grey', 'tab:cyan']) * cycler(linestyle=['-']))
mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']) * cycler(linestyle=['-', '-']))
linestyle = itertools.cycle(('-', '-.', '--', '--')) 
linestyle = itertools.cycle(('-', '-.', '--')) 
marker = itertools.cycle(('X', 'o'))


plt.rc('text', usetex=True)
plt.rc('font', family='arial', weight='bold')

def heatmap(des, realization_index, N, plot_range, plot_interval, dt, linewidth=0):
    """plot and save figure for animation

    :des: the destination where data is saved and the figures are put
    :realization_index: which data is chosen
    :plot_range: the last moment to plot 
    :plot_interval: the interval to plot
    :dt: simulation interval
    :returns: None

    """
    des_sub = des + 'heatmap/realization' + str(realization_index) + '/'
    if not os.path.exists(des_sub):
        os.makedirs(des_sub)
    des_file = des + f'evolution/realization{realization_index}_T_{plot_range[0]}_{plot_range[1]}.npy'
    data = np.load(des_file)
    xmin = np.mean(data[0])
    xmax = np.mean(data[-1])
    #xmin = np.min(data)
    #xmax = np.max(data)
    rho = (data - xmin) / (xmax - xmin)
    for i in np.arange(0, plot_range[1]-plot_range[0], plot_interval):
        data_snap = rho[int(i/dt)].reshape(int(np.sqrt(N)), int(np.sqrt(N)))
        #fig = sns.heatmap(data_snap, vmin=0, vmax=1, linewidths=linewidth, cbar_kws = {"orientation" : "horizontal"})

        fig = sns.heatmap(data_snap, vmin=0, vmax=1, linewidths=linewidth)
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=0.3 * fs)
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

def rho_from_data(des, plot_num, t, label, title, log, ave, color, fit, interval, fit_rho):
    """read from given destination, plot distribution for P_not. 

    :des: destination where lifetime.csv saves
    :plot_num: how many realizations should be plotted
    :t: time 
    :title: 
    :log: plot log(1-rho) ~ t**3 if log==1; plot 1 - rho ~ t if log==0
    :ave: take average of all data if ave == 1
    :returns: None

    """
    filename = os.listdir(des + 'ave/')
    file_num = np.size(filename)
    if file_num < plot_num:
        print(file_num)
    average_plot = []
    actual_num = 0
    count_num = 0
    while actual_num < plot_num:
        ave_file = des + 'ave/' + filename[count_num]
        if ast.literal_eval(ave_file[ave_file.find('T')+2 : ave_file.rfind('_')] ) == 0:
            average = np.load(ave_file)
            average_plot = np.append(average_plot, average)
            actual_num += 1
        count_num += 1
    average_plot = np.transpose(average_plot.reshape(actual_num, int(np.size(average_plot)/actual_num)))
    x_L = np.mean(average_plot[0])
    x_H = np.mean(average_plot[-1])
    rho_plot = (average_plot -x_L ) / (x_H - x_L)
    rho_plot = rho_plot[: np.size(t)]

    if ave == 1:
        rho_plot = np.mean(rho_plot, -1)
        filenames = os.listdir(des + 'ave/')
        filenum = len(filenames)
        T_start = []
        for ave_file in filenames:
            t0 = ast.literal_eval(ave_file[ave_file.find('T')+2 : ave_file.rfind('_')] )
            t_end = ast.literal_eval(ave_file[ave_file.rfind('_')+1: ave_file.rfind('.')] )
            T_start.append(t0)
        Tmax = np.max(T_start)
        dT = t_end - t0
        rho_average = np.zeros((filenum, (Tmax + dT)*interval[0] ))
        for ave_file, i in zip(filenames, range((filenum))):
            t0 = ast.literal_eval(ave_file[ave_file.find('T')+2 : ave_file.rfind('_')] )
            t_end = ast.literal_eval(ave_file[ave_file.rfind('_')+1: ave_file.rfind('.')] )
            dT = t_end - t0
            if t0 != 0:
                rho_average[i, :t0*interval[0]] = x_L
                rho_average[i, t0*interval[0]: (t0+ 100)*interval[0] ] = np.load(des +'ave/' + ave_file)[::int(100/interval[0])][:-1]
            elif t0 == 0:
                rho_average[i, :dT*interval[0] ] = np.load(des + 'ave/' + ave_file)[::int(100/interval[0])][:-1]
            rho_average[i, (t0+dT)*interval[0]:] = x_H

        rho_ave = np.mean(rho_average, 0) 
        rho_plot = (rho_ave - x_L)/(x_H - x_L)

        lifetime = tau_combine(des, np.arange(0, 1000, 1), 1)
        t = np.linspace(0, (Tmax + dT), (Tmax + dT)*interval[0])

        if log==1:
            if fit == 0:
                t_tau = (t/lifetime) ** 3
                t_tau_plot = t_tau[::interval[1]]
                rho_plot_plot = rho_plot[::interval[1]]
                if label == None:
                    plt.semilogy(t_tau_plot, 1-rho_plot_plot, '.', markersize = 3)

                else:
                    plt.semilogy(t_tau_plot, 1-rho_plot_plot, linewidth=lw, alpha = alpha, linestyle=next(linestyle), label=label)
                    plt.legend(frameon=False, fontsize = legendsize)
                plt.xlabel('$(t/\\tau)^3$' , fontsize = fs)
            elif fit == 1:
                if fit_rho == '3':
                    t_tau = (t/lifetime) ** 3
                    t_tau_plot = t_tau[::interval[1]]
                    rho_plot_plot = rho_plot[::interval[1]]
                    rho_equal_space = [np.argmin(np.abs(np.log10(1-rho_plot+1e-4) - i)) for i in np.arange(0, -4, -0.4)]
                    plt.semilogy(t_tau[rho_equal_space]  , 1-rho_plot[rho_equal_space], next(marker), markersize = 10, linestyle='--', linewidth=lw, label=label)

                    fit_index = np.where(((1-rho_plot)>1e-4))[0]
                    x_fit = t_tau[rho_equal_space]
                    y_fit = np.log(1 - rho_plot[rho_equal_space])
                    z = np.polyfit(x_fit, y_fit, 1)
                    y_theory = np.exp(z[0]*x_fit+z[1])

                    uncertainty_std = np.sqrt(1/(len(x_fit) -2) * np.sum((y_fit - np.log(y_theory)) ** 2) )
                    print(z, uncertainty_std)
                    plt.semilogy(x_fit, y_theory, linestyle='--', alpha = alpha, linewidth=lw)
                    plt.legend( frameon=False, fontsize = legendsize)
                    plt.xlabel('$(t/\\tau)^3$' , fontsize = fs)
                elif fit_rho == 'loglog':
                    "not useful loglog fit"
                    t_tau = (t/lifetime) 
                    if np.sum(rho_plot<0):
                        index_start = np.where(rho_plot<0)[0][-1] + 1
                    else: 
                        index_start = 0
                    if np.sum(rho_plot>1):
                        index_end = np.where(rho_plot>1)[0][0] 
                    else:
                        index_end = -1
                    t_tau_plot = t_tau[index_start: index_end]
                    rho_plot_plot = -np.log(1-rho_plot[index_start: index_end])
                    plt.loglog(t_tau_plot, rho_plot_plot, '.', markersize = 3)
                    fit_index = np.where(((1-rho_plot)<5e-1) &((1-rho_plot)>1e-3))[0]
                    x_fit = t_tau[fit_index]
                    y_fit = rho_plot_plot[fit_index]
                    z = np.polyfit(np.log(x_fit), np.log(y_fit), 1)
                    print(z)
                    plt.loglog(x_fit, np.exp(z[0]*np.log(x_fit)+z[1]), label=label)
                    plt.legend(frameon=False, fontsize = legendsize)
                    plt.xlabel('$(t/\\tau)$', fontsize = fs)


        else:
            if label == None:
                plt.plot(t, 1 - rho_plot)
            else:
                plt.plot(t, 1 - rho_plot, linestyle=next(linestyle), alpha = alpha, linewidth=lw, label=label)
                plt.legend(frameon=False, fontsize = legendsize)
            plt.xlabel('$t$', fontsize = fs)

    else:
        if log == 0 and label == None:
            plt.plot(t, 1 - rho_plot, color=color, linewidth=1)
            plt.plot(t, 1 - rho_plot[:, 1], color='k', linewidth = lw)
            plt.xlabel('$t$', fontsize = fs)
        else:
            print('wrong arguments')
        
    if fit_rho == 'loglog':
        plt.ylabel('$\\ln[-(1 - \\rho)]$', fontsize=fs)
    else:
        plt.ylabel('$1-\\rho$', fontsize=fs)
    # plt.title(title, fontsize=fs)
    return None

def P_from_data(des, plot_num, bins, label, title, log, fit):
    """read from given destination, plot distribution for P_not. 

    :des: destination where lifetime.csv saves
    :bins: plot range and interval
    :label, title: 
    :log: plot log(p_not) if log==1; plot p_not if log==0
    :returns: None

    """

    tau_file = des + 'lifetime.csv'
    # tau_file = des + 'mfpt.csv'
    tau = np.array(pd.read_csv(tau_file, header=None).iloc[:plot_num, :])
    num = np.zeros(np.size(bins))
    for i in range(np.size(bins)):
        num[i] = np.sum(tau<bins[i])
    p = num/np.size(tau) 
    if log == 1:
        if fit == 0:
            if label == None:
                plt.semilogy(bins, 1-p)
            else:
                plt.semilogy(bins, 1-p, label=label)
                plt.legend(frameon=False, fontsize = legendsize)
        elif fit == 1:
            if label == None:
                plt.semilogy(bins, 1-p, '.', linewidth=1)
            else:
                plt.semilogy(bins, 1-p, '.', markersize=1)
            num = np.zeros(np.size(bins))
            for i in range(np.size(bins)):
                num[i] = np.sum(tau<bins[i])
            p = num/np.size(tau) 
            if p[-1]< 0.5 :
                start = next(x for x, y in enumerate(p) if y > 0)  + 50
            else:
                start = next(x for x, y in enumerate(p) if y > 0.1) 
            if 1- p[-1] < 5e-2:
                end = next(x for x, y in enumerate(p) if y > 1-5e-2)
            else:
                end = np.size(bins)
            z = np.polyfit(bins[start:end], np.log(1-p[start:end]), 1, full=True)
            k, b =z[0]
            plt.semilogy(bins[start:], np.exp(bins[start:] * k + b), label=label, linewidth=lw, alpha = 1, linestyle = next(linestyle))
            plt.legend(frameon=False, fontsize = legendsize)
            #return -1/z[0]

    else:
        if label == None:
            plt.plot(bins, 1-p)
        else: 
            plt.plot(bins, 1-p, alpha = alpha, linewidth=lw, label=label, linestyle = next(linestyle))
            plt.legend(frameon=False, fontsize = legendsize)

    plt.ylabel('$P_{\\textrm{not}}$',fontsize=fs) 
    plt.xlabel('$t$', fontsize=fs)
    # plt.title(title , fontsize=fs)

    return None

def tn_N(dynamics, N_set, c, sigma, R, bins):
    exponent = []
    degree = 4
    for N in N_set:
        if R == 0.2:
            des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/'
        else:
            des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_R' + str(R) + '/'
        tau_file = des + 'lifetime.csv'
        tau = np.array(pd.read_csv(tau_file, header=None).iloc[:, :])
        num = np.zeros(np.size(bins))
        for i in range(np.size(bins)):
            num[i] = np.sum(tau<bins[i])
        p = num/np.size(tau) 
        if p[-1]< 0.5 :
            start = next(x for x, y in enumerate(p) if y > 0)  + 100
        else:
            start = next(x for x, y in enumerate(p) if y > 0.5) 
        if 1- p[-1] < 5e-2:
            end = next(x for x, y in enumerate(p) if y > 1-5e-2)
        else:
            end = np.size(bins)

        '''
        start = next(x for x, y in enumerate(p) if y > 0) + 20
        if 1- p[-1] < 5e-2:
            end = next(x for x, y in enumerate(p) if y > 1-5e-2)
            if p[end] == 1:
                end = np.where(p<1)[0][-1]
        else:
            end = np.size(bins)
        '''
        z = np.polyfit(bins[start:end], np.log(1-p[start:end]), 1)
        exponent.append(-1/z[0])
    N_set = np.array(N_set)
    if 2 in N_set:
        N_set[np.where(N_set==2)[0][0]] = 1
    exponent = np.array(exponent)
    plt.plot(N_set, 1/exponent, '.', color='tab:blue')
    k = np.polyfit(N_set, 1/exponent, 1)
    plt.plot(N_set, k[0] * N_set + k[1], color='tab:red')
    plt.xlabel('$N$', fontsize=fs)
    plt.ylabel('$1/\\langle t_n \\rangle$', fontsize=fs)

def plot_P_rho(dynamics, c, R_set, plot_type, N_set, sigma_set, initial_noise=0, arguments=None, plot_num=None, bins=None, t=None, realization_index=None, log=None, ave=None, xlim=None, ylim=None, color=None, fit=None, interval=None, fit_rho=None):
    # fig = plt.figure()
    fig, ax = plt.subplots()
    R_size = np.size(R_set)
    N_size = np.size(N_set)
    sigma_size = np.size(sigma_set)
    if dynamics == 'mutual':
        c_symbol = '$\\beta$'
    else:
        c_symbol = '$c$'
    for N in N_set:
        for sigma in sigma_set:
            for R in R_set:
                
                if dynamics != 'quadratic' and R == 0.2:
                    if initial_noise == 0:
                        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/'
                    elif type(initial_noise) == float:
                        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_x_i' + str(initial_noise) + '/'
                    elif initial_noise == 'metastable':
                        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_' + initial_noise + '/'

                elif dynamics != 'quadratic' and R != 0.2:
                    if initial_noise == 0:
                        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_R' + str(R) + '/'
                    elif type(initial_noise) == float:
                        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_R' + str(R) + '_x_i' + str(initial_noise) + '/'
                    elif initial_noise == 'metastable':
                        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_R' + str(R) + '_' + initial_noise + '/'

                elif dynamics == 'quadratic':
                    des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/x2=' + str(c) + '/strength=' + str(sigma) + '/' + f'A1={arguments[0]}_A2={arguments[1]}_R={R}/'

                if N_size == 1 and sigma_size == 1 and R_size == 1:
                    label = None
                    title = c_symbol + f'$=${c}_N={N}_$\\sigma=${sigma}'
                elif N_size == 1 and sigma_size > 1:
                    label = f'$\\sigma$={sigma}'
                    title = c_symbol + f'$=${c}_$N={N}$'
                elif N_size > 1 and sigma_size == 1:
                    label = f'N={N}'
                    title = c_symbol + f'$=${c}_$\\sigma={sigma}$'
                elif N_size == 1 and sigma_size == 1 and R_size > 1:
                    label = f'R={R}'
                    title = c_symbol + f'$=${c}_N={N}_$\\sigma=${sigma}'
                if plot_type == 'P':
                    P_from_data(des, plot_num, bins, label, title, log, fit)
                elif plot_type == 'rho':
                    rho_from_data(des, plot_num, t, label, title, log, ave, color, fit, interval, fit_rho)
                elif plot_type == 'heatmap':
                    heatmap(des, realization_index, N, plot_range=[0, 1000], plot_interval=10, dt=0.01)
                elif plot_type == 'tn_N':
                    tn_N(dynamics, N_set, c, sigma, R, bins)
                elif plot_type == 'rho_compare_metastable':
                    if R == 0.2:
                        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/'
                    else:
                        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_R'  + str(R) + '/'
                    rho_from_data(des, plot_num, t, '$x_L$', title, log, ave, color, fit, interval, fit_rho)
                    if R == 0.2:
                        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_' + initial_noise + '/'
                    else:
                        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_R' + str(R) + '_' + initial_noise + '/'

                    rho_from_data(des, plot_num, t, 'prepared metastable', title, log, ave, color, fit, interval, fit_rho)


                if ylim != None:
                    plt.ylim(ylim)
                if xlim != None:
                    plt.xlim(xlim)

    if plot_type == 'P' and log == 1 and ylim == None:
        ax.yaxis.set_minor_locator(plt.MaxNLocator(6))
        ax.get_yaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1g') )
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
        ax.tick_params(axis='both', which='minor', labelsize=ticksize*0.9)
    else:
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        if log == 1:
            plt.gcf().axes[-1].minorticks_off()

        else:
            ax.locator_params(axis='y', nbins=3)
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1g') )
    #ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.locator_params(axis='x', nbins=3)

    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    #ax.locator_params(which = 'both', tight=True, nbins=3)
    plt.subplots_adjust(left=0.28, right=0.96, wspace=0.25, hspace=0.25, bottom=0.28, top=0.96)
    #plt.savefig(save_des, format="svg") 
    #plt.close('all')
 

    plt.show()
    return fig

save_des = "../manuscript/F5h_legend.svg"
degree =4
N_set = [9, 16, 25, 36, 49, 64, 81, 100, 400, 900, 1600, 2500, 3600, 4900, 6400, 8100, 10000, 22500, 40000]
T = 1000
dt = 0.01
t = np.arange(0, T, dt)
t = np.linspace(0, T, int(T/dt+1))
plot_num = 300000
plot_num = 100
plot_range = [0, 100]
plot_range = [0, 1000]
plot_interval = 10
bins = np.arange(plot_range[0], plot_range[1], plot_interval) 
bins = np.logspace(np.log10(plot_range[0] + 0.1), np.log10(plot_range[1]), 1001)
realization_index = 1
log = 1
ave = 1
fit = 1
fs = 50 *0.6
ticksize = 40 *0.6
legendsize= 30 *0.7
alpha = 0.8
lw = 3
marksize = 15



A1 = 0.1
A2 = 1
x1 = 1
x2 = 1.2
x3 = 5

dynamics_set = ['mutual', 'harvest', 'eutrophication', 'vegetation', 'quadratic']
c_set = [4, 1.8, 1, 2.6, x2]
index = 0
R_set = [0, 0.0005, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5, 1]
R_set = [20, 21, 22, 23, 24]
R_set = [0.02]
R_set = [0.2]
N_set = [10000]
N_set = [100, 900, 2500, 10000]
N_set = [900, 2500, 10000]
N_set = [9, 16, 25, 36, 49, 64, 81, 100]
N_set = [10000]
sigma_set = [0.08]
sigma_set = [0.08, 0.085, 0.09, 0.095, 0.1]



arguments = (A1, A2, x1, x3)
plot_type = 'P'
plot_type = 'heatmap'
plot_type = 'rho_compare_metastable'
plot_type = 'rho'
color ='tab:red'
interval= [10, 10] 
initial_noise = 0
initial_noise = 'metastable'
fit_rho = '3'
ylim = [1e-4, 1.5]
xlim = [-5, 310]
ylim = None
xlim = None
ylim = [1e-4, 1.5]
xlim = [-1, 12]
plot_P_rho(dynamics_set[index], c_set[index], R_set, plot_type, N_set, sigma_set, initial_noise, arguments, plot_num, bins, t, realization_index,  log, ave, xlim, ylim, color, fit, interval, fit_rho)

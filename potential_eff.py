import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
import sys
sys.path.insert(1, '/home/mac/RPI/research/')

from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, network_generate, stable_state
from read_lifetime import tau_fit, tau_average, tau_combine

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sin
import matplotlib as mpl
import main
from scipy.integrate import odeint
import pandas as pd 

import networkx as nx
K = 10
a = 0.5
r = 1
rv = 0.5
hv = 0.2
R = 0.001
fs = 20
B = 0.1
C = 1
K_mutual = 5
D = 5 
E = 0.9
H = 0.1

alpha = 0.8
lw = 3
marksize = 10
fs = 25
ticksize = 18
legendsize= 20


#color = itertools.cycle(('#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'))



def V_x(x, beta):
    """effective potential

    :x: TODO
    :beta: TODO
    :returns: TODO

    """
    V_x = B + x * (1 - x/K) * (x/C - 1) + beta * x ** 2 / (D + (E + H) *x)  
    return V_x

def mutual_1D(x, t, c, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :neighbor: the degree of each node 
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    B, C, D, E, H, K = arguments
    dxdt = B + x * (1 - x/K) * ( x/C - 1) + c * x**2 / (D + (E+H) * x)
    return dxdt

def harvest_1D(x, t, c, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :c: bifurcation parameter 
    :returns: derivative of x 

    """
    r, K = arguments
    dxdt = r * x * (1 - x/K) - c * x**2 / (x**2 + 1)
    return dxdt

def eutrophication_1D(x, t, c, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :c: bifurcation parameter 
    :returns: derivative of x 

    """
    a, r = arguments
    dxdt = a - r * x + c * x**8 / (x**8 + 1)
    return dxdt

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

def vegetation_1D(x, t, c, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :c: bifurcation parameter 
    :returns: derivative of x 

    """
    r, rv, hv = arguments 
    dxdt = rv * x * (1 - x * (r**4 + (hv * c / (hv + x))**4)/r**4)
    return dxdt

def potential(dynamics, c, arguments, x_list):
    """TODO: Docstring for potential.

    :dynamics: TODO
    :arguments: TODO
    :returns: TODO

    """
    V_list = []
    for x1 in x_list:
        V, error = sin.quad(dynamics, 0, x1, args=(0, c, arguments))
        V_list.append(-V)
    V_list = np.array(V_list)

    return V_list

def plot_potential(dynamics, c, arguments, x, x_fix, color_line, save_des):
    """TODO: Docstring for plot_potential.

    :dynamics: TODO
    :c: TODO
    :arguments: TODO
    :x: TODO
    :returns: TODO

    """

    V = potential(dynamics, c, arguments, x)

    V_fix = [potential(dynamics, c, arguments, [x_i])[0] for x_i in x_fix]
    plt.plot(x, V, alpha=1, linewidth=lw, color=color_line)
    colors = ['#fb9a99', '#fb9a99', '#a6cee3']
    for x_i, V_i, color_i in zip(x_fix, V_fix, colors):
        plt.plot(x_i, V_i, 'o', alpha = 1 , markersize=marksize, color = color_i)
    plt.xlabel('$x$', fontsize=fs)
    plt.ylabel('$V_{\\mathrm{eff}}$', fontsize=fs)
    plt.xticks(fontsize=ticksize) 
    plt.yticks(fontsize=ticksize) 
    plt.locator_params(which='both', tight=True, nbins=6)
    plt.subplots_adjust(left=0.20, right=0.98, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.savefig(save_des, format="svg") 

    plt.show()

def transition_noise(dynamics, c, arguments, strength, low_high, color_line, save_des):
    """TODO: Docstring for transition_noise.

    :dynamics: TODO
    :t: TODO
    :c: TODO
    :arguments: TODO
    :strength: TODO
    :returns: TODO

    """
    dt = 0.01
    t = np.arange(0, 200, dt)
    xs_low = odeint(dynamics, 0.1, t, args=(c, arguments) )[-1]
    xs_high = odeint(dynamics, 5.0, t, args=(c, arguments) )[-1]
    local_state = np.random.RandomState(1)
    t = np.arange(0, 1000, dt)
    noise= local_state.normal(0, np.sqrt(dt), (np.size(t)-1, 1)) * strength
    if low_high:
        x_initial = xs_low
    else:
        x_initial = xs_high
    dyn = main.sdesolver(main.close(dynamics, *(c, arguments)), x_initial, t, dW=noise.reshape(np.size(noise), 1) )  
    rho = (dyn - xs_low)/(xs_high - xs_low)
    plt.plot(t, rho, alpha=1, linewidth=2, color=color_line)
    plt.xlabel('$t$', fontsize=fs)
    plt.ylabel('$\\rho$', fontsize=fs)
    plt.xticks(fontsize=ticksize) 
    plt.yticks(fontsize=ticksize) 
    plt.locator_params(which='both', tight=True, nbins=6)
    plt.subplots_adjust(left=0.20, right=0.98, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.savefig(save_des, format="svg") 
    plt.show()
    return None

def bifurcation(dynamics, save_des):
    degree = 4
    bifurcation_file = '../data/' + dynamics + str(degree) + '/' + 'bifurcation.csv'
    data = np.array(pd.read_csv(bifurcation_file, header=None).iloc[:, 60:])
    # extract data of bifurcation parameter and fixed points 
    c = data[0]
    data_use = np.where((c<8) & (c> 0.1))[0]
    c = data[0, data_use]
    stable = data[1:3, data_use].transpose()
    unstable = data[-1, data_use]
 
    bifur_index = np.where(unstable != 0)[0]
    bifur_c = c[bifur_index]
    bifur_stable = stable[bifur_index]
    bifur_unstable = unstable[bifur_index]
    unstable_positive = unstable[unstable != 0]
    color_line = '#f4cae4'
    plt.semilogx(c[:bifur_index[-1]], stable[:bifur_index[-1], 0], linewidth = 3, color=color_line)

    plt.semilogx(c[bifur_index[0]:], stable[bifur_index[0]:, 1],  linewidth = 3, color=color_line)
    plt.semilogx(c[unstable != 0], unstable_positive, '--',  linewidth = 3, color=color_line)
 
    plt.plot(bifur_c[-1] * np.ones(100), np.linspace(0, bifur_stable[-1, 1],100), '--', color='#e6f5c9' )
    plt.plot((bifur_c[0] -0.01) * np.ones(100), np.linspace(0, bifur_stable[0, 1],100), '--', color='#e6f5c9')

    plt.plot(0.87 * np.ones(100), np.linspace(0, bifur_stable[np.abs(0.87-bifur_c)<1e-10, 1],100), '--', linewidth=2.5, color='#fdcdac' )
    plt.plot(1 * np.ones(100), np.linspace(0, bifur_stable[np.abs(1-bifur_c)<1e-10, 1],100), '--', linewidth=2.5, color='#b3e2cd' )
    plt.plot(4 * np.ones(100), np.linspace(0, bifur_stable[np.abs(4-bifur_c)<1e-10, 1],100), '--', linewidth=2.5, color='#cbd5e8' )
    #plt.xlabel('$\\beta$', fontsize = fs)
    plt.ylabel('$x$', fontsize = fs)
    plt.text(bifur_c[-1]+0.5, 0-1.5, '$\\beta_{c_2}$',size=fs)
    plt.text(bifur_c[0]-0.5, 0-1.5, '$\\beta_{c_1}$',size=fs)
    plt.text(0.87, 0-1.5, '$\\beta_{1}$',size=fs)
    plt.text(1, 0-1.5, '$\\beta_{2}$',size=fs)
    plt.text(4-0.5, 0-1.5, '$\\beta_{3}$',size=fs)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlim(0.5, 11)
    plt.savefig(save_des, format="svg") 
    plt.show()
    return None 

def mean_field(dynamics, N, c, arguments, R, strength, color_line, save_des):
    """TODO: Docstring for transition_noise.

    :dynamics: TODO
    :t: TODO
    :c: TODO
    :arguments: TODO
    :strength: TODO
    :returns: TODO

    """
    degree = 4
    G = nx.grid_graph(dim=[int(np.sqrt(N)), int(np.sqrt(N))], periodic=True)
    A = np.array(nx.adjacency_matrix(G).todense())
    index = np.where(A!=0)
    A_interaction = A[index].reshape(N, degree)

    local_state = np.random.RandomState(20)
    dt = 0.01
    t = np.arange(0, 300, dt)
    noise=  local_state.normal(0, np.sqrt(dt), (np.size(t)-1, N)) * strength
    _, noise_eff = betaspace(A, noise)
    xs_low = odeint(dynamics, 0.1, np.arange(0, 200, dt), args=(c, arguments) )[-1]
    xs_high = odeint(dynamics, 5.0, np.arange(0, 200, dt), args=(c, arguments) )[-1]
    x_initial = xs_low
    dyn_eff = main.sdesolver(main.close(dynamics, *(c, arguments)), x_initial, t, dW=noise_eff.reshape(np.size(noise_eff), 1) )  
    dyn_multi = main.sdesolver(main.close(globals()[dynamics.__name__[: dynamics.__name__.find('_')] + '_lattice'], *(N, index, degree, A_interaction, c, arguments+(R,))), x_initial * np.ones(N), t, dW=noise)  
    rho_eff = (dyn_eff - xs_low)/(xs_high - xs_low)
    rho_multi = (dyn_multi - xs_low)/(xs_high - xs_low)
    rho_mean = np.mean(rho_multi, 1)
    plt.plot(t, rho_multi[:, 0], '-', alpha=1, linewidth=2, color='#ccebc5', label='multi')
    plt.plot(t, rho_multi[:, 1:], '-', alpha=1, linewidth=2, color='#ccebc5')
    plt.plot(t, rho_mean, '--', alpha=1, linewidth=2, color='#b3cde3', label='average')
    plt.plot(t, rho_eff, '-.', alpha=1, linewidth=2, color='#fbb4ae', label='effective')
    plt.xlabel('$t$', fontsize=fs)
    plt.ylabel('$\\rho$', fontsize=fs)
    plt.xticks(fontsize=ticksize) 
    plt.yticks(fontsize=ticksize) 
    #plt.locator_params(which='both', tight=True, nbins=6)
    plt.xlim(250, 301)
    plt.ylim(0.985, 1.01)
    plt.subplots_adjust(left=0.20, right=0.98, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize)
    plt.savefig('../manuscript/SI/figure/compare_rho_eff_xH.svg', format="svg") 
    plt.show()
    return None

def P_from_data(dynamics, N, c, sigma, R, plot_num, bins):
    """read from given destination, plot distribution for P_not. 

    :des: destination where lifetime.csv saves
    :bins: plot range and interval
    :label, title: 
    :log: plot log(p_not) if log==1; plot p_not if log==0
    :returns: None

    """
    degree = 4
    if dynamics == 'mutual':
        des_nucleation = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/'
        des_effective = '../data/' + dynamics + str(degree) + '/effective/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/'

    else:
        des_nucleation = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + f'_R{R}/'
        des_effective = '../data/' + dynamics + str(degree) + '/effective/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + f'_R{R}/'

    for des, labels, colors in zip([des_nucleation, des_effective], ['multi', 'effective'], ['#ccebc5', '#fbb4ae']):
        tau_file = des + 'lifetime.csv'
        tau= np.array(pd.read_csv(tau_file, header=None).iloc[:plot_num, :])
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
        plt.semilogy(bins, 1-p, '.', linewidth=1, color=colors)
        plt.semilogy(bins[start:], np.exp(bins[start:] * k + b), linewidth=lw, alpha = 1, color=colors, label=labels)
    plt.minorticks_off()
    plt.legend(frameon=False, fontsize = legendsize)
    plt.ylabel('$P_{\\textrm{not}}$',fontsize=fs) 
    plt.xlabel('$t$', fontsize=fs)
    plt.xticks(fontsize=ticksize) 
    plt.yticks(fontsize=ticksize) 
    #plt.locator_params(tight=True, nbins=6)
    plt.subplots_adjust(left=0.20, right=0.98, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.savefig('../manuscript/SI/figure/compare_P.svg', format="svg") 
    return None

def compare_lifetime(dynamics, N, c, sigma_nucleation, sigma_effective, R, bins, criteria=1):
    """TODO: Docstring for lifetime_sigma.

    :arg1: TODO
    :returns: TODO

    """
    degree = 4
    tau_nucleation = np.zeros((np.size(sigma_nucleation))) 
    tau_effective = np.zeros((np.size(sigma_effective))) 
    for i, sigma in enumerate(sigma_nucleation):
        des_nucleation = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) 
        if dynamics == 'mutual':
            des_nucleation += '/'
        else:
            des_nucleation = des_nucleation + f'_R{R}/'
        tau_nucleation[i] = tau_combine(des_nucleation, bins, criteria)
    for i, sigma in enumerate(sigma_effective):
        des_effective = '../data/' + dynamics + str(degree) + '/effective/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) 
        if dynamics == 'mutual':
            des_effective += '/'
        else:
            des_effective = des_effective + f'_R{R}/'
        tau_effective[i] = tau_combine(des_effective, bins, criteria)
    plt.semilogy(1/sigma_nucleation**2, tau_nucleation, '-', marker='o', markersize=10,  linewidth=lw, color='#ccebc5', label='multi')
    plt.semilogy(1/sigma_effective**2, tau_effective, '-.',marker='*', markersize=12, linewidth=lw, color='#fbb4ae', label='effective')
    plt.minorticks_off()
    plt.xlabel('$1/\\sigma ^2$', fontsize=fs)
    plt.ylabel('$\\langle \\tau \\rangle$', fontsize=fs)
    plt.xticks(fontsize=ticksize) 
    plt.yticks(fontsize=ticksize) 
    #plt.locator_params(which='both', tight=True, nbins=6)
    plt.subplots_adjust(left=0.20, right=0.98, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize)
    plt.savefig('../manuscript/SI/figure/compare_tau.svg', format="svg") 
    plt.show()
    return None

def lifetime_sigma(dynamics, N_set, c, sigma_set, R, bins, criteria=1):
    """TODO: Docstring for lifetime_sigma.

    :arg1: TODO
    :returns: TODO

    """
    degree = 4
    color_set = ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4' ] 
    marker_set = ['d', 'v', 'o', 'X']
    for N, sigma_effective, colors, markers in zip(N_set, sigma_set, color_set, marker_set):
        tau_effective = np.zeros((np.size(sigma_effective))) 
        sigma_effective = np.array(sigma_effective)
        for i, sigma in enumerate(sigma_effective):
            des_effective = '../data/' + dynamics + str(degree) + '/effective/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) 
            if dynamics == 'mutual':
                des_effective += '/'
            else:
                des_effective = des_effective + f'_R{R}/'
            tau_effective[i] = tau_combine(des_effective, bins, criteria)
        plt.semilogy(1/sigma_effective**2, tau_effective, '--',marker=markers, markersize=10, linewidth=lw, color=colors, label=f'N={N}')
    plt.minorticks_off()
    plt.xlabel('$1/\\sigma ^2$', fontsize=fs)
    plt.ylabel('$\\langle \\tau \\rangle$', fontsize=fs)
    plt.xticks(fontsize=ticksize) 
    plt.yticks(fontsize=ticksize) 
    #plt.locator_params(which='both', tight=True, nbins=6)
    plt.subplots_adjust(left=0.20, right=0.98, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize)
    plt.savefig('../manuscript/SI/figure/tau_effective.svg', format="svg") 
    plt.show()
    return None




dynamics = eutrophication_1D
arguments = (a, r)


"c = 4"
c = 4
x = np.arange(0.1, 5.5, 0.01)
x_fix = np.array([0.52, 4.50, 0.685])

"c = 3"
c = 3
x = np.arange(0.1, 4.5, 0.01)
x_fix = np.array([0.51, 3.50, 0.73])

"c = 0.87"
c = 0.87
x = np.arange(0.3, 1.5, 0.01)
x_fix = np.array([0.50, 1.23, 1.12])
color_line='#fdcdac'

"c = 1"
c = 1
x = np.arange(0.2, 1.8, 0.01)
x_fix = np.array([0.50, 1.45, 1])
color_line='#b3e2cd'

c = 1.1
x = np.arange(0.2, 1.8, 0.01)
x_fix = np.array([0.50, 1.57, 0.96])
color_line='#b3e2cd'


save_des = f'../manuscript/SI/figure/eutrophication_c={c}_V.svg'
plot_potential(dynamics, c, arguments, x, x_fix, color_line, save_des)


c = 0.87
strength = 0.04
low_high = 0
color_line = '#fdcdac'

c = 4
strength = 0.04
low_high = 1
color_line = '#cbd5e8'

c = 1.0
strength = 0.2
low_high = 1
color_line = '#b3e2cd'


save_des = f'../manuscript/SI/figure/eutrophication_c={c}_rho.svg'
#transition_noise(dynamics, c, arguments, strength, low_high, color_line, save_des)
save_des = f'../manuscript/SI/figure/eutrophication_c={c}_bifurcation_logx.svg'
#bifurcation('eutrophication', save_des)
N = 9
c = 6
strength = 0.02
R = 0.02
#mean_field(dynamics, N, c, arguments, R, strength, color_line, save_des)

N = 9
c = 6
sigma = 0.02
R = 0.02
plot_num = 10000
bins = np.arange(0, 1000, 1)
#P_from_data('eutrophication', N, c, sigma, R, plot_num, bins)
N = 9
sigma_nucleation = np.array([0.0065, 0.007, 0.008, 0.009, 0.01, 0.02])
sigma_effective = np.array([0.016, 0.017, 0.02, 0.03, 0.04, 0.05])
#compare_lifetime('eutrophication', N, c, sigma_nucleation, sigma_effective, R, bins, criteria=1)
N_set = [9, 25, 49, 100]
sigma_set = [[0.017, 0.02, 0.03, 0.04, 0.05, 0.07], [0.03, 0.04, 0.05, 0.06, 0.08, 0.1], [0.04, 0.05,  0.08, 0.1, 0.15], [0.06, 0.07, 0.1, 0.2]]
#lifetime_sigma('eutrophication', N_set, c, sigma_set, R, bins, criteria=1)

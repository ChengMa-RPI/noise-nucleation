import multiprocessing as mp 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.integrate import odeint
import sdeint
from scipy.optimize import fsolve, root
import random 
import networkx as nx
import time 
import os 
import pandas as pd 
from random import gauss
from scipy import interpolate
from numpy import linalg as LA
import seaborn as sns

# dynamics parameter
B = 0.1
C = 1
K = 5
D = 5 
E = 0.9
H = 0.1
fs = 18

def sdesolver(f, y0, tspan, dW):
    """Solve stochastic differential equation using Euler method.

    :f: function that governs the deterministic part
    :y0: initial condition
    :tspan: simulation period
    :dW: independent noise
    :returns: solution of y 

    """
    N = len(tspan)
    d = np.size(y0)
    dt = (tspan[N-1] - tspan[0])/(N - 1)
    # allocate space for result
    y = np.zeros((N, d), dtype=type(y0[0]))
    y[0] = y0
    for n in range(N-1):
        tn = tspan[n]
        yn = y[n]
        dWn = dW[n]
        y[n+1] = yn + f(yn, tn) * dt + dWn
    return y

def mutual(x, t, N, index, neighbor, A_interaction):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :neighbor: the degree of each node 
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    x[np.where(x<0)] = 0  # Negative x is forbidden
    x_tile = np.broadcast_to(x, (N,N))  # copy vector x to N rows
    x_j = x_tile[index]  # select interaction term j with i
    x_i = x_tile.transpose()[index]
    dxdt_ij = A_interaction * x_j / (D + E * x_i + H * x_j)
    dxdt = x * np.add.reduceat(dxdt_ij, np.r_[0, np.cumsum(neighbor)[:-1]]) + B + x * (1 - x/K) * ( x/C - 1)  
    return dxdt

def mutual_lattice(x, t, N, index, degree, A_interaction):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :degree: the degree of lattice  
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    x[np.where(x<0)] = 0  # Negative x is forbidden
    t2 = time.time()
    x_tile = np.broadcast_to(x, (N,N))  # copy vector x to N rows
    x_j = x_tile[index].reshape(N, degree) # select interaction term j with i
    dxdt = B + x * (1 - x/K) * ( x/C - 1) + x * np.sum(A_interaction * x_j / (D + E * x.reshape(N, 1) + H * x_j), -1)
    return dxdt

def stable_state(A, degree):
    """calculate stables states for a given interaction matrix and dynamics-main.mutual

    :A: Adjacency matrix
    :degree: degree of lattice 
    :returns: stable states

    """
    t = np.arange(0, 5000, 0.01)
    N = np.size(A, -1)
    index = np.where(A!=0)
    A_interaction = A[index].reshape(N, degree)
    xs_low = odeint(mutual_lattice, np.ones(N) * 0, t, args=(N, index, degree, A_interaction))[-1]
    xs_high = odeint(mutual_lattice, np.ones(N) * K, t, args=(N, index, degree, A_interaction))[-1]
    return xs_low, xs_high

def decouple(x, t, x_eff, s_in):
    """decouple form of original system 

    :x: N variables, 1 * N vector
    :t: time 
    :x_eff: effective x at lower stable state, constant
    :s_in: degree of each node
    :returns: derivative of x 

    """
    dxdt = B + x * (1 -x/K) * (x/C - 1) + s_in * x * x_eff / (D + E * x + H * x_eff)
    return dxdt

def semi_decouple(x_all, t, beta, s_in):
    """semi_decouple form of original system, decouple form coupled with beta system

    :x_all: N variable together with effective x 
    :t: time 
    :beta: effective interaction strength  
    :s_in: in coming degree of each node
    :returns: derivative of x 

    """
    x = x_all[: -1]
    x_eff = x_all[-1]
    dxdt = B + x * (1 -x/K) * (x/C - 1) + s_in * x * x_eff / (D + E * x + H * x_eff)
    dxdt_eff = B + x_eff * (1-x_eff/K) * (x_eff/C - 1) + beta * x_eff**2 / (D + (E+H) * x_eff)
    return np.hstack((dxdt, dxdt_eff))

def beta_dyn(x, t, beta):
    """dynamics of beta system 

    :x, t: required
    :beta: parameter, effective beta
    :returns: derivative of x 

    """
    dxdt = B + x * (1 -x/K) * (x/C - 1) + beta * x ** 2 / (D + (E + H) *x)
    return dxdt

def eta_diag(x, t, N):
    """noise matrix for semi_decouple system, no correlated term 

    :x, t: required 
    :returns: noise matrix, N * N

    """
    return np.diag(np.ones(N) )

def close(func, *args):
    """closure function to pass parameters to function f and g when using sedint

    :func: function f or g
    :*args: arguments of f or g 
    :returns: function with arguments  

    """
    def new_func(x, t):
        """function with arguments 

        :x, t: required when using deint 
        :returns: f(x, t, args)

        """
        return func(x, t, *args)
    return new_func

def betaspace(A, x):
    """calculate  beta_eff and x_eff from A and x

    :A: adjacency matrix
    :x: state vector
    :returns: TODO

    """
    s_out = A.sum(0)
    s_in = A.sum(-1)
    if sum(s_out) == 0:
        return 0, x[0] 
    else:
        beta_eff = np.mean(s_out * s_in) / np.mean(s_out)
        if np.ndim(x) == 1:
            x_eff = np.mean(x * s_out)/ np.mean(s_out)
        elif np.ndim(x) == 2:  # x is matrix form 
            x_eff = np.mean(x * s_out, -1)/np.mean(s_out)
        return beta_eff, x_eff

def Gcc_A_mat(A):
    """find the survive nodes which is in giant connected components for a given adjacency matrix
    
    :A: original Adjacency matrix
    :returns: A of the giant component, and node index in gcc

    """
    G = nx.from_numpy_matrix(A)
    "only consider giant connected component? "
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)[0]
    survive = list(Gcc)
    A_update = A[survive, :][:, survive] 
    return A_update, survive

def spectral(A, x, noise=None):
    """Calculate parameters of reduced 1D dynamics by eigenvalue theory.

    :A: Adjacency matrix 
    :x: dynamic variable
    :returns: alpha, beta, effective x, effective noise, eigenvalue

    """
    k_in = np.sum(A, -1)
    eigenvalue, eigenvector = LA.eig(A)
    domi_eigenval = np.max(eigenvalue)
    if domi_eigenval < 0:
        print('negative eigenvalue')
    domi_eigenvec = eigenvector[:, np.where(eigenvalue==domi_eigenval)[0][0]]    
    if sum(domi_eigenvec.imag !=0) == 0:
        domi_eigenvec = domi_eigenvec.real
    else:
        print('complex', domi_eigenvec)
    domi_vec_norm = domi_eigenvec / sum(domi_eigenvec)
    alpha = sum(domi_vec_norm * k_in)
    beta = np.sum(domi_vec_norm ** 2 * k_in) / np.sum(domi_vec_norm ** 2) / alpha
    R = sum(domi_vec_norm * x)
    if noise is None:
        noise_eff = 0
    else:
        noise_eff = np.sum(domi_vec_norm * noise, -1)
    return alpha, beta, R, noise_eff, eigenvalue

def beta_transition(dynamics, x_end, T ,dt):
    """TODO: Docstring for transition_time.

    :dynamics: TODO
    :beta: TODO
    :returns: TODO

    """
    result = (dynamics>x_end)
    index = np.where(result == 1)[0]
    if np.size(index) == 0:
        transition = T
    else:
        transition = dt * index[0]
    return transition

def beta_transition_N(dynamics, x_end, T ,dt):
    """TODO: Docstring for transition_time.

    :dynamics: TODO
    :beta: TODO
    :returns: TODO

    """
    result = (dynamics>x_end)
    N = np.size(result, -1)
    transition_set = np.zeros(N)
    for i in range(N):
        index = np.where(result[:, i] == 1)[0]
        if np.size(index) == 0:
            transition_set[i] = T
        else:
            transition_set[i] = dt * index[0]
    return transition_set

def system_transition(dynamics, N, T ,dt, des):
    """determine transition time for given dynamics and criteria

    :dynamics: evolutions in the given period
    :N: the number of variables
    x_end: criteria to determine when the transition takes place
    :Dir: the directory where transition.csv is saved
    :returns: If no transition for any x, return T; if not all of x transition to x_end, return -1; if all of x transition, return transition time for each x. 

    """
    result = np.where(dynamics>K)
    if len(result[0]) == 0:
        transition_set= T * np.ones(N)

    elif np.size((np.unique(result[1] ))) != N :
        transition_set = -1 * np.ones(N) 

    else:
        transition_set = np.zeros(N)
        for i in range(N):
            transition_set[i] = next(x for x, y in enumerate(dynamics[:,i]) if y > K)
        transition_set = transition_set * dt
    df = pd.DataFrame(transition_set.reshape(1, N))
    df.to_csv(des +  'transition.csv', mode='a', index=False, header=False)
    return transition_set

def transition_from_data(N, T, dt, des, start_index, end_index):
    """Find transition time for several realizations.

    :A: adjacency matrix 
    :T, dt: time 
    des: the destination to save transition data
    start_index, end_index: read realization file from start to end
    :returns: None

    """
    t = np.arange(0, T, dt)
    for i in range(start_index, end_index + 1):
        file_name = des + 'realization' + str(i) + '.h5'
        data = np.array(pd.read_hdf(file_name))
        transition = system_transition(data, N, T, dt, des)
    return None

def system_collect(store_index, N, index, degree, A_interaction, strength, x_initial, t, dt, des):
    """one realization to run sdeint and save dynamics

    """
    local_state = np.random.RandomState(store_index)
    noise= local_state.normal(0, np.sqrt(dt), (np.size(t)-1, N)) * strength
    dyn_all = sdesolver(close(mutual_lattice, *(N, index, degree, A_interaction)), x_initial, t, dW = noise)
    des_file = des + 'realization' + str(store_index) + '.h5'
    if os.path.exists(des_file):
        print(f'file exists!{store_index}')
    data = pd.DataFrame(dyn_all)
    data.to_hdf(des_file, key='data', mode='w')

    return None

def network_ensemble_ER(N_original, p, seed, beta_fix):

    N = 0
    while N != N_original:
        G = nx.fast_gnp_random_graph(N_original, p, seed=seed)
        np.random.seed(seed)
        weights = np.random.rand(N_original, N_original)
        A_original = np.array(nx.adjacency_matrix(G).todense()) * weights
        A_connected, _, _  = Gcc_A_mat(A_original, np.ones(N_original, []))
        N = np.size(A_connected, -1)
        seed = seed + 1
    np.random.seed(None)
    beta_connected, _ = betaspace(A_connected, np.ones(N))
    factor = beta_fix / beta_connected
    A = A_connected * factor
    beta_eff, _ = betaspace(A, np.ones(N))
    if abs(beta_eff - beta_fix) > 1e-10:
        print('rescale fails')
        return None
    else:
        index = np.where(A!=0)
        neighbor = np.sum(A!=0, -1)
        A_interaction = A[index]

        xs_low = odeint(mutual, np.ones(N) * 0, t, args=(A, N, index, neighbor, A_interaction))[-1]
        xs_high = odeint(mutual, np.ones(N) * 5, t, args=(A, N, index, neighbor, A_interaction))[-1]
        alpha, beta, R, noise_eff, eigenvalue = spectral(A, xs_low)
        xs_beta_low = odeint(beta_dyn, 0, t, args=(beta_eff,))[-1, 0]
        xs_beta_high = odeint(beta_dyn, 5, t, args=(beta_eff,))[-1, 0]
        xs_decouple_high = odeint(decouple, np.ones(N) * 5, t, args=(A, xs_beta_low))[-1]
        xs_semi_decouple_high = odeint(semi_decouple, np.ones(N+1) * 5, t, args=(A, beta_eff))[-1]
        return seed, A, beta_eff, xs_low, xs_high, xs_beta_low, xs_beta_high, xs_decouple_high, xs_semi_decouple_high

def network_ensemble_grid(N, num_col, degree, beta_fix):
    G = nx.grid_graph(dim=[num_col,int(N/num_col)], periodic=True)
    A_original = np.array(nx.adjacency_matrix(G).todense())
    beta_connected, _ = betaspace(A_original, np.ones(N))
    factor = beta_fix / beta_connected
    A = A_original * factor
    beta_eff, _ = betaspace(A, np.ones(N))
    if abs(beta_eff - beta_fix) > 1e-10:
        print('rescale fails')
        return None
    else:
        return A

def system_parallel(A, degree, strength, T, dt, parallel, cpu_number, des, exist_index=0):
    """parallel computing or series computing 

    :A: adjacency matrix 
    :strength: std of noise
    :T: evolution time 
    :dt: simulation dt
    :parallel: the number of parallel realizations 
    :cpu_number: parallel computing with cpu_number if positive, and series computing if 0
    :des: destination to store the data
    :exist_index: exist file index
    :returns: None

    """
    N = np.size(A, -1)
    index = np.where(A!=0)
    t = np.arange(0, T ,dt)
    A_interaction = A[index].reshape(N, degree)
    xs_low = odeint(mutual_lattice, np.ones(N) * 0, np.arange(0, 100, 0.01), args=(N, index, degree, A_interaction))[-1]
    if not os.path.exists(des):
        os.makedirs(des)
    des_file = des + 'realization' + str(exist_index) + '.h5'
    while os.path.exists(des_file):
        exist_index += 1
        des_file = des + 'realization' + str(exist_index) + '.h5'
    if cpu_number > 0:
        p = mp.Pool(cpu_number)
        p.starmap_async(system_collect, [(i + exist_index, N, index, degree, A_interaction, strength, xs_low, t, dt, des) for i in range(parallel)]).get()
        p.close()
        p.join()
    else:
        for i in range(parallel):
            system_collect(i + exist_index, N, index, degree, A_interaction, strength, xs_low, t, dt, des)
    return None

def rho_lifetime_saving(realization_range, length, des, T, dt, strong_noise):
    """for given range of realization files [realization_start +1 to realization_end], find rho which is normalized average evolution, and lifetime which is the time when rho exceeds 1/2, and also save x_h data.

    :realization_range: read files from realization_range[0] + 1 to realization_range[1] 
    :length: length of time
    :des: the destination where data is saved
    :x_l, x_h: lower and higher stable states
    :returns: None

    """
    realization_num = realization_range[1] - realization_range[0] 
    x = np.zeros((realization_num, length))
    y = []  # add data that has transitioned 
    tau = np.ones(realization_num) * T 
    for i in range(realization_num):
        des_file = des + 'realization' + str(i+realization_range[0]+1) + '.h5'
        data = np.array(pd.read_hdf(des_file))
        x[i] = np.mean(data, -1)
        if strong_noise == 0:
            if np.sum(data[-1] < K) == 0:
                y.append(x[i, -1])
        else:
            y.append(x[i, -1])
    x_l = np.mean(x[:, 0])
    if np.size(y) != 0:
        x_h = np.mean(y)
        rho = (x - x_l) / (x_h - x_l)
        rho_last = rho[:, -1]
        succeed = np.where(rho_last > 1/2)[0]
        x_h_file = des + 'x_h.csv'
        if os.path.exists(x_h_file):
            x_h_old = np.array(pd.read_csv(x_h_file, header=None).iloc[0, 0])
            x_h = np.mean([x_h_old, x_h])
        pd.DataFrame(np.ones((1,1)) * x_h).to_csv(x_h_file, index=False, header=False)
        for i in succeed:
            tau[i] = dt * next(x for x, y in enumerate(rho[i]) if y > 1/2)
        rho_df = pd.DataFrame(rho)
        rho_df.to_csv(des +  'rho.csv', mode='a', index=False, header=False)
    tau_df = pd.DataFrame(tau.reshape(realization_num, 1))
    tau_df.to_csv(des +  'lifetime.csv', mode='a', index=False, header=False)
    return None

def P_from_data(des, bins, label, title, log):
    """read from given destination, plot distribution for P_not. 

    :des: destination where lifetime.csv saves
    :bins: plot range and interval
    :label, title: 
    :log: plot log(p_not) if log==1; plot p_not if log==0
    :returns: None

    """

    tau_file = des + 'lifetime.csv'
    tau = np.array(pd.read_csv(tau_file, header=None).iloc[:,: ])
    num = np.zeros(np.size(bins))
    for i in range(np.size(bins)):
        num[i] = np.sum(tau<bins[i])
    p = num/np.size(tau) 
    if log == 1:
        plt.semilogy(bins, 1-p, label=label)
    else:
        plt.plot(bins, 1-p, label=label)

    plt.ylabel('$p_{not}$',fontsize=fs) 
    plt.xlabel('t (s)', fontsize=fs)
    plt.title(title , fontsize=fs)
    plt.legend()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return None

def rho_from_data(des, plot_num, t, label, title, log, ave):
    """read from given destination, plot distribution for P_not. 

    :des: destination where lifetime.csv saves
    :plot_num: how many realizations should be plotted
    :t: time 
    :title: 
    :log: plot log(1-rho) ~ t**3 if log==1; plot 1 - rho ~ t if log==0
    :ave: take average of all data if ave == 1
    :returns: None

    """

    rho_file = des + 'rho.csv'
    rho = np.array(pd.read_csv(rho_file, header=None).iloc[:plot_num, :])
    rho_plot = np.transpose(rho)
    if ave == 1:
        rho_plot = np.mean(rho_plot, -1)
    if log==1:
        plt.semilogy(t**3, 1 - rho_plot, label=label)
        plt.xlabel('$t^3 (s^3)$', fontsize = fs)
    else:
        plt.plot(t, 1 - rho_plot, label=label)
        plt.xlabel('$t (s)$', fontsize = fs)
    plt.ylabel('$1 - \\rho$', fontsize=fs)
    plt.title(title, fontsize=fs)
    plt.legend()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return None

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
    des_file = des + 'realization' + str(realization_index) + '.h5'
    data = np.array(pd.read_hdf(des_file))
    xmin = np.mean(data[0])
    if np.sum(data[-1] < K) == 0:
        xmax = np.mean(data[-1])
    elif np.sum(data[-1] > K ) == 0:
        print('No transition')
        return None
    else:
        xmax = np.mean(data[-1, data[-1] > K])
    rho = (data - xmin) / (xmax - xmin)
    for i in np.arange(0, plot_range, plot_interval):
        data_snap = rho[int(i/dt)].reshape(int(np.sqrt(N)), int(np.sqrt(N)))
        fig = sns.heatmap(data_snap, vmin=0, vmax=1, linewidths=linewidth)
        fig = fig.get_figure()
        plt.title('time = ' + str(round(i, 2)) + 's')
        fig.savefig(des_sub + str(int(i/plot_interval)) + '.png')
        plt.close()
    return None


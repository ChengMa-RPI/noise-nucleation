import main
import file_operation
import numpy as np 
from scipy.integrate import odeint 
import matplotlib.pyplot as plt
import time 
import os
import pandas as pd
import multiprocessing as mp
from scipy.optimize import fsolve, root
import networkx as nx
import scipy.integrate as sin
import seaborn as sns


N_set = [100, 400, 900, 1600, 2500]
sigma_set = [0.5]
N_set = [100]
parallel_index_initial = np.arange(100)
parallel_every = 100
degree = 4
cpu_number = 4
T_start = 0
T_end = 10000
T_every = 100
dt = 0.01
K = 10
R = 0.2
r = 1
fs = 18
c = 1.8
remove = 0

def harvest_1D(x, t, c):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :neighbor: the degree of each node 
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    dxdt = r * x * (1 - x/K) - c * x**2 / (x**2 + 1)
    return dxdt

def f(x, c):
    """TODO: Docstring for f.

    :x: TODO
    :c: TODO
    :returns: TODO

    """
    
    return r * x * (1 - x/K) - c * x**2 / (x**2 + 1)

def harvest_lattice(x, t, N, index, degree, A_interaction, c):
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
    x_tile = np.broadcast_to(x, (N,N))  # copy vector x to N rows
    x_j = x_tile[index].reshape(N, degree) # select interaction term j with i
    dxdt = r * x * (1 - x/K) - c * x**2 / (x**2 + 1) - 4 * R * x + R * np.sum(A_interaction * x_j, -1)
    return dxdt

def harvest(x, t, A, c):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :degree: the degree of lattice  
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    dxdt = r * x * (1 - x/K) - c * x**2 / (x**2 + 1) - 4 * R * x + R * np.dot(A, x)
    return dxdt

def stable_state(A, degree):
    """calculate stables states for a given interaction matrix and dynamics-main.mutual

    :A: Adjacency matrix
    :degree: degree of lattice 
    :returns: stable states for all nodes x_l, x_h

    """
    t = np.arange(0, 5000, 0.01)
    N = np.size(A, -1)
    index = np.where(A!=0)
    A_interaction = A[index].reshape(N, degree)
    xs_low = odeint(harvest, np.ones(N) * 1, t, args=(A, c))[-1]
    xs_high = odeint(harvest, np.ones(N) * K, t, args=(A, c))[-1]
    return xs_low, xs_high

def network_ensemble_grid(N, num_col):
    G = nx.grid_graph(dim=[num_col,int(N/num_col)], periodic=True)
    A = np.array(nx.adjacency_matrix(G).todense())
    return A

def check_exist_index(des):
    """TODO: Docstring for check_exist_index.

    :des: TODO
    :returns: TODO

    """
    if not os.path.exists(des):
        os.makedirs(des)
    exist_index = 0
    des_file = des + 'realization' + str(exist_index) + '.h5'
    while os.path.exists(des_file):
        exist_index += 1
        des_file = des + 'realization' + str(exist_index) + '.h5'
    return exist_index

def system_collect(store_index, N, index, degree, A_interaction, strength, x_initial, T_start, T_end, t, dt,  des_evolution, des_ave, des_high, c):
    """one realization to run sdeint and save dynamics

    """
    local_state = np.random.RandomState(store_index + T_start ) # avoid same random process.
    noise= local_state.normal(0, np.sqrt(dt), (np.size(t)-1, N)) * strength
    dyn_all = main.sdesolver(main.close(harvest_lattice, *(N, index, degree, A_interaction, c)), x_initial, t, dW = noise)
    evolution_file = des_evolution + f'realization{store_index}_T_{T_start}_{T_end}'
    np.save(evolution_file, dyn_all)
    x_high = np.mean(dyn_all[-1])
    x_high_df = pd.DataFrame(np.ones((1, 1)) * x_high)
    x_high_df.to_csv(des_high + f'realization{store_index}.csv', mode='a', index=False, header=False)
    if x_high > K/2:

        ave_file = des_ave + f'realization{store_index}_T_{T_start}_{T_end}'
        np.save(ave_file, np.mean(dyn_all, -1))

    return None

def system_parallel(A, degree, strength, T_start, T_end, dt, parallel_index, cpu_number, des, c):
    """parallel computing or series computing 

    :A: adjacency matrix 
    :strength: std of noise
    :T: evolution time 
    :dt: simulation dt
    :parallel: the number of parallel realizations 
    :cpu_number: parallel computing with cpu_number if positive, and series computing if 0
    :des: destination to store the data
    :returns: None

    """
    parallel_size = np.size(parallel_index)
    N = np.size(A, -1)
    index = np.where(A!=0)
    A_interaction = A[index].reshape(N, degree)
    xs_low = odeint(harvest_lattice, np.ones(N) * 1, np.arange(0, 100, 0.01), args=(N, index, degree, A_interaction, c))[-1]
    t = np.linspace(T_start, T_end, int((T_end-T_start)/dt + 1))
    des_evolution = des + 'evolution/'
    des_ave = des + 'ave/'
    des_high = des + 'high/'
    for i in [des, des_evolution, des_ave, des_high]:
        if not os.path.exists(i):
            os.makedirs(i)

    if T_start == 0:
        x_start = np.broadcast_to(xs_low, (parallel_size, N))
    else:
        x_start = np.zeros((parallel_size, N))
        for realization, i in zip(parallel_index, range(parallel_size)):
            evolution_file = des_evolution + f'realization{realization}_T_{2*T_start-T_end}_{T_start}.npy'
            x_start[i] = np.load(evolution_file)[-1]
            os.remove(evolution_file)

    if cpu_number > 0:
        p = mp.Pool(cpu_number)
        p.starmap_async(system_collect, [(realization, N, index, degree, A_interaction, strength, x_start[i], T_start, T_end, t, dt, des_evolution, des_ave, des_high, c) for realization, i in zip(parallel_index, range(parallel_size))]).get()
        p.close()
        p.join()
    else:
        for i in parallel_index:
            system_collect(realization, N, index, degree, A_interaction, strength, x_start, T_start, T_end, t, dt, des_evolution, des_ave, des_high)
    return None

def generate_save_section(N, sigma, degree, T_start, T_end, c, parallel_index_initial, remove):
    """ generate data from 'realization_end+1', save and remove data, to get lifetime and rho

    :N_set: a set of N
    :sigma_set: a set of sigma
    :degree: degree
    :T: simulation time 
    :beta_fix: beta 
    :returns: None

    """

    des = '../data/harvest' + str(degree) + '/' + 'size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/'
    if T_start == 0:
        parallel_index = parallel_index_initial
    else:
        _, parallel_index = transition_index(des, parallel_index_initial)
    if len(parallel_index) != 0:

        num_col = int(np.sqrt(N))
        A = network_ensemble_grid(N, num_col)
        t1 =time.time()
        system_parallel(A, degree, sigma, T_start, T_end, dt, parallel_index, cpu_number, des, c)
        t2 =time.time()
        print('generate data:', N, sigma, T_start, T_end, parallel_index, t2 -t1)
        return 1
    else:
        if remove == 1:
            for _, _, filenames in os.walk(des + 'evolution/'):
                for filename in filenames:
                    os.remove(des + 'evolution/' + filename)
                break
        return 0


def  T_continue(N_set, sigma_set, T_start, T_end, T_every, parallel_index_initial, parallel_every, c, remove):
    """TODO: Docstring for T_continue.

    :T_start: TODO
    :T_end: TODO
    :T_every: TODO
    :returns: TODO

    """
    T_section = int((T_end - T_start) / T_every)
    parallel_section = int((parallel_index_initial[-1] + 1 - parallel_index_initial[0])/parallel_every )
    
    for N in N_set:
        for sigma in sigma_set:
            for j in range(parallel_section):
                parallel_index = parallel_index_initial[j*parallel_every: (j+1)*parallel_every]
                for i in range(T_section):
                    t_start = T_start + i * T_every
                    t_end = T_start + (i+1) * T_every
                    outcome = generate_save_section(N, sigma, degree, t_start, t_end, c, parallel_index, remove) 
                    if outcome == 0:
                        break

    return None

def transition_index(des, parallel_index_initial):
    """TODO: Docstring for transition_index.

    :des: TODO
    :returns: TODO

    """
    des_high = des + 'high/'
    succeed = []
    realization = parallel_index_initial[0]
    x_h_file = des_high + f'realization{realization}.csv'
    while os.path.exists(x_h_file) and realization <= parallel_index_initial[-1]:
        high = np.array(pd.read_csv(x_h_file, header=None).iloc[-1, :])
        if high > K/2:
            succeed.append(realization)
        realization += 1 
        x_h_file = des_high + f'realization{realization}.csv'

    parallel_index = np.setdiff1d(parallel_index_initial, succeed)
    return succeed, parallel_index

def cal_rho_lifetime(des, T_start, T_end, T_every, dt, parallel_index):
    """TODO: Docstring for cal_rho_lifetime.

    :des: TODO
    :returns: TODO

    """
    des_high = des + 'high/'
    des_ave = des + 'ave/'
    x_h = []
    succeed, _ = transition_index(des, parallel_index)
    parallel_size = np.size(parallel_index)
    for realization in succeed:

        x_h_file = des_high + f'realization{realization}.csv'
        high = np.array(pd.read_csv(x_h_file, header=None).iloc[-1, :])
        x_h.append(high)
    
    tau = np.ones(np.size(parallel_index)) * T_end
    if np.size(x_h) > 0:
        x_high = np.mean(x_h) 
    else:
        print('no transition')
        return None
    criteria = x_high / 2
    for realization in succeed:
        t_start = T_start 
        t_end = T_start + T_every 
        ave_file = des_ave + f'realization{realization}_T_{t_start}_{t_end}.npy'
        while not os.path.exists(ave_file):

            t_start += T_every
            t_end += T_every
            ave_file = des_ave + f'realization{realization}_T_{t_start}_{t_end}.npy'

        dyn_ave = np.load(ave_file)
        tau[realization] = dt * next(x for x, y in enumerate(dyn_ave) if y > criteria) + t_start

    tau_df = pd.DataFrame(tau.reshape((parallel_size), 1))
    tau_df.to_csv(des +  'lifetime.csv', index=False, header=False)
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
    des_file = des + f'evolution/realization{realization_index}_T_{0}_{plot_range}.npy'
    data = np.load(des_file)
    xmin = np.mean(data[0])
    if np.sum(data[-1] < K/2) == 0:
        xmax = np.mean(data[-1])
    elif np.sum(data[-1] > K/2 ) == 0:
        print('No transition')
        return None
    else:
        xmax = np.mean(data[-1, data[-1] > K/2])
    rho = (data - xmin) / (xmax - xmin)
    for i in np.arange(0, plot_range, plot_interval):
        data_snap = rho[int(i/dt)].reshape(int(np.sqrt(N)), int(np.sqrt(N)))
        fig = sns.heatmap(data_snap, vmin=0, vmax=1, linewidths=linewidth)
        fig = fig.get_figure()
        plt.title('time = ' + str(round(i, 2)) + 's')
        fig.savefig(des_sub + str(int(i/plot_interval)) + '.png')
        plt.close()
    return None

def bifurcation(c=np.arange(1, 3, 0.01), initial_x=np.arange(1,20,1), t=np.linspace(0, 1000, 100001), decimal = 5, error = 1e-10 ):
    """TODO: Docstring for bifurcation.

    :arg1: TODO
    :returns: TODO

    """
    c_num = np.size(c)
    stable = np.zeros((c_num, 2))
    unstable = np.zeros(c_num)
    for i in range(c_num):
        dyn_stable = odeint(harvest_1D, initial_x, t, args=(c[i],))[-1, :]
        stable_point = np.unique(dyn_stable.round(decimal))
        if np.size(stable_point) == 1:
            stable[i, :] = np.tile(stable_point, 2)
        elif np.size(stable_point) == 2:
            stable[i, :] = stable_point 
            fixed_point_set = []
            for j in range(np.size(initial_x)):
                fixed_point = fsolve(f, initial_x[j], args=(c[i],))
                if abs(f(fixed_point, c[i]))<error:
                    fixed_point_set = np.append(fixed_point_set, fixed_point)
            unstable_point = np.unique(np.setdiff1d(fixed_point_set.round(decimal), stable_point.round(decimal)).round(decimal))
            unstable_point_pos = unstable_point[unstable_point>0]
            if np.size(unstable_point_pos) == 1:
                unstable[i] = unstable_point_pos
            else:
                print(np.size(unstable_point_pos), i)
        else:
            print(np.size(unstable_point_pos), i)

    unstable_positive = unstable[unstable != 0]

    plt.semilogy(c, stable, 'k')
    plt.semilogy(c[unstable != 0], unstable_positive, '--')
    plt.xlabel('$c$', fontsize = fs)
    plt.ylabel('$x$', fontsize=fs)
    plt.subplots_adjust(left=0.18, wspace=0.25, hspace=0.25,
                    bottom=0.13, top=0.91)
    plt.title('Theoretical bifurcation ',fontsize=fs)
    plt.show()

def V_eff(c_set, x_set=np.arange(0,10, 0.01)):
    for c in c_set:
        V_set = []
        for x1 in x_set:
            V, error = sin.quad(f, 0, x1, args=(c,))
            V_set = np.append(V_set, -V)
        plt.plot(x_set[: :], V_set[: :], label='$\\c=$' + str(c))
        plt.xlabel('$x$', fontsize=fs)
        plt.ylabel('$V_{eff}$', fontsize=fs)
        plt.xticks(fontsize=15) 
        plt.yticks(fontsize=15) 
        plt.title('Effective potential $V_{eff}$ ', fontsize=20)
        plt.legend()
    plt.show()

def example():

    N = 100
    num_col = int(np.sqrt(N))
    x_initial = np.ones(N) * 1
    A = network_ensemble_grid(N, num_col)
    N = np.size(A, -1)
    index = np.where(A!=0)
    A_interaction = A[index].reshape(N, degree)
    local_state = np.random.RandomState(0 ) # avoid same random process.
    noise= local_state.normal(0, np.sqrt(dt), (np.size(t)-1, N)) * strength
    dyn = main.sdesolver(main.close(harvest_lattice, *(N, index, degree, A_interaction, c)), x_initial, t, dW = noise)

T_continue(N_set, sigma_set, T_start, T_end, T_every, parallel_index_initial, parallel_every, c, remove) 

parallel_index_initial = np.arange(1000)
des = '../data/harvest' + str(degree) + '/size' + str(N_set[0]) + '/c' + str(c) + '/strength=' + str(sigma_set[0]) + '/'
cal_rho_lifetime(des, T_start, T_end, T_every, dt, parallel_index_initial)
realization_index = 0
heatmap(des, realization_index, N_set[0], plot_range, plot_interval, dt)


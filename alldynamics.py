import main
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
import ast
import sympy as sp
from scipy.interpolate import interp1d
import shutil

degree = 4
cpu_number = 10
dt = 0.01
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
A1 = 0.1
A2 = 1
x1 = 1
x2 = 1.2
x3 = 5
remove = 0
del_evo = 0

def xj_xi(c, arguments, dxj):
    B, C, D, E, H, K = arguments
    xi, xj = sp.symbols('xi xj')
    f = B + xi * (1 - xi/K) * ( xi/C -1) + c * xi * xj / (D + E * xi + H* xj)
    dfi = sp.diff(f, xi).evalf(subs={xi: 0.136, xj: 0.136})
    dfj = sp.diff(f, xj).evalf(subs={xi: 0.136, xj: 0.136})
    dxi = - dfj/dfi * dxj    
    return dxi

def xj_from_xi(dynamics, degree, c, arguments, epsilon=0.001):
    B, C, D, E, H, K = arguments
    des = '../data/' + dynamics + str(degree) + '/' 
    bifurcation_file = des + 'bifurcation.csv'
    data = np.array(pd.read_csv(bifurcation_file, header=None).iloc[:, :])
    c_bifur = data[0]
    index = np.where(abs(c_bifur-c)<1e-10)[0]
    xl = data[1, index][0]
    xh = data[2, index][0]
    xu = data[3, index][0]
    x1 = np.arange(0, xh, 0.001)
    X = []
    Y = []
    f = lambda xj, xi: B + xj * (1 - xj/K) * ( xj/C -1) + c * xj * xi / (D + E * xj + H* xi)
    for xi in x1:
        '''
        xj = fsolve(f, xh, args=(xi))[0]
        if abs(f(xj, xi))< 1e-5:
            X.append(xi)
            Y.append(xj)
        '''

        xj = fsolve(f, xl, args=(xi))[0]
        if abs(f(xj, xi))< 1e-5 and xj>0:
            X.append(xi)
            Y.append(xj)
    xy_relation = pd.DataFrame(np.vstack((np.array(X), np.array(Y))))
    xy_relation.to_csv(des + f'/size1/c{c}/xy_relation.csv', mode='w', index=False, header=False)
    return X, Y 

def xy_interpolation(xi, c, arguments):
    B, C, D, E, H, K = arguments
    des = '../data/' + 'mutual' + str(4) + f'/size1/c{c}/' 
    xy_relation = np.array(pd.read_csv(des + 'xy_relation.csv', header=None).iloc[:, :])
    f = interp1d(xy_relation[0], xy_relation[1])
    if xi> xy_relation[1, -1]:
        xj = xy_relation[1, -1]
    else:
        xj= f(xi)
    return xj

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

def mutual_beta(x, t, c, arguments, f, xi1, xi2):
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
    # x_i, x_j, x_eff = x
    x_i = x
    if x_i < xi1:
        x_j = x_i 
    elif x_i > xi2:
        x_j = x_i 
    else:
        x_j = f(x_i)
    #x_j = 0.136 + xj_xi(c, arguments, x_i-0.136)
    dx_i = B + x_i * (1 - x_i/K) * (x_i/C - 1) + c * x_i * x_j / (D + E * x_i + H * x_j)
    # dx_j = B + x_j * (1 - x_j/K) * (x_j/C - 1) + x_j * x_i / ( D + E * x_j + H * x_i) + (c-1) * x_j * x_i /(D + E * x_j + H * x_i)
    # dx_eff = B + x_eff * (1 - x_eff/K) * ( x_eff/C - 1) + c * x_eff**2 / (D + (E+H) * x_eff)
    # return np.array([dx_i, dx_j, dx_eff])
    return dx_i

def mutual_ij(x, t, c, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :neighbor: the degree of each node 
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    B, C, D, E, H, K, R = arguments
    # x_i, x_j, x_eff = x
    x_i, x_j = x
    dx_i = B + x_i * (1 - x_i/K) * (x_i/C - 1) + c * x_i * x_j / (D + E * x_i + H * x_j)
    dx_j = B + x_j * (1 - x_j/K) * (x_j/C - 1) + 2 * x_j * x_i / ( D + E * x_j + H * x_i) + 2 * x_j * x_j /(D + E * x_j + H * x_j)
    # dx_eff = B + x_eff * (1 - x_eff/K) * ( x_eff/C - 1) + c * x_eff**2 / (D + (E+H) * x_eff)
    # return np.array([dx_i, dx_j, dx_eff])
    return np.array([dx_i, dx_j])

def mutual_lattice(x, t, N, index, degree, A_interaction, c, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :degree: the degree of lattice  
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    B, C, D, E, H, K, R= arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    x_tile = np.broadcast_to(x, (N,N))  # copy vector x to N rows
    x_j = x_tile[index].reshape(N, degree) # select interaction term j with i
    dxdt = B + x * (1 - x/K) * ( x/C - 1) + c/4 * x * np.sum(A_interaction * x_j / (D + E * x.reshape(N, 1) + H * x_j), -1)
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

def harvest_lattice(x, t, N, index, degree, A_interaction, c, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :degree: the degree of lattice  
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    r, K, R = arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    x_tile = np.broadcast_to(x, (N,N))  # copy vector x to N rows
    x_j = x_tile[index].reshape(N, degree) # select interaction term j with i
    dxdt = r * x * (1 - x/K) - c * x**2 / (x**2 + 1) - 4 * R * x + R * np.sum(A_interaction * x_j, -1)
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

def eutrophication_ij(x, t, c, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :c: bifurcation parameter 
    :returns: derivative of x 

    """
    a, r, R = arguments
    xi, xj = x 
    dxi = a - r * xi + c * xi**8 / (xi**8 + 1) + 4 * R * (xj - xi) 
    dxj = a - r * xj + c * xj**8 / (xj**8 + 1) + 4 * R * (xi - xj) 
    return np.array([dxi, dxj])

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

def vegetation_lattice(x, t, N, index, degree, A_interaction, c, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :degree: the degree of lattice  
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    r, rv, hv, R = arguments 
    x[np.where(x<0)] = 0  # Negative x is forbidden
    x_tile = np.broadcast_to(x, (N,N))  # copy vector x to N rows
    x_j = x_tile[index].reshape(N, degree) # select interaction term j with i
    dxdt = rv * x * (1 - x * (r**4 + (hv * c / (hv + x))**4)/r**4) - 4 * R * x + R * np.sum(A_interaction * x_j, -1)
    return dxdt

def harvestmay_1D(x, t, c, arguments):
    """TODO: Docstring for harvest_may.

    :x: TODO
    :t: TODO
    :gamma: TODO
    :alpha: TODO
    :returns: TODO

    """
    gamma = c
    alpha = arguments
    dxdt = x * (1 - x) - gamma * x**2 / (alpha**2 + x**2)
    return dxdt

def cubic_1D(x, t, c, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :c: bifurcation parameter 
    :returns: derivative of x 

    """
    x1, x2, x3 = arguments 
    dxdt = -c * (x - x1) * (x - x2) * (x - x3)
    return dxdt

def quadratic_1D(x, t, c, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :c: bifurcation parameter 
    :returns: derivative of x 

    """
    x2 = c
    A1, A2, x1, x3 = arguments 
    dxdt = A1 * (x - x1) * (x - x2) * np.heaviside(x2 - x, 0) - A2 * (x - x2) * (x - x3) * np.heaviside(x - x2, 0)
    return dxdt

def quadratic_lattice(x, t, N, index, degree, A_interaction, c, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :degree: the degree of lattice  
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    x2 = c
    A1, A2, x1, x3, R = arguments 
    x[np.where(x<0)] = 0  # Negative x is forbidden
    x_tile = np.broadcast_to(x, (N,N))  # copy vector x to N rows
    x_j = x_tile[index].reshape(N, degree) # select interaction term j with i
    dxdt = A1 * (x - x1) * (x - x2) * np.heaviside(x2 - x, 0) - A2 * (x - x2) * (x - x3) * np.heaviside(x - x2, 0) - 4 * R * x + R * np.sum(A_interaction * x_j, -1)
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
    xs_low = odeint(dynamics, np.ones(N) * low, t, args=(N, index, degree, A_interaction, c, arguments))[-1]
    xs_high = odeint(dynamics, np.ones(N) * high, t, args=(N, index, degree, A_interaction, c, arguments))[-1]
    return xs_low, xs_high

def bifurcation(c, dynamics, arguments, initial_x, t=np.linspace(0, 1000, 100001), decimal = 5, error = 1e-10 ):
    """TODO: Docstring for bifurcation.

    :arg1: TODO
    :returns: TODO

    """
    c_num = np.size(c)
    stable = np.zeros((c_num, 2))
    unstable = np.zeros(c_num)
    for i in range(c_num):
        dyn_stable = odeint(dynamics, initial_x, t, args=(c[i], arguments))[-1, :]
        stable_point = np.unique(dyn_stable.round(decimal))
        if np.size(stable_point) == 1:
            stable[i, :] = np.tile(stable_point, 2)
        elif np.size(stable_point) == 2:
            stable[i, :] = stable_point 
            fixed_point_set = []
            for j in range(np.size(initial_x)):
                fixed_point = fsolve(dynamics, initial_x[j], args=(0, c[i], arguments))
                if abs(dynamics(fixed_point, 0, c[i], arguments))<error:
                    fixed_point_set = np.append(fixed_point_set, fixed_point)
            unstable_point = np.unique(np.setdiff1d(fixed_point_set.round(decimal), stable_point.round(decimal)).round(decimal))
            unstable_point_pos = unstable_point[unstable_point>0]
            if np.size(unstable_point_pos) == 1:
                unstable[i] = unstable_point_pos
            else:
                print(np.size(unstable_point_pos), i)
        else:
            print(np.size(stable_point), i, stable_point)
    des = '../data/'+ dynamics.__name__[: dynamics.__name__.find('_')] + str(degree) + '/'
    bifurcation = pd.DataFrame(np.vstack((c, stable.transpose(), unstable)))
    bifurcation.to_csv(des + 'bifurcation.csv', mode='w', index=False, header=False)

    unstable_positive = unstable[unstable != 0]
    plt.plot(c, stable, 'k')
    plt.plot(c[unstable != 0], unstable_positive, '--')
    plt.xlabel('$c$', fontsize = fs)
    plt.ylabel('$x$', fontsize=fs)
    plt.subplots_adjust(left=0.18, wspace=0.25, hspace=0.25, bottom=0.13, top=0.91)
    plt.title('Bifurcation of ' + dynamics.__name__[: dynamics.__name__.find('_')],fontsize=fs)
    plt.show()

def network_ensemble_grid(N, num_col):
    G = nx.grid_graph(dim=[num_col,int(N/num_col)], periodic=True)
    A = np.array(nx.adjacency_matrix(G).todense())
    return A

def check_exist_index(des):
    """TODO: Docstring for check_exist_index.

    :des: TODO
    :returns: TODO

    """
    des_high = des+'high/'
    exist_index = 0
    if os.path.exists(des_high):
        des_file = des_high + 'realization' + str(exist_index) + '.csv'
        while os.path.exists(des_file):
            exist_index += 1
            des_file = des_high + 'realization' + str(exist_index) + '.csv'
    return exist_index

def check_exist_T(des):
    des_ave = des + 'evolution/'
    T = []
    for filename in os.listdir(des_ave):
        t = ast.literal_eval(filename[filename.rfind('_')+1:filename.find('.')])
        T.append(t)
    return np.max(T)

def system_collect(store_index, N, index, degree, A_interaction, strength, x_initial, T_start, T_end, t, dt, des_evolution, des_ave, des_high, dynamics, c, arguments, transition_to_high, criteria, remove, initial_noise):
    """one realization to run sdeint and save dynamics

    """
    evolution_file = des_evolution + f'realization{store_index}_T_{T_start}_{T_end}'
    # local_state = np.random.RandomState(store_index + T_start * int(parallel_size_all/T_every) ) # avoid same random process.
    local_state = np.random.RandomState(store_index)
    if initial_noise == 'metastable' and T_start == 0:
        pre_t = np.arange(0, 50, dt)
        dynamics_pre = globals()[dynamics.__name__[: dynamics.__name__.find('_')] + '_preprocess']
        noise= local_state.normal(0, np.sqrt(dt), (np.size(pre_t)-1, N)) * strength
        x_initial = main.sdesolver(main.close(dynamics_pre, *(N, index, degree, A_interaction, c, arguments)), x_initial, pre_t, dW = noise)[-1] 


    for i in range(int(T_start/T_every)+1):
        noise= local_state.normal(0, np.sqrt(dt), (np.size(t)-1, N)) * strength
    if N == 1:
        dynamics = globals()[dynamics.__name__[: dynamics.__name__.find('_')] + '_1D']
        dyn_all = main.sdesolver(main.close(dynamics, *(c, arguments[:-1])), x_initial, t, dW=noise.reshape(np.size(noise), 1) ) 
        '''
        des = '../data/' + 'mutual' + str(4) + f'/size1/c{c}/' 
        xy_relation = np.array(pd.read_csv(des + 'xy_relation.csv', header=None).iloc[:, :])
        f = interp1d(xy_relation[0], xy_relation[1])

        dynamics = globals()[dynamics.__name__[: dynamics.__name__.find('_')] + '_beta']
        dyn_all = main.sdesolver(main.close(dynamics, *(c, arguments[:-1], f, xy_relation[0, 0], xy_relation[0, -1] )), x_initial, t, dW=noise.reshape(np.size(noise), 1) ) 
        '''
        dyn_ave = np.mean(dyn_all, -1)
        x_high = np.mean(dyn_all[-1])

    elif N == 2:
        dynamics = globals()[dynamics.__name__[: dynamics.__name__.find('_')] + '_ij']
        noise[:, 1] = noise[:, 1] / 2 
        dyn_all = main.sdesolver(main.close(dynamics, *(c, arguments)), x_initial, t, dW=noise )
        x_high = dyn_all[-1, 0]
        dyn_ave = dyn_all[:, 0]
    else:
        dyn_all = main.sdesolver(main.close(dynamics, *(N, index, degree, A_interaction, c, arguments)), x_initial, t, dW = noise)
        dyn_ave = np.mean(dyn_all, -1)
        x_high = np.mean(dyn_all[-1])
    x_high_df = pd.DataFrame(np.ones((1, 1)) * x_high)
    x_high_df.to_csv(des_high + f'realization{store_index}.csv', mode='a', index=False, header=False)
    if (transition_to_high == 1 and x_high > criteria) or (transition_to_high == 0 and x_high < criteria):
        ave_file = des_ave + f'realization{store_index}_T_{T_start}_{T_end}'
        np.save(ave_file, dyn_ave)
        if remove == 0:
            np.save(evolution_file, dyn_all)
    else:
        np.save(evolution_file, dyn_all)
    del noise, dyn_all, x_high, dyn_ave
    return None

def system_parallel(A, degree, strength, T_start, T_end, dt, parallel_index, cpu_number, des, dynamics, x_initial, c, arguments, transition_to_high, criteria, remove, initial_noise):
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
    if N == 1 or N == 2:
        index = []
        A_interaction = []
    else:
        index = np.where(A!=0)
        A_interaction = A[index].reshape(N, degree)
    t = np.linspace(T_start, T_end, int((T_end-T_start)/dt + 1))
    des_evolution = des + 'evolution/'
    des_ave = des + 'ave/'
    des_high = des + 'high/'
    for i in [des, des_evolution, des_ave, des_high]:
        if not os.path.exists(i):
            os.makedirs(i)
    if T_start == 0:
        x_start = np.broadcast_to(x_initial, (parallel_size, N))
    else:
        x_start = np.zeros((parallel_size, N))
        for realization, i in zip(parallel_index, range(parallel_size)):
            evolution_file = des_evolution + f'realization{realization}_T_{2*T_start-T_end}_{T_start}.npy'
            x_start[i] = np.load(evolution_file)[-1]
            os.remove(evolution_file)
    if cpu_number > 0:
        p = mp.Pool(cpu_number)
        p.starmap_async(system_collect, [(realization, N, index, degree, A_interaction, strength, x_start[i], T_start, T_end, t, dt, des_evolution, des_ave, des_high, dynamics, c, arguments, transition_to_high, criteria, remove, initial_noise) for realization, i in zip(parallel_index, range(parallel_size))]).get()
        p.close()
        p.join()
        del p
    else:
        for i, i_index in zip(range(parallel_size), parallel_index):
            system_collect(i_index, N, index, degree, A_interaction, strength, x_start[i], T_start, T_end, t, dt, des_evolution, des_ave, des_high, dynamics, c, arguments, transition_to_high, criteria, remove, initial_noise)
    del x_start
    return None

def T_continue(N_set, sigma_set, T_start, T_end, T_every, parallel_index_initial, parallel_every, remove, del_evo, continue_evolution, dynamics, c, arguments, transition_to_high, low, high, one_transition, initial_noise):
    """TODO: Docstring for T_continue.

    :T_start: TODO
    :T_end: TODO
    :T_every: TODO
    :initial_noise: x_l is calculated without noise if 0; input argument if x_l is given
    :returns: TODO

    """
    if remove == 0:
        input_variable = input("Are you sure that you want to keep data? input 'yes' if you want to keep data, input 'no' to break.")
        if input_variable == 'yes':
            remove == 0
        elif input_variable == 'no':
            return None
        else:
            remove == 1
    if continue_evolution == 1:
        input_continue = input('do you want to continue evolution T?')
        if input_continue == 'yes':
            continue_evolution = 1
        elif input_continue == 'no':
            return None


    T_section = int((T_end - T_start) / T_every)
    parallel_section = int((parallel_index_initial[-1] + 1 - parallel_index_initial[0])/parallel_every )
    for N in N_set:
        "unweighted adjacency matrix"
        A = network_ensemble_grid(9, int(np.sqrt(9)))
        xs_low, xs_high = stable_state(A, degree, dynamics, c, low, high, arguments)
        xs_low_mean = np.mean(xs_low)
        xs_high_mean = np.mean(xs_high)
        xs_low = xs_low_mean * np.ones(N)
        xs_high = xs_high_mean * np.ones(N)
        A = network_ensemble_grid(N, int(np.sqrt(N)))
        if transition_to_high == 1:
            "change initial value according to the noise if initial state is given"
            if initial_noise == 0 or initial_noise == 'metastable':
                x_initial = xs_low
            elif type(initial_noise) == float :
                x_initial = initial_noise * np.ones(N)

            if one_transition == 1:
                x_initial[int(N/2+np.sqrt(N)/2)] = xs_high_mean
        elif transition_to_high == 0:
            x_initial = xs_high
            if one_transition == 1:
                x_initial[int(N/2+np.sqrt(N)/2)] = xs_low_mean
        criteria = (xs_low_mean + xs_high_mean) / 2

        for sigma in sigma_set:
            if dynamics == mutual_lattice:
                if initial_noise == 0:
                    des = '../data/' + dynamics.__name__[: dynamics.__name__.find('_')]+ str(degree) + '/' + 'size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/'
                elif type(initial_noise) == float:
                    des = '../data/' + dynamics.__name__[: dynamics.__name__.find('_')]+ str(degree) + '/' + 'size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_x_i' + str(initial_noise) + '/'
                elif initial_noise == 'metastable':
                    des = '../data/' + dynamics.__name__[: dynamics.__name__.find('_')]+ str(degree) + '/' + 'size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_' + initial_noise + '/'

            elif dynamics != quadratic_lattice and arguments[-1] != 0.2:
                if initial_noise == 0:
                    des = '../data/' + dynamics.__name__[: dynamics.__name__.find('_')]+ str(degree) + '/' + 'size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_R' + str(arguments[-1]) + '/'
                elif type(initial_noise) == float:
                    des = '../data/' + dynamics.__name__[: dynamics.__name__.find('_')]+ str(degree) + '/' + 'size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_R' + str(arguments[-1])+ '_x_i' + str(initial_noise) + '/'
                elif initial_noise == 'metastable':
                    des = '../data/' + dynamics.__name__[: dynamics.__name__.find('_')]+ str(degree) + '/' + 'size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_R' + str(arguments[-1])+ '_' + initial_noise + '/'
            elif dynamics != quadratic_lattice and arguments[-1] == 0.2:
                if initial_noise == 0:
                    des = '../data/' + dynamics.__name__[: dynamics.__name__.find('_')]+ str(degree) + '/' + 'size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/'
                elif type(initial_noise) == float:
                    des = '../data/' + dynamics.__name__[: dynamics.__name__.find('_')]+ str(degree) + '/' + 'size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_x_i' + str(initial_noise) + '/'
                elif initial_noise == 'metastable':
                    des = '../data/' + dynamics.__name__[: dynamics.__name__.find('_')]+ str(degree) + '/' + 'size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_' + initial_noise + '/'

            elif dynamics == quadratic_lattice:
                des = '../data/' + dynamics.__name__[: dynamics.__name__.find('_')]+ str(degree) + '/' + 'size' + str(N) + '/x2=' + str(c) + '/strength=' + str(sigma) + '/' + f'A1={arguments[0]}_A2={arguments[1]}_R={arguments[-1]}/'

            if not os.path.exists(des):
                os.makedirs(des)

            if parallel_index_initial[0] < check_exist_index(des):
                if continue_evolution == 1:
                    if check_exist_T(des) != T_start :
                        T_start = check_exist_T(des)
                        print(T_start)
                elif continue_evolution == 0:
                    print('simulation data already exists!')
                    break

            for j in range(parallel_section):
                parallel_index = parallel_index_initial[j*parallel_every: (j+1)*parallel_every]
                for i in range(T_section):
                    t_start = T_start + i * T_every
                    t_end = T_start + (i+1) * T_every
                    if t_start != 0:
                        _, parallel_index = transition_index(des, parallel_index, transition_to_high, criteria)
                    if len(parallel_index) != 0:
                        t1 =time.time()
                        system_parallel(A, degree, sigma, t_start, t_end, dt, parallel_index, cpu_number, des, dynamics, x_initial, c, arguments, transition_to_high, criteria, remove, initial_noise)
                        t2 =time.time()
                        print('generate data:', dynamics.__name__, N, sigma, arguments[-1], t_start, t_end, parallel_index, t2 -t1)
                    else:
                        break
                if del_evo == 1:
                    shutil.rmtree(des+ 'evolution/')
            cal_rho_lifetime(des, T_start, T_end, T_every, dt, transition_to_high, N, degree, dynamics, c, trial_low, trial_high, arguments)

    del A
    return None

def T_continue_mfpt(N_set, sigma_set, T_start, T_end, T_every, parallel_index_initial, parallel_every, remove, continue_evolution, dynamics, c, arguments, transition_to_high, low, high):
    """TODO: Docstring for T_continue.

    :T_start: TODO
    :T_end: TODO
    :T_every: TODO
    :returns: TODO

    """
    if remove == 0:
        input_variable = input("Are you sure that you want to keep data? input 'yes' if you want to keep data, input 'no' to break.")
        if input_variable == 'yes':
            remove == 0
        elif input_variable == 'no':
            return None
        else:
            remove == 1
    if continue_evolution == 1:
        input_continue = input('do you want to continue evolution T?')
        # input_continue = 'yes'
        if input_continue == 'yes':
            continue_evolution = 1
        elif input_continue == 'no':
            return None

    T_section = int((T_end - T_start) / T_every)
    parallel_section = int((parallel_index_initial[-1] + 1 - parallel_index_initial[0])/parallel_every )
    for N in N_set:
        "unweighted adjacency matrix"
        A = network_ensemble_grid(9, int(np.sqrt(9)))
        xs_low, xs_high = stable_state(A, degree, dynamics, c, low, high, arguments)
        xs_low_mean = np.mean(xs_low)
        xs_high_mean = np.mean(xs_high)
        xs_low = xs_low_mean * np.ones(N)
        xs_high = xs_high_mean * np.ones(N)
        A = network_ensemble_grid(N, int(np.sqrt(N)))
        if transition_to_high == 1:
            x_initial = xs_low
        elif transition_to_high == 0:
            x_initial = xs_high
        # criteria = (xs_low_mean + xs_high_mean) / 2
        "calculate MFPT, cross the barrier"
        criteria = -1

        for sigma in sigma_set:
            if dynamics == mutual_lattice:
                des = '../data/' + dynamics.__name__[: dynamics.__name__.find('_')]+ str(degree) + '/' + 'size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/'
            elif dynamics != quadratic_lattice and arguments[-1] != 0.2:
                des = '../data/' + dynamics.__name__[: dynamics.__name__.find('_')]+ str(degree) + '/' + 'size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '_R' + str(arguments[-1]) + '/'
            elif dynamics != quadratic_lattice and arguments[-1] == 0.2:
                des = '../data/' + dynamics.__name__[: dynamics.__name__.find('_')]+ str(degree) + '/' + 'size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/'
            elif dynamics == quadratic_lattice:
                des = '../data/' + dynamics.__name__[: dynamics.__name__.find('_')]+ str(degree) + '/' + 'size' + str(N) + '/x2=' + str(c) + '/strength=' + str(sigma) + '/' + f'A1={arguments[0]}_A2={arguments[1]}_R={arguments[-1]}/'

            if not os.path.exists(des):
                os.makedirs(des)

            if parallel_index_initial[0] < check_exist_index(des):
                if continue_evolution == 1:
                    if check_exist_T(des) != T_start :
                        T_start = check_exist_T(des)
                        print(T_start)
                elif continue_evolution == 0:
                    print('simulation data already exists!')
                    break

            for j in range(parallel_section):
                parallel_index = parallel_index_initial[j*parallel_every: (j+1)*parallel_every]
                for i in range(T_section):
                    t_start = T_start + i * T_every
                    t_end = T_start + (i+1) * T_every
                    if t_start != 0:
                        _, parallel_index = transition_index(des, parallel_index, transition_to_high, criteria)
                    if len(parallel_index) != 0:
                        t1 =time.time()
                        system_parallel(A, degree, sigma, t_start, t_end, dt, parallel_index, cpu_number, des, dynamics, x_initial, c, arguments, transition_to_high, criteria, remove)
                        t2 =time.time()
                        print('generate data:', dynamics.__name__, N, sigma, arguments[-1], t_start, t_end, parallel_index, t2 -t1)
                    else:
                        break
            # cal_rho_lifetime(des, T_start, T_end, T_every, dt, transition_to_high, N, degree, dynamics, c, trial_low, trial_high, arguments)
            cal_mfpt(des, dynamics.__name__[: dynamics.__name__.find('_')], degree, dt, transition_to_high)

    return None

def transition_index(des, parallel_index_initial, transition_to_high, criteria):
    """TODO: check untransitioned realization and transitioned index

    :des: TODO
    :returns: TODO

    """
    des_high = des + 'high/'
    succeed = []
    realization = parallel_index_initial[0]
    x_h_file = des_high + f'realization{realization}.csv'
    while os.path.exists(x_h_file) and realization <= parallel_index_initial[-1]:
        x_high = np.array(pd.read_csv(x_h_file, header=None).iloc[-1, :])

        if (transition_to_high == 1 and x_high > criteria) or (transition_to_high == 0 and x_high < criteria):
            succeed.append(realization)
        realization += 1 
        x_h_file = des_high + f'realization{realization}.csv'

    parallel_index = np.setdiff1d(parallel_index_initial, succeed)
    return succeed, parallel_index

def cal_rho_lifetime(des, T_start, T_end, T_every, dt, transition_to_high, N, degree, dynamics, c, low, high, arguments):
    """TODO: Docstring for cal_rho_lifetime.

    :des: TODO
    :returns: TODO

    """
    des_high = des + 'high/'
    des_ave = des + 'ave/'
    A = network_ensemble_grid(9, int(np.sqrt(9)))
    xs_low, xs_high = stable_state(A, degree, dynamics, c, low, high, arguments)
    xs_low_mean = np.mean(xs_low)
    xs_high_mean = np.mean(xs_high)
    criteria_fake = (xs_low_mean + xs_high_mean) / 2
    x_h = []
    total_realiztion_num = len(os.listdir(des_high))
    succeed, _ = transition_index(des, np.arange(total_realiztion_num), transition_to_high, criteria_fake)
    for realization in succeed:
        x_h_file = des_high + f'realization{realization}.csv'
        high = np.array(pd.read_csv(x_h_file, header=None).iloc[-1, :])
        x_h.append(high)
    
    tau = np.ones(total_realiztion_num) * T_end
    if np.size(x_h) > 0:
        x_high = np.mean(x_h) 
    else:
        print('no transition')
        return None

    for filename, realization in zip(os.listdir(des_ave), succeed):
        ave_file = des_ave + filename
        t_start = ast.literal_eval(ave_file[ave_file.find('T')+2 : ave_file.rfind('_')] )
        dyn_ave = np.load(ave_file)
        if transition_to_high == 1:
            criteria = (x_high + xs_low_mean) / 2
            criteria = (xs_high_mean + xs_low_mean) / 2
            tau[realization] = dt * next(x for x, y in enumerate(dyn_ave) if y > criteria) + t_start
        elif transition_to_high == 0:
            criteria = (x_high + xs_high_mean) / 2
            criteria = (xs_low_mean + xs_high_mean) / 2
            tau[realization] = dt * next(x for x, y in enumerate(dyn_ave) if y < criteria) + t_start
    tau_df = pd.DataFrame(tau.reshape((total_realiztion_num), 1))
    tau_df.to_csv(des +  'lifetime.csv', index=False, header=False)
    return None

def cal_mfpt(des, dynamics, degree, dt, transition_to_high):
    """TODO: Docstring for cal_rho_lifetime.

    :des: TODO
    :returns: TODO

    """
    des_high = des + 'high/'
    des_ave = des + 'ave/'
    total_realiztion_num = len(os.listdir(des_high))
    tau = np.ones(total_realiztion_num) * T_end
    bifurcation_file = '../data/' + dynamics + str(degree) + '/' + 'bifurcation.csv'
    data = np.array(pd.read_csv(bifurcation_file, header=None).iloc[:, :])
    c_bifur = data[0]
    unstable = data[-1]
    xu = unstable[abs(c_bifur-c)<1e-10]
    total_realiztion_num = len(os.listdir(des_high))
    succeed, _ = transition_index(des, np.arange(total_realiztion_num), transition_to_high, -1)

    for filename, realization in zip(os.listdir(des_ave), succeed):
        ave_file = des_ave + filename
        t_start = ast.literal_eval(ave_file[ave_file.find('T')+2 : ave_file.rfind('_')] )
        t_end = ast.literal_eval(ave_file[ave_file.rfind('_')+1 : ave_file.rfind('.')] )
        dyn_ave = np.load(ave_file)
        criteria = xu
        if transition_to_high == 1:
            if np.sum(dyn_ave > criteria)==0:
                tau[realization] = t_end
            else:
                tau[realization] = dt * next(x for x, y in enumerate(dyn_ave) if y > criteria) + t_start
        elif transition_to_high == 0:
            if np.sum(dyn_ave < criteria) == 0:
                tau[realization] = t_end
            else:
                tau[realization] = dt * next(x for x, y in enumerate(dyn_ave) if y < criteria) + t_start
    tau_df = pd.DataFrame(tau.reshape((total_realiztion_num), 1))
    tau_df.to_csv(des +  'mfpt.csv', index=False, header=False)
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
    average_file = os.listdir(des + 'ave/')
    file_num = np.size(average_file)
    if file_num > plot_num:
        average_file = average_file[: plot_num]
    average_plot = []
    for i in average_file:
        average = np.load(des + 'ave/' + i)
        average_plot = np.append(average_plot, average)
    average_plot = np.transpose(average_plot.reshape(plot_num, int(np.size(average_plot)/plot_num)))
    rho_plot = average_plot/ average_plot[-1]
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
    des_file = des + f'evolution/realization{realization_index}_T_{0}_{plot_range}.npy'
    data = np.load(des_file)
    xmin = np.mean(data[0])
    xmax = np.mean(data[-1])
    rho = (data - xmin) / (xmax - xmin)
    for i in np.arange(0, plot_range, plot_interval):
        data_snap = rho[int(i/dt)].reshape(int(np.sqrt(N)), int(np.sqrt(N)))
        fig = sns.heatmap(data_snap, vmin=0, vmax=1, linewidths=linewidth)
        fig = fig.get_figure()
        plt.title('time = ' + str(round(i, 2)) )
        fig.savefig(des_sub + str(int(i/plot_interval)) + '.png')
        plt.close()
    return None

def V_eff(dynamics, c_set, arguments, x_set=np.arange(0,10, 0.01)):
    for c in c_set:
        V_set = []
        for x1 in x_set:
            V, error = sin.quad(dynamics, x_set[0], x1, args=(0, c, arguments))
            V_set = np.append(V_set, -V) 
        plt.plot(x_set[: :], V_set[: :], label='$c=$' + str(round(c, 2)))
        plt.xlabel('$x$', fontsize=fs)
        plt.ylabel('$V_{eff}$', fontsize=fs)
        plt.xticks(fontsize=15) 
        plt.yticks(fontsize=15) 
        plt.title('Effective potential $V_{eff}$ ', fontsize=20)
        plt.legend()
        xsym = sp.symbols('x')
        df = sp.diff(dynamics(xsym, 0, c, arguments)) 
        Vmax = V_set.max()
        Vmin = V_set.min()
        xmax = x_set[np.where(Vmax == V_set)[0][0]]
        xmin = x_set[np.where(Vmin == V_set)[0][0]]
        dV = Vmax - Vmin
        dVmax = df.subs(xsym, xmax)
        dVmin = df.subs(xsym, xmin)
        print(dVmax, xmax, dVmin, xmin, (abs(dVmax * dVmin)))
        b = sp.log(2 * np.pi/sp.sqrt(abs(dVmax * dVmin)))

    plt.show()
    return V_set, dV, b

def mfpt_fit(dynamics, c, arguments):

    return dxi

def example(dynamics, c, arguments, N, sigma, low, high, transition_to_high):

    t = np.arange(0, 1000, 0.01)
    num_col = int(np.sqrt(N))
    A = network_ensemble_grid(9, 3)
    xs_l, xs_h = stable_state(A, degree, dynamics, c, low, high, arguments) 
    A = network_ensemble_grid(N, num_col)
    if transition_to_high == 1:
        x_initial = np.mean(xs_l) * np.ones(N) 
    else:
        x_initial = np.mean(xs_h) * np.ones(N)
    index = np.where(A!=0)
    A_interaction = A[index].reshape(N, degree)
    local_state = np.random.RandomState(5) # avoid same random process.
    noise= local_state.normal(0, np.sqrt(dt), (np.size(t)-1, N)) * sigma
    dyn = main.sdesolver(main.close(dynamics, *(N, index, degree, A_interaction, c, arguments)), x_initial, t, dW = noise)
    #plt.plot(t, np.mean(dyn, -1))
    plt.plot(t, dyn)
    plt.xlabel('t', fontsize=fs)
    plt.ylabel('x', fontsize=fs)
    plt.title(f'$c=${c}_$N=${N}_$\\sigma=${sigma}', fontsize=fs)
    plt.show()
    return index, dyn, xs_l, xs_h

def compare(dynamics, dynamics_ij, dynamics_beta, c, arguments, N, sigma, low, high, transition_to_high):

    t = np.arange(0, 100, 0.01)
    num_col = int(np.sqrt(N))
    A = network_ensemble_grid(9, 3)
    xs_l, xs_h = stable_state(A, degree, dynamics, c, low, high, arguments) 
    A = network_ensemble_grid(N, num_col)
    if transition_to_high == 1:
        x_initial = np.mean(xs_l) * np.ones(N) 
    else:
        x_initial = np.mean(xs_h) * np.ones(N)
    index = np.where(A!=0)
    A_interaction = A[index].reshape(N, degree)
    local_state = np.random.RandomState(1) # avoid same random process.
    noise= local_state.normal(0, np.sqrt(dt), (np.size(t)-1, N)) * sigma
    dyn = main.sdesolver(main.close(dynamics, *(N, index, degree, A_interaction, c, arguments)), x_initial, t, dW = noise)
    transition_first = np.where(dyn > (xs_l[0] + xs_h[0])/2)[1][0]
    # transition_first = 8
    first_neighbor = index[1][transition_first * c : (transition_first+1) * c]
    noise_beta = np.zeros((np.size(t)-1, 2))
    noise_beta[:, 0] = noise[:, transition_first] 
    noise_beta[:, 1] = np.mean(np.take(noise, first_neighbor, axis=1), -1)
    # noise_beta[:, 2] = np.mean(noise, -1)

    dyn_ij = main.sdesolver(main.close(dynamics_ij, *(c,arguments[:-1])), x_initial[:2], t, dW = noise_beta)

    '''
    des = '../data/' + 'mutual' + str(4) + f'/size1/c{c}/' 
    xy_relation = np.array(pd.read_csv(des + 'xy_relation.csv', header=None).iloc[:, :])
    f = interp1d(xy_relation[0], xy_relation[1])
    xj_stable = np.zeros((np.size(dyn_ij[:, 0])))
    plt.plot(t, xj_stable)
    for xi, i in zip(dyn_ij[:, 0], range(np.size(dyn_ij[:, 0]))):
        if xi < xy_relation[0, 0]:
            xj = xy_relation[1, 0]
        elif xi > xy_relation[0, -1]:
            xj = xy_relation[1, -1]
        else:
            xj = f(xi)
        xj_stable[i] = xj
    dyn_beta = main.sdesolver(main.close(dynamics_beta, *(c,arguments[:-1]), f, xy_relation[0, 0], xy_relation[0, -1]), x_initial[:1], t, dW = noise_beta[:, 0].reshape(np.size(noise_beta[:,0]), 1))
    plt.plot(t, dyn_beta, label='beta')
    '''
    plt.plot(t, dyn[:, transition_first], label='first_original')
    plt.plot(t, np.mean(np.take(dyn, first_neighbor, axis=1), -1), label='neighbor_original')
    plt.plot(t, dyn_ij[:, 0], label='first_reduction')
    plt.plot(t, dyn_ij[:, 1], label='neighbor_reduction')
    
    #plt.plot(t, np.mean(dyn, -1))
    plt.xlabel('t', fontsize=fs)
    plt.ylabel('x', fontsize=fs)
    # plt.title(f'$c=${c}_$N=${N}_$\\sigma=${sigma}', fontsize=fs)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.subplots_adjust(left=0.15, right=0.99, wspace=0.25, hspace=0.25, bottom=0.13, top=0.99)

    plt.legend(fontsize=14)
    plt.xlim(right=40)
    plt.show()
    return index, dyn, xs_l, xs_h

def tg_from_one_transition(dynamics, c, arguments, N, sigma, low, high, transition_to_high):
    """TODO: Docstring for tg_from_one_transition.

    :N: TODO
    :sigma: TODO
    :returns: TODO

    """
    t = np.arange(0, 100, 0.01)
    num_col = int(np.sqrt(N))
    A = network_ensemble_grid(9, 3)
    xs_l, xs_h = stable_state(A, degree, dynamics, c, low, high, arguments) 
    A = network_ensemble_grid(N, num_col)
    if transition_to_high == 1:
        x_initial = np.mean(xs_l) * np.ones(N) 
    else:
        x_initial = np.mean(xs_h) * np.ones(N)
    x_half = (np.mean(xs_l) + np.mean(xs_h))/2
    x_initial[int(num_col/2)] = np.mean(xs_h)
    index = np.where(A!=0)
    A_interaction = A[index].reshape(N, degree)
    local_state = np.random.RandomState(5) # avoid same random process.
    noise= np.random.normal(0, np.sqrt(dt), (np.size(t)-1, N)) * sigma
    dyn = main.sdesolver(main.close(dynamics, *(N, index, degree, A_interaction, c, arguments)), x_initial, t, dW = noise)
    rho = np.mean(dyn, -1)
    tg = next(x for x, y in enumerate(rho) if y > x_half) * 0.01
    return tg
    
def mutual_preprocess(x, t, N, index, degree, A_interaction, c, arguments):
    """TODO: Docstring for preprocess.

    :arg1: TODO
    :returns: TODO

    """
    B, C, D, E, H, K, R= arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    bound_forbid = 0.8
    bins = np.arange(-1, bound_forbid, 0.001)
    hist = np.histogram(x, bins)[0]
    hist_norm = hist/np.sum(hist)
    exceed_index = np.where(x>=bound_forbid)[0]
    exceed_neighbor = A_interaction[exceed_index]
    change_index = np.unique(np.hstack((exceed_index, np.ravel(exceed_neighbor))))
    x[change_index] = np.random.choice(bins[:-1], np.size(exceed_index), p=hist_norm)
    x_tile = np.broadcast_to(x, (N,N))  # copy vector x to N rows
    x_j = x_tile[index].reshape(N, degree) # select interaction term j with i
    dxdt = B + x * (1 - x/K) * ( x/C - 1) + c/4 * x * np.sum(A_interaction * x_j / (D + E * x.reshape(N, 1) + H * x_j), -1)
    return dxdt

def harvest_preprocess(x, t, N, index, degree, A_interaction, c, arguments):
    """TODO: Docstring for preprocess.

    :arg1: TODO
    :returns: TODO

    """
    r, K, R = arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    bound_forbid = 1.5
    bins = np.arange(-1, bound_forbid, 0.001)
    hist = np.histogram(x, bins)[0]
    hist_norm = hist/np.sum(hist)
    exceed_index = np.where(x>=bound_forbid)[0]
    exceed_neighbor = A_interaction[exceed_index]
    change_index = np.unique(np.hstack((exceed_index, np.ravel(exceed_neighbor))))
    x[change_index] = np.random.choice(bins[:-1], np.size(exceed_index), p=hist_norm)
    x_tile = np.broadcast_to(x, (N,N))  # copy vector x to N rows
    x_j = x_tile[index].reshape(N, degree) # select interaction term j with i
    dxdt = r * x * (1 - x/K) - c * x**2 / (x**2 + 1) - 4 * R * x + R * np.sum(A_interaction * x_j, -1)
    return dxdt

def eutrophication_preprocess(x, t, N, index, degree, A_interaction, c, arguments):
    """TODO: Docstring for preprocess.

    :arg1: TODO
    :returns: TODO

    """
    a, r, R = arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    bound_forbid = 0.8
    bins = np.arange(-1, bound_forbid, 0.001)
    hist = np.histogram(x, bins)[0]
    hist_norm = hist/np.sum(hist)
    exceed_index = np.where(x>=bound_forbid)[0]
    exceed_neighbor = A_interaction[exceed_index]
    change_index = np.unique(np.hstack((exceed_index, np.ravel(exceed_neighbor))))
    x[change_index] = np.random.choice(bins[:-1], np.size(exceed_index), p=hist_norm)
    x_tile = np.broadcast_to(x, (N,N))  # copy vector x to N rows
    x_j = x_tile[index].reshape(N, degree) # select interaction term j with i
    dxdt = a - r * x + c * x**8 / (x**8 + 1) - 4 * R * x + R * np.sum(A_interaction * x_j, -1)
    return dxdt

def vegetation_preprocess(x, t, N, index, degree, A_interaction, c, arguments):
    """TODO: Docstring for preprocess.

    :arg1: TODO
    :returns: TODO

    """
    r, rv, hv, R = arguments 
    x[np.where(x<0)] = 0  # Negative x is forbidden
    bound_forbid = 0.2
    bins = np.arange(-1, bound_forbid, 0.001)
    hist = np.histogram(x, bins)[0]
    hist_norm = hist/np.sum(hist)
    exceed_index = np.where(x>=bound_forbid)[0]
    exceed_neighbor = A_interaction[exceed_index]
    change_index = np.unique(np.hstack((exceed_index, np.ravel(exceed_neighbor))))
    x[change_index] = np.random.choice(bins[:-1], np.size(exceed_index), p=hist_norm)
    x_tile = np.broadcast_to(x, (N,N))  # copy vector x to N rows
    x_j = x_tile[index].reshape(N, degree) # select interaction term j with i
    dxdt = rv * x * (1 - x * (r**4 + (hv * c / (hv + x))**4)/r**4) - 4 * R * x + R * np.sum(A_interaction * x_j, -1)
    return dxdt


parallel_size_all = 1
dynamics_all_set = [mutual_lattice, harvest_lattice, eutrophication_lattice, vegetation_lattice, quadratic_lattice]
parallel_index_initial = np.arange(parallel_size_all)
trial_low = 0.1
trial_high = 10
dynamics_set = []
arguments_set = []
N_set = [9, 16, 25, 36, 49, 64, 81, 100, 900, 2500]
T_every = 100

continue_evolution = 0
parallel_every = 1
T_start = 0
T_end = 100
transition_to_high_set = [1]
one_transition = 0
initial_noise = 0
initial_noise = 'metastable'
index_set = [3]
c_set = [4]
c_set = [6]
c_set = [1.8]
c_set = [2.6]
sigma_set_all = [[0.009, 0.011, 0.012, 0.013, 0.014, 0.016, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.2, 0.3, 0.4, 0.5, 1, 5, 10, 0.007, 0.006 ]]
sigma_set_all = [[0.01]]
N_set = [900]
N_set = [2500]
N_set = [16, 25, 36, 49, 64, 81]
N_set = [10000]
N_set = [100]
R_set = [0.02]

arguments_all_set = [(B, C, D, E, H, K_mutual), (r, K), (a, r), (r, rv, hv), (A1, A2, x1, x3)]
for index in index_set:
    dynamics_set = dynamics_set + [dynamics_all_set[index]]
    arguments_set = arguments_set + [arguments_all_set[index]]
t1 = time.time()
for R in R_set:
    for dynamics, c, arguments, sigma_set, transition_to_high in zip(dynamics_set, c_set, arguments_set, sigma_set_all, transition_to_high_set):
        T_continue(N_set, sigma_set, T_start, T_end, T_every, parallel_index_initial, parallel_every, remove, del_evo, continue_evolution, dynamics, c, arguments + (R,), transition_to_high, trial_low, trial_high, one_transition, initial_noise)
t2 = time.time()
print(t2 -t1)
'''
for N in N_set:
    for sigma in sigma_set:
        des = f'../data/harvest{degree}/size{N}/c{c}/strength={sigma}/'
        cal_rho_lifetime(des, T_start, T_end, T_every, dt, transition_to_high, N, degree, dynamics, c, trial_low, trial_high, arguments)
heatmap(des, 0, N, 100, 1, 0.01)

"plot bifurcation"
c_bifurcation = [np.arange(-3,10,0.1), np.arange(1.6, 3,0.01), np.arange(0.1, 8, 0.01), np.arange(2, 4, 0.01)]
dynamics_bifurcation = [mutual_1D, harvest_1D, eutrophication_1D, vegetation_1D]
for c, dynamics, arguments in zip (c_bifurcation, dynamics_bifurcation, arguments_set):
    bifurcation(c, dynamics, arguments, np.arange(0.1, 20, 0.5))

'''
# bifurcation(np.arange(0.1, 0.3, 0.002), harvestmay_1D, (0.1), np.arange(0.01, 1, 0.001))




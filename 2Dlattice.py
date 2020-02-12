import main
import file_operation
import numpy as np 
from scipy.integrate import odeint 
import matplotlib.pyplot as plt
import time 
import os
import pandas as pd
import multiprocessing as mp

N_set = [100, 400, 900, 1600, 2500]
sigma_set = [0.2, 0.3, 0.4, 0.5]
sigma_set = [0.05, 0.06, 0.07]
N_set = [25, 100, 400]
parallel = 100
degree = 4
beta_fix = 4
cpu_number = 4
T_every = 2000
T_num = 1
T = T_every * T_num
dt = 0.01
repeat = 8

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

def system_collect(store_index, N, index, degree, A_interaction, strength, x_initial, t, dt, T_num,  des):
    """one realization to run sdeint and save dynamics

    """
    local_state = np.random.RandomState(store_index)
    noise= local_state.normal(0, np.sqrt(dt), (np.size(t)-1, N)) * strength
    dyn_all = main.sdesolver(main.close(main.mutual_lattice, *(N, index, degree, A_interaction)), x_initial, t, dW = noise)
    des_file = des + 'realization' + str(store_index) + '.h5'
    data = pd.DataFrame(dyn_all[1:])
    data.to_hdf(des_file, key='data', mode='a', append=True)

    return dyn_all[-1]

def system_parallel(A, degree, strength, T_num, T_every, dt, parallel, cpu_number, des, exist_index):
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
    N = np.size(A, -1)
    index = np.where(A!=0)
    A_interaction = A[index].reshape(N, degree)
    xs_low = odeint(main.mutual_lattice, np.ones(N) * 0, np.arange(0, 100, 0.01), args=(N, index, degree, A_interaction))[-1]
    for k in range(parallel):
        des_file = des + 'realization' + str(k+exist_index) + '.h5'
        data = pd.DataFrame(xs_low.reshape(1, N))
        data.to_hdf(des_file, key='data', mode='a', append=True)

    x_start = np.broadcast_to(xs_low, (parallel, N))
    for j in range(T_num):
        t = np.arange(j*T_every-np.heaviside(j, 0) * dt, (j+1)*T_every-1e-10, dt)
        if cpu_number > 0:
            p = mp.Pool(cpu_number)
            x_start = p.starmap_async(system_collect, [(i + exist_index, N, index, degree, A_interaction, strength, x_start[i], t, dt, des) for i in range(parallel)]).get()
            p.close()
            p.join()
        else:
            for i in range(parallel):
                system_collect(i + exist_index, N, index, degree, A_interaction, strength, x_start, t, dt, des)
    return None

def generate_save_data(N_set, sigma_set, degree, T, beta_fix):
    """ generate data from 'realization_end+1', save and remove data, to get lifetime and rho

    :N_set: a set of N
    :sigma_set: a set of sigma
    :degree: degree
    :T: simulation time 
    :beta_fix: beta 
    :returns: None

    """

    t = np.arange(0, T, dt)
    length = np.size(t)
    for N in N_set:
        for sigma in sigma_set:
            des = '../data/grid' + str(degree) + '/' + 'size' + str(N) + '/beta' + str(beta_fix) + '/strength=' + str(sigma) + '_T=' + str(T) + '/'
            if not os.path.exists(des):
                os.makedirs(des)

            num_col = int(np.sqrt(N))
            A = main.network_ensemble_grid(N, num_col, degree, beta_fix)
            t1 =time.time()
            realization_start, realization_end = file_operation.file_range(des) 
            main.system_parallel(A, degree, sigma, T, dt, parallel, cpu_number, des, realization_end + 1)
            t2 =time.time()
            print('generate data:', N, sigma, T, t2 -t1)

            t1 = time.time()
            realization_start, realization_end = file_operation.file_range(des)
            realization_range = [realization_start, realization_end]
            main.rho_lifetime_saving(realization_range, length, des, T, dt)
            for i in range(realization_start, realization_end):
                os.remove(des + f'realization{i}.h5')
            t2 = time.time()
            print('save data:', N, sigma, realization_range, t2 -t1)
    return None

for i in range(repeat):
    generate_save_data(N_set, sigma_set, degree, T, beta_fix)

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
sigma_set = [0.05]
N_set = [100]
parallel_index_initial = np.arange(100)  
degree = 4
beta_fix = 4
cpu_number = 4
T_start = 0
T_end = 200000
T_every = 100
dt = 0.01
repeat = 0



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

def system_collect(store_index, N, index, degree, A_interaction, strength, x_initial, T_start, T_end, t, dt,  des_evolution, des_ave, des_high):
    """one realization to run sdeint and save dynamics

    """
    local_state = np.random.RandomState(store_index + T_start * 100) # avoid same random process.
    noise= local_state.normal(0, np.sqrt(dt), (np.size(t)-1, N)) * strength
    dyn_all = main.sdesolver(main.close(main.mutual_lattice, *(N, index, degree, A_interaction)), x_initial, t, dW = noise)
    evolution_file = des_evolution + f'realization{store_index}_T_{T_start}_{T_end}'
    np.save(evolution_file, dyn_all)
    x_high = np.mean(dyn_all[-1])
    x_high_df = pd.DataFrame(np.ones((1, 1)) * x_high)
    x_high_df.to_csv(des_high + f'realization{store_index}.csv', mode='a', index=False, header=False)
    if x_high > main.K:

        ave_file = des_ave + f'realization{store_index}_T_{T_start}_{T_end}'
        np.save(ave_file, np.mean(dyn_all, -1))

    return None

def system_parallel(A, degree, strength, T_start, T_end, dt, parallel_index, cpu_number, des):
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
    xs_low = odeint(main.mutual_lattice, np.ones(N) * 0, np.arange(0, 100, 0.01), args=(N, index, degree, A_interaction))[-1]
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
        p.starmap_async(system_collect, [(realization, N, index, degree, A_interaction, strength, x_start[i], T_start, T_end, t, dt, des_evolution, des_ave, des_high) for realization, i in zip(parallel_index, range(parallel_size))]).get()
        p.close()
        p.join()
    else:
        for i in parallel_index:
            system_collect(realization, N, index, degree, A_interaction, strength, x_start, T_start, T_end, t, dt, des_evolution, des_ave, des_high)
    return None

def generate_save_delete_data(N_set, sigma_set, degree, T, beta_fix, remove_or_not = 1, strong_noise=0):
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
            main.system_parallel(A, degree, sigma, T, dt, parallel, cpu_number, realization_end + 1)
            t2 =time.time()
            print('generate data:', N, sigma, T, t2 -t1)

            t1 = time.time()
            realization_start, realization_end = file_operation.file_range(des)
            realization_range = [realization_start, realization_end]
            main.rho_lifetime_saving(realization_range, length, des, T, dt, strong_noise)
            t2 = time.time()
            print('save data:', N, sigma, realization_range, t2 -t1)

            if remove_or_not == 1:

                for i in range(realization_start, realization_end):
                    os.remove(des + f'realization{i}.h5')
    return None

def generate_save_section(N, sigma, degree, T_start, T_end, beta_fix, parallel_index_initial):
    """ generate data from 'realization_end+1', save and remove data, to get lifetime and rho

    :N_set: a set of N
    :sigma_set: a set of sigma
    :degree: degree
    :T: simulation time 
    :beta_fix: beta 
    :returns: None

    """

    des = '../data/grid' + str(degree) + '/' + 'size' + str(N) + '/beta' + str(beta_fix) + '/strength=' + str(sigma) + '/'
    if T_start == 0:
        parallel_index = parallel_index_initial
    else:
        _, parallel_index = transition_index(des, parallel_index_initial)
    if len(parallel_index) != 0:

        num_col = int(np.sqrt(N))
        A = main.network_ensemble_grid(N, num_col, degree, beta_fix)
        t1 =time.time()
        system_parallel(A, degree, sigma, T_start, T_end, dt, parallel_index, cpu_number, des)
        t2 =time.time()
        print('generate data:', N, sigma, T_start, T_end, parallel_index, t2 -t1)
        return 1
    else:
        return 0


def  T_continue(N_set, sigma_set, T_start, T_end, T_every, parallel_index_initial):
    """TODO: Docstring for T_continue.

    :T_start: TODO
    :T_end: TODO
    :T_every: TODO
    :returns: TODO

    """

    section = int((T_end - T_start) / T_every)
    for N in N_set:
        for sigma in sigma_set:
            for i in range(section):
                t_start = T_start + i * T_every
                t_end = T_start + (i+1) * T_every
                outcome = generate_save_section(N, sigma, degree, t_start, t_end, beta_fix, parallel_index_initial) 
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
        if high > main.K:
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
    tau_df.to_csv(des +  'lifetime.csv', mode='a', index=False, header=False)
    return None

    

for i in range(repeat):
    generate_save_delete_data(N_set, sigma_set, degree, T, beta_fix, 1, 1)

T_continue(N_set, sigma_set, T_start, T_end, T_every, parallel_index_initial)

des = '../data/grid' + str(degree) + '/' + 'size' + str(N_set[0]) + '/beta' + str(beta_fix) + '/strength=' + str(sigma_set[0]) + '/'
# cal_rho_lifetime(des, T_start, T_end, T_every, dt, parallel_index_initial)


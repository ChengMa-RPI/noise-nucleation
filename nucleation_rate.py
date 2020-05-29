import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os 
import time 
import multiprocessing as mp

def convert_index(index, N):
    """Convert index into row index and column index

    :index: TODO
    :returns: TODO

    """
    M = int(np.sqrt(N))
    index_row = np.floor(index/M)
    index_column = index % M 
    return index_row, index_column
    
def connection_condition(index1, index2, N):
    """TODO: Docstring for connection.

    :index1: TODO
    :index2: TODO
    :returns: TODO

    """
    row1, column1 = convert_index(index1, N)
    row2, column2 = convert_index(index2, N)
    row1 = row1.reshape(np.size(row1), 1)
    column1 = column1.reshape(np.size(column1), 1)
    condition1 = (row1 - row2) * (column1 - column2)
    condition2 = abs(row1 - row2) + abs(column1 - column2)
    M = int(np.sqrt(N))
    select = np.where((condition1 == 0) &((condition2-1)%(M-2) == 0 ))
    return select

def cluster_division(cluster, index_before, index_after, index_add, N):
    """TODO: Docstring for connect_condition.

    :arg1: TODO
    :returns: TODO

    """
    if np.size(index_before) == 0:
        cluster.append([index_add[0]])
        index_rest = np.setdiff1d(index_add, index_add[0])
        index_add =index_rest

    index_rest = np.array(index_add)
    while np.size(index_rest):
        indicator = 0
        len_set = [len(cluster[i]) for  i in range(len(cluster))]
        len_cum = np.cumsum(len_set)
        cluster_array = np.hstack(cluster)
        select = connection_condition(index_rest, cluster_array, N)
        if np.size(select):
            select = np.vstack(select)  # change select to array type
            unique_index = [select[0].tolist().index(x) for x in set(select[0])]  # make the chosen index appear only once.  
            select = select[:, unique_index]
            for select_append, index_flat in zip(index_rest[select[0]], select[1]):
                index = np.where([len_cum > index_flat])[0][0]
                cluster[index].append(select_append)
            index_rest = np.setdiff1d(index_rest, index_rest[select[0]])
            indicator = 1
        else:
            cluster.append([index_rest[0]])
            index_rest = np.setdiff1d(index_rest, index_rest[0])

    return cluster

def nucleation(dynamics, degree, c, N, sigma, realization, interval, initial_noise, bound, T_start, T_end, dt):
    """TODO: Docstring for nucleation.

    :dynamics: TODO
    :realization: TODO
    :returns: TODO

    """
    t = np.arange(T_start, T_end, dt*interval)
    t_num = np.size(t)
    number_l_set = np.zeros((t_num))
    cluster_set = np.zeros((t_num))
    low_ave = np.ones((t_num))
    des =  '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma)  
    if initial_noise == 0:
        des_evo = des + '/evolution/'
    else:
        des_evo = des +  '_x_i' + str(initial_noise) + '/evolution/'
    evolution_file = des_evo + f'realization{realization}_T_{T_start}_{T_end}.npy'
    evolution = np.load(evolution_file)
    evolution_interval = evolution[::interval]
    x_ave = np.mean(evolution_interval, 1)
    x_l = x_ave[0]
    x_h = x_ave[-1]
    "the value of x_h should be changed according to the dynamics"
    if x_h < 1:
        x_h = 6.964
    rho = (evolution_interval - x_l)/(x_h-x_l)  # global state
    index_before = []
    cluster = []
    for i in range(t_num):
        h_index = np.where(rho[i] > bound)[0]
        number_l_set[i] = N - np.size(h_index)
        if number_l_set[i] > 0:
            if i == 0:
                low_ave[i] = np.mean(rho[i][rho[i]<bound])
            else:
                low_ave[i] = np.mean(rho[i-1][rho[i]<bound])
        if np.size(h_index)>0 and np.size(h_index)<N:
            index_after = h_index
            index_add = np.setdiff1d(index_after, index_before)
            cluster = cluster_division(cluster, index_before, index_after, index_add, N)
            index_before = index_after
            cluster_set[i] = len(cluster)
        elif np.size(h_index) == N:
            break
    number_nucleation = np.hstack((cluster_set[0], np.diff(cluster_set)))
    return cluster_set, number_l_set, number_nucleation, low_ave

def nucleation_parallel(dynamics, degree, c, N, sigma, parallel_index_initial, parallel_every, interval, initial_noise, bound, T_start=0, T_end=100, dt=0.01):
    """TODO: Docstring for nucleation_parallel.

    :arg1: TODO
    :returns: TODO

    """
    t = np.arange(T_start, T_end, dt*interval)
    parallel_section = int((parallel_index_initial[-1] + 1 - parallel_index_initial[0])/parallel_every )
    result = []
    for j in range(parallel_section):
        p = mp.Pool(cpu_number)
        t1 = time.time()
        parallel_index = parallel_index_initial[j*parallel_every: (j+1)*parallel_every]
        parallel_size = np.size(parallel_index)
        result_temp = p.starmap_async(nucleation, [(dynamics, degree, c, N, sigma, realization, interval, bound, T_start, T_end, dt) for realization, i in zip(parallel_index, range(parallel_size))]).get()
        result.extend(result_temp)
        t2 = time.time()
        p.close()
        p.join()
        del result_temp, p
    return t, result



dynamics = 'mutual'
degree = 4
c = 4 
N = 10000
sigma = 0.1
initial_noise = 0.1707
parallel_index_initial = [0]
parallel_index_initial = np.arange(1000) + 1000
parallel_every = 1
interval = 100
bound = 0.1
cpu_number = 38

t, result = nucleation_parallel(dynamics, degree, c, N, sigma, parallel_index_initial, parallel_every, interval, initial_noise, bound)

t_num = np.size(t)
realization_num = np.size(parallel_index_initial)
number_l_set = np.zeros((realization_num, t_num))
cluster_set = np.zeros((realization_num, t_num))
nucleation_set = np.zeros((realization_num, t_num))
low_ave_set = np.zeros((realization_num, t_num))

for i in range(realization_num):
    cluster_set[i], number_l_set[i], nucleation_set[i], low_ave_set[i] = result[i]

data = np.vstack((t, cluster_set, number_l_set, nucleation_set, low_ave_set))

data_df = pd.DataFrame(data)
data_df.to_csv('../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/nucleation.csv', index=False, header=False)

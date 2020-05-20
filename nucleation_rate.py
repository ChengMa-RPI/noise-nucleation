import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os 


    

def convert_index(index, N):
    """Convert index into row index and column index

    :index: TODO
    :returns: TODO

    """
    M = int(np.sqrt(N))
    index_row = np.floor(index/M)
    index_column = index % M 
    index_rc = np.vstack((index_row, index_column, index))
    return index_rc
    
def connection_condition(index1, index2):
    """TODO: Docstring for connection.

    :index1: TODO
    :index2: TODO
    :returns: TODO

    """
    index1 = np.array(index1)
    index2 = np.array(index2)
    row1, column1, total1 = convert_index(index1, N)
    row2, column2, total2 = convert_index(index2, N)
    row1 = row1.reshape(np.size(row1), 1)
    column1 = column1.reshape(np.size(column1), 1)
    condition1 = (row1 - row2) * (column1 - column2)
    condition2 = abs(row1 - row2) + abs(column1 - column2)
    if np.sum(condition1 == 0) and (np.sum(condition2 == 1) + np.sum(condition2 == N-1)):
        connection = 1
        select = np.where((condition1 == 0) &((condition2-1)%(N-2) == 0 ))[0]
    else:
        connection = 0
        select = []
    return connection, select

def cluster_division(cluster, index_before, index_after, index_add):
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
        for j in cluster:
            connection, select = connection_condition(index_rest, j)
            if connection:
                for select_append in index_rest[select]:
                    j.append(select_append)
                index_rest = np.setdiff1d(index_rest, index_rest[select])
                indicator = 1
        if indicator == 0:
            cluster.append([index_rest[0]])
            index_rest = np.setdiff1d(index_rest, index_rest[0])
    return cluster


def nucleation(dynamics, degree, c, N, sigma, realization, interval, bound=1/2, T_start=0, T_end=100, dt=0.01):
    """TODO: Docstring for nucleation.

    :dynamics: TODO
    :realization: TODO
    :returns: TODO

    """
    t = np.arange(T_start, T_end, dt*interval)
    t_num = np.size(t)
    realization_num = np.size(realization)
    number_l_set = np.zeros((realization_num, t_num))
    cluster_set = np.zeros((realization_num, t_num))
    nucleation_set = np.zeros((realization_num, t_num))
    des_evo = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/evolution/'
    for j in realization:
        evolution_file = des_evo + f'realization{j}_T_{T_start}_{T_end}.npy'
        evolution = np.load(evolution_file)
        evolution_interval = evolution[::interval]
        x_ave = np.mean(evolution_interval, 1)
        x_l = x_ave[0]
        x_h = x_ave[-1]
        rho = (evolution_interval - x_l)/(x_h-x_l)  # global state
        index_before = []
        cluster = []

        for i in range(t_num):
            h_index = np.where(rho[i] > bound)[0]
            number_l_set[j, i] = N - np.size(h_index)
            if np.size(h_index)>0 and np.size(h_index)<N:
                index_after = h_index
                index_add = np.setdiff1d(index_after, index_before)
                cluster = cluster_division(cluster, index_before, index_after, index_add)
                index_before = index_after
                cluster_set[j, i] = len(cluster)
            elif np.size(h_index) == N:
                break
        number_nucleation = np.hstack((cluster_set[j, 0], np.diff(cluster_set[j])))
    return t, cluster_set, number_l_set, number_nucleation

dynamics = 'mutual'
degree = 4
c = 4 
N = 10000
sigma = 0.1
realization = [0]
interval = 100

t, cluster_set, number_l_set, number_nucleation = nucleation(dynamics, degree, c, N, sigma, realization, interval)

import main
import numpy as np 
from scipy.integrate import odeint 
import matplotlib.pyplot as plt
import time 
import os
B = 0.1
C = 1
K = 5
D = 5 
E = 0.9
H = 0.1

degree = 4
beta_fix = 4
T = 200
dt = 0.01
def stable_state(A):
    """calculate stables states for a given interaction matrix and dynamics-main.mutual

    :A: Adjacency matrix
    :t: time 
    :returns: stable states

    """
    t = np.arange(0, 100, 0.01)
    index = np.where(A!=0)
    neighbor = np.sum(A!=0, -1)
    A_interaction = A[index]

    xs_low = odeint(main.mutual, np.ones(N) * 0, t, args=(N, index, neighbor, A_interaction))
    xs_high = odeint(main.mutual, np.ones(N) * 5, t, args=(N, index, neighbor, A_interaction))
    return xs_low, xs_high

def high_index(N, index_per_row):
    """TODO: Docstring for high_index.
    :returns: TODO

    """
    index = []
    row_num = list(index_per_row.keys())
    row_value = list(index_per_row.values())
    for i in range(len(row_num)):
        index.extend((row_num[i] * int(np.sqrt(N)) + np.array(row_value[i])))
    return index 

reduction = 0.1
N = 100
num_col = 10
A = main.network_ensemble_grid(N, num_col, degree, beta_fix) * reduction
index = np.where(A!=0)
neighbor = np.sum(A!=0, -1)
A_interaction = A[index]

t = np.arange(0, T, dt)
xs_low, xs_high = stable_state(A, t)
x = xs_low[-1]
high_initial = {1: [3], 2:[2,4], 3:[3]}
high_convert = high_index(N, high_initial)
x[high_convert] = xs_high[-1, high_convert]
dyn = odeint(main.mutual, x, t, args=(N, index, neighbor, A_interaction))
high_final = np.where(dyn[-1] > main.K)[0]
A_high = A[high_final, :][:, high_final]
A_high_gcc, A_high_gcc_index = main.Gcc_A_mat(A_high)
A_high_gcc_node = high_final[A_high_gcc_index]

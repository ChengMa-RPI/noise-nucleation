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
import main 
# dynamics parameter
B = 0.1
C = 1
K = 5
D = 5 
E = 0.9
H = 0.1

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

strength = 0.1
degree = 4
beta_fix = 4
T = 20
dt = 0.01
N = 3600
num_col = int(np.sqrt(N))
A = main.network_ensemble_grid(N, num_col, degree, beta_fix) 
index = np.where(A!=0)
neighbor = np.sum(A!=0, -1)
A_interaction = A[index]

t = np.arange(0, T, dt)
x = np.ones(N)
x_initial = np.ones(N)
local_state = np.random.RandomState(1)
noise= local_state.normal(0, np.sqrt(dt), (np.size(t)-1, N)) * strength
index_lattice = np.where(A!=0)
A_interaction_lattice = A[index_lattice].reshape(N, degree)
t1 = time.time()
dyn = odeint(mutual, x, t, args=(N, index, neighbor, A_interaction))
t2 =time.time()
dyn_lattice = odeint(mutual_lattice, x, t, args=(N, index_lattice, degree, A_interaction_lattice))
t3 =time.time()
dyn_all = sdeint.itoEuler(close(mutual, *(N, index, neighbor, A_interaction)), close(eta_diag, *(N, )), x_initial, t, dW = noise)
t4 = time.time()
dyn_all_lattice = sdeint.itoEuler(close(mutual_lattice, *(N, index_lattice, degree, A_interaction_lattice)), close(eta_diag, *(N, )), x_initial, t, dW = noise)
t5 = time.time()
dyn_sde = sdesolver(close(mutual, *(N, index, neighbor, A_interaction)), x_initial, t, dW = noise) 
t6 = time.time()
dyn_sde_lattice = sdesolver(close(mutual_lattice, *(N, index_lattice, degree, A_interaction_lattice)), x_initial, t, dW = noise) 
t7 = time.time()
print(t2 -t1, t3 -t2, t4 -t3, t5 -t4, t6 -t5, t7 -t6)

"""
Conclusion: mutual lattice is a little bit faster than mutual, about 1s. Not necessary to make changes. 
"""

import main
import file_operation
import fnmatch 
import os
import numpy as np 
import time
import pandas as pd 
import ast


degree =4
dt = 0.01
N_set = [25, 100, 400, 2500, 6400, 10000]
sigma_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
sigma_set = [0.08]
N_set = [100]
T_set = [100]
des_all = file_operation.all_dir(degree)
for des in des_all:
    N, beta_fix, sigma, T = file_operation.extract_info(des)
    if N in N_set and sigma in sigma_set and beta_fix == 4 and T in T_set:
        # des_evolution = des + 'evolution/'
        t = np.arange(0, T, dt)
        length = np.size(t)
        realization_start, realization_end = file_operation.file_range(des)
        realization_range = [realization_start, realization_end]
        t1 = time.time()
        main.rho_lifetime_saving(realization_range, length, des, T, dt)
        for i in range(realization_start, realization_end):
            os.remove(des + f'realization{i}.h5')
        t2 =time.time()
        print(N, beta_fix, sigma, T, realization_range, t2 -t1)



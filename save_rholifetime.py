import main
import file_operation
import fnmatch 
import os
import numpy as np 
import time
import pandas as pd 


degree =4
dt = 0.01
N_set = [25, 100, 400, 2500, 6400, 10000]
sigma_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
des_all = file_operation.all_dir(degree)
for des in des_all:
    N, beta_fix, sigma, T = file_operation.extract_info(des)
    if N in N_set and sigma in sigma_set:
        t = np.arange(0, T, dt)
        length = np.size(t)
        realization = len(fnmatch.filter(os.listdir(des), '*.h5'))
        t1 = time.time()
        main.rho_lifetime(realization, length, des, T, dt)
        t2 =time.time()
        print(N, beta_fix, sigma, T, realization, t2 -t1)

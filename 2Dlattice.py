import main
import numpy as np 
from scipy.integrate import odeint 
import matplotlib.pyplot as plt
import time 
import os
import pandas as pd

N_set = [25]
strength = 0.1
parallel = 100
degree = 4
beta_fix = 4
cpu_number = 4
T = 100
dt = 0.01

for N in N_set:
    des = '../data/grid' + str(degree) + '/' + 'size' + str(N) + '/beta' + str(beta_fix) + '/strength=' + str(strength) + '_T=' + str(T) + '/'
    if not os.path.exists(des):
        os.makedirs(des)

    num_col = int(np.sqrt(N))
    A = main.network_ensemble_grid(N, num_col, degree, beta_fix)
    t1 =time.time()
    main.system_parallel(A, degree, strength, T ,dt, parallel, cpu_number, des)
    t2 = time.time()
    print(N, t2 -t1)



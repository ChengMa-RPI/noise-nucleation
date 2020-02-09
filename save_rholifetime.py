import main
import file_operation
import fnmatch 
import os
import numpy as np 
import time


degree =4
dt = 0.01

des_all = file_operation.all_dir(degree)
for des in des_all:
    N, beta_fix, sigma, T = file_operation.extract_info(des)
    t = np.arange(0, T, dt)
    length = np.size(t)
    realization = len(fnmatch.filter(os.listdir(des), '*.h5'))
    t1 = time.time()
    main.rho_lifetime(realization, length, des, T, dt)
    t2 =time.time()
    print(N, beta_fix, sigma, T, realization, t2 -t1)

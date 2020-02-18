import main
import file_operation
import fnmatch 
import os
import numpy as np 
import time
import pandas as pd 
import ast
def rho_lifetime_saving(realization_range, length, des, T, dt, strong_noise):
    """for given range of realization files [realization_start +1 to realization_end], find rho which is normalized average evolution, and lifetime which is the time when rho exceeds 1/2, and also save x_h data.

    :realization_range: read files from realization_range[0] + 1 to realization_range[1] 
    :length: length of time
    :des: the destination where data is saved
    :x_l, x_h: lower and higher stable states
    :returns: None

    """
    realization_num = realization_range[1] - realization_range[0] 
    x = np.zeros((realization_num, length))
    y = []  # add data that has transitioned 
    tau = np.ones(realization_num) * T 
    for i in range(realization_num):
        des_file = des + 'realization' + str(i+realization_range[0]+1) + '.h5'
        data = np.array(pd.read_hdf(des_file))
        x[i] = np.mean(data, -1)
        if strong_noise == 0:
            if np.sum(data[-1] < K) == 0:
                y.append(x[i, -1])
        else:
            y.append(x[i, -1])
    x_l = np.mean(x[:, 0])
    if np.size(y) != 0:
        x_h = np.mean(y)
        rho = (x - x_l) / (x_h - x_l)
        rho_last = rho[:, -1]
        succeed = np.where(rho_last > 1/2)[0]
        x_h_file = des + 'x_h.csv'
        if os.path.exists(x_h_file):
            x_h_old = np.array(pd.read_csv(x_h_file, header=None).iloc[0, 0])
            x_h = np.mean([x_h_old, x_h])
        pd.DataFrame(np.ones((1,1)) * x_h).to_csv(x_h_file, index=False, header=False)
        for i in succeed:
            tau[i] = dt * next(x for x, y in enumerate(rho[i]) if y > 1/2)
        rho_df = pd.DataFrame(rho)
        rho_df.to_csv(des +  'rho.csv', mode='a', index=False, header=False)
    tau_df = pd.DataFrame(tau.reshape(realization_num, 1))
    tau_df.to_csv(des +  'lifetime.csv', mode='a', index=False, header=False)
    return None


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



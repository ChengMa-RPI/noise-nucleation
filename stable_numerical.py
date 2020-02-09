import main
import numpy as np
import pandas as pd
from scipy.integrate import odeint 

degree = 4
N = 100
beta_fix = 4
strength = 0.1
T = 100
des = 'data/grid' + str(degree) + '/size' + str(N) + '/beta' + str(beta_fix) + '/strength=' + str(strength) + '_T=' + str(T) + '/'

realization = 100
x_last = []

for i in range(realization):
    des_file = des + f'realization{i}.h5'
    data = np.array(pd.read_hdf(des_file))
    x_last.append(np.mean(data[-1]))

np.mean(x_last)

T = 100
dt = 0.01
N = 10000
num_col = int(np.sqrt(N))
A = main.network_ensemble_grid(N, num_col, degree, beta_fix)
index = np.where(A!=0)
t = np.arange(0, T ,dt)
A_interaction = A[index].reshape(N, degree)
xs_low = odeint(main.mutual_lattice, np.ones(N) * 0, np.arange(0, 100, 0.01), args=(N, index, degree, A_interaction))


import main
import numpy as np 
import matplotlib.pyplot as plt

degree =4
N_set = [9, 16, 25, 36, 49, 64, 81, 100, 400, 900, 1600, 2500, 3600, 4900, 6400, 8100, 10000, 22500, 40000]
N_set = [9, 16, 25, 36, 49, 64, 81, 100, 10000]
beta_fix = 4
sigma_set =[0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4]
sigma_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
sigma_set= [0.2]
N_set = [6400]
T = 30
dt = 0.01
t = np.arange(0, T, dt)
plot_num = 100
plot_range = [0, T]
plot_interval = 0.01
bins = np.arange(plot_range[0], plot_range[1], plot_interval) 
log = 0
ave = 0
for N in N_set:

    for sigma in sigma_set:

        des = '../data/grid' + str(degree) + '/size' + str(N) + '/beta' + str(beta_fix) + '/strength=' + str(sigma) + '_T=' + str(T) + '/'
        #label = f'sigma={sigma}'
        label = None
        title = f'$\\beta=${beta_fix}_N={N}_$\\sigma=${sigma}'
        # main.P_from_data(des, bins, label, title, log)
        main.rho_from_data(des, plot_num, t, label, title, log, ave)
#plt.ylim( 5e-2, 1)
#plt.xlim(-1, 100)
# plt.show()

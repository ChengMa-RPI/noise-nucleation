import main
import imageio
import numpy as np
import os 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

'''
N = 10000
beta_fix = 4
sigma = 50
T = 100

degree = 4
dt = 0.01
realization_index = 99

des = '../data/grid' + str(degree) + '/size' + str(N) + '/beta' + str(beta_fix) + '/strength=' + str(sigma) + '_T=' + str(T) + '/'

main.heatmap(des, realization_index, N, plot_range, plot_interval, dt)
'''
duration = 0.3
plot_range = 100
plot_interval = 1

des_gif = '../report/presentation102921/realization1014/' 
with imageio.get_writer(des_gif + f'single_cluster.gif', mode='I', duration=duration) as writer:
    for i in np.arange(20, plot_range, plot_interval):
        filename = des_gif + str(int(i/plot_interval)) + '.png' 
        image = imageio.imread(filename)
        writer.append_data(image)

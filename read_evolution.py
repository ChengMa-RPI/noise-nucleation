import main
import imageio
import numpy as np
import os 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

N = 6400
beta_fix = 4
sigma = 50
T = 100
plot_interval = 0.1

plot_range = 20
duration = 0.3
degree = 4
dt = 0.01
realization_index = 99

des = '../data/grid' + str(degree) + '/size' + str(N) + '/beta' + str(beta_fix) + '/strength=' + str(sigma) + '_T=' + str(T) + '/'

main.heatmap(des, realization_index, N, plot_range, plot_interval, dt)

des_gif = des + f'heatmap/realization{realization_index}/'
with imageio.get_writer(des_gif + f'N={N}_beta={beta_fix}_sigma_{sigma}.gif', mode='I', duration=duration) as writer:
    for i in np.arange(0, plot_range, plot_interval):
        filename = des_gif + str(int(i/plot_interval)) + '.png' 
        image = imageio.imread(filename)
        writer.append_data(image)

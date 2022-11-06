import numpy as np 
import matplotlib.pyplot as plt
import scipy.integrate as sin
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import pandas as pd
from cycler import cycler
import os
import ast
import matplotlib as mpl
from scipy.signal import argrelextrema
from matplotlib.colors import LogNorm
#from read_rho_lifetime import heatmap, rho_from_data, P_from_data, tn_N, plot_P_rho
#from read_lifetime import tau_fit, tau_average, tau_combine, tau_all
from matplotlib import rc
import networkx as nx

degree = 4
x1 = 1
x2 = 2
x3 = 4
fs = 24
lw = 3
alpha = 0.8

def cubic_1D(x, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :c: bifurcation parameter 
    :returns: derivative of x 

    """
    x1, x2, x3 = arguments 
    dxdt = -(x - x1) * (x - x2) * (x - x3)
    return dxdt

def V_eff():
    x_set=np.arange(0, 5, 0.01)
    V_set = []
    for x_end in x_set:
        V, error = sin.quad(cubic_1D, x_set[0], x_end, args=((x1, x2, x3),))
        V_set = np.append(V_set, -V) 
    plt.plot(x_set, V_set[: :], color='k', linewidth=3)
    plt.xlabel('$x$', fontsize=fs)
    plt.ylabel('$V_{eff}$', fontsize=fs)
    plt.xticks(fontsize=15) 
    plt.yticks(fontsize=15) 
    #plt.title('Potential landscape', fontsize=20)
    plt.show()

def demo_con_style(ax, x1, x2, y1, y2, color, linestyle, linewidth, shrink, connectionstyle):
    ax.annotate("", xy=(x1, y1), xycoords='data',  xytext=(x2, y2), textcoords='data', arrowprops=dict(arrowstyle="->", color=color, linewidth=linewidth, ls = linestyle , shrinkA=shrink, shrinkB=shrink, patchA=None, patchB=None, connectionstyle=connectionstyle, ), )

def landscape():
    y1 = -2.88
    y2 = -2.46
    y3 = -5.12
    circle1 = plt.Circle((x1, y1), 0.15, color='brown')
    circle2 = plt.Circle((x2, y2), 0.15, color='brown', alpha = 0.5)
    circle3 = plt.Circle((x3, y3), 0.15, color='brown')

    fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    demo_con_style(ax, x2, x1, y2, y1, 'slateblue', '-', 2, 10,  "angle3,angleA=90,angleB=0")
    demo_con_style(ax, x3, x2, y3, y2, 'darkorange', '-', 2, 30,  "angle3,angleA=0,angleB=90")


    xs = np.linspace(1.4, 2.6, 151)
    ys = 0.5 * np.sin(9 * xs)
    ys += np.linspace(-2.3, 0, 151)
    ax.plot(xs, ys, color="gray", lw="3")

    verts = np.array([[-1,2],[0,0],[1, 2],[-1,2]]).astype(float) * 0.1
    verts[:,0] += xs[0]
    verts[:,1] += ys[0] - 0.1
    path = mpath.Path(verts)
    patch = mpatches.PathPatch(path, fc='gray', ec="gray")
    ax.add_patch(patch)

    '''
    xs1 = np.linspace(3.2, 3.7, 151)
    ys1 = 0.5 * np.sin(18 * xs1)  
    ys1 += np.linspace(-1, -2.5, 151) 
    ax.plot(xs1, ys1, color="gray", lw="3")

    verts1 = np.array([[-1,2],[0,0],[1, 2],[-1,2]]).astype(float) * 0.1
    verts1[:,0] += xs1[-1]
    verts1[:,1] += ys1[-1] - 0.1
    path = mpath.Path(verts1)
    patch = mpatches.PathPatch(path, fc='gray', ec="gray")
    ax.add_patch(patch)
    '''

    ax.annotate("", xy=(3.5, -2.7), xytext=(4, -1.7), arrowprops=dict(arrowstyle="->", color='slategrey', linewidth=3,  ), )

    ax.text(2.5, -0.3, 'noise',size=18)
    #ax.text(0.9, -0.7, 'noise',size=18)
    ax.text(4., -1.6, '$F(x)$',size=18)
    ax.text(x1 -0.2, -3.5, '$x_L$',size=18)
    ax.text(x2 - 0.2, -3.2, '$x_u$',size=18)
    ax.text(x3 - 0.2, -5.8, '$x_H$',size=18)

    ax.axis("off")
    plt.axis('equal')
    V_eff()

def plot_arrow(plot_index, c, stable, unstable, upper, lower, noise):
    for i, plot_index_loop in zip(range(np.size(plot_index)), plot_index):
        if i == 1:
            linewidth = 2
        elif i == 0:
            linewidth = 5
        stable_plot = stable[plot_index_loop]
        if np.size(unstable) == 0:
            y = np.linspace(stable_plot[0], upper, 10)
            plt.annotate("", xy=(c[plot_index_loop], y[0]), xytext=(c[plot_index_loop], np.exp(np.log(y[-1]) - 0.1)), arrowprops=dict(arrowstyle="->", color='tab:grey', lw = 1.8))
            y = np.linspace(lower, stable_plot[0], 10)
            plt.annotate("", xytext=(c[plot_index_loop], y[0]), xy=(c[plot_index_loop], np.exp(np.log(y[-1]) - 0.1)), arrowprops=dict(arrowstyle="->", color='grey', lw = 1.8))

        else:
            unstable_plot = unstable[plot_index_loop]

            y = np.linspace(stable_plot[0], unstable_plot, 10)
            if noise == 1:
                plt.annotate("", xytext=(c[plot_index_loop], y[0]), xy=(c[plot_index_loop], np.exp(np.log(y[-1]))), arrowprops=dict(arrowstyle="->", color='slateblue', lw = linewidth))
            else:
                plt.annotate("", xy=(c[plot_index_loop], y[0]), xytext=(c[plot_index_loop], np.exp(np.log(y[-1]))), arrowprops=dict(arrowstyle="->", color='grey', lw = 1.8))

            y = np.linspace(unstable_plot , stable_plot[1] , 10)
            plt.annotate("", xytext=(c[plot_index_loop], np.exp(np.log(y[0] + 0.1))), xy=(c[plot_index_loop], y[-1]), arrowprops=dict(arrowstyle="->", color='grey', lw = 1.8))

            y = np.linspace(stable_plot[1], upper, 10)
            plt.annotate("", xy=(c[plot_index_loop], y[0]), xytext=(c[plot_index_loop], np.exp(np.log(y[-1]) - 0.1)), arrowprops=dict(arrowstyle="->", color='grey', lw = 1.8))

            y = np.linspace(lower, stable_plot[0], upper, 10)
            plt.annotate("", xytext=(c[plot_index_loop], y[0]), xy=(c[plot_index_loop], np.exp(np.log(y[-1]) - 0.1)), arrowprops=dict(arrowstyle="->", color='grey', lw = 1.8))

def bifurcation(dynamics, upper, lower, noise):
    bifurcation_file = '../data/' + dynamics + str(degree) + '/' + 'bifurcation.csv'
    data = np.array(pd.read_csv(bifurcation_file, header=None).iloc[:, :])
    # extract data of bifurcation parameter and fixed points 
    c = data[0]
    stable = data[1:3].transpose()
    unstable = data[-1]
 
    bifur_index = np.where(unstable != 0)[0]
    bifur_c = c[bifur_index]
    bifur_stable = stable[bifur_index]
    bifur_unstable = unstable[bifur_index]

    unstable_positive = unstable[unstable != 0]
 
    plot_index1 = (np.exp(np.linspace(np.log(bifur_index[2]), np.log(bifur_index[-2]), 8)[1:3])).astype(int)
    plot_index1 = (np.exp(np.linspace(np.log(bifur_index[2]), np.log(bifur_index[-2]), 6)[-3:-1])).astype(int)
    
    plot_index2 = (np.exp(np.linspace(np.log(bifur_index[-1]), np.log(np.size(c)), 3)[1:-1])).astype(int)
    if bifur_index[0] != 0:
        plot_index3 = (np.exp(np.linspace(0, np.log(bifur_index[0]), 10)[-2:-1])).astype(int)
        plot_arrow(plot_index3, c, stable, [], upper, lower, 1)
    
    plot_arrow(plot_index1, c, stable, unstable, upper, lower, noise)
    plot_arrow(plot_index2, c, stable, [], upper, lower, 1)

    plt.loglog(c, stable[:, 1], 'tab:red')
    plt.loglog(bifur_c, bifur_stable[:, 0], 'tab:blue')
    plt.loglog(c[unstable != 0], unstable_positive, '--', color ='tab:green')
    plt.loglog(bifur_c[-1] * np.ones(100), np.linspace(lower, bifur_stable[-1, 1],100), '--k')
    if dynamics == 'mutual':
        plt.xlabel('$\\beta$', fontsize = fs)
        plt.text(7-0.5, lower-0.02, '$\\beta_{c}$',size=fs)
    else:
        plt.xlabel('$c$', fontsize = fs)
    plt.ylabel('$x$', fontsize=fs)
    plt.subplots_adjust(left=0.18, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    # plt.title('Bifurcation of ' + dynamics, fontsize=fs)
    plt.ylim(lower, upper)
    plt.xlim(0.4,15)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.show()

def bifurcation_dynamics(dynamics, upper, lower, noise):
    plt.rc('text', usetex=True)
    plt.rc('font', family='arial')

    bifurcation_file = '../data/' + dynamics + str(degree) + '/' + 'bifurcation.csv'
    data = np.array(pd.read_csv(bifurcation_file, header=None).iloc[:, :])
    # extract data of bifurcation parameter and fixed points 
    c = data[0]
    stable = data[1:3].transpose()
    unstable = data[-1]
 
    bifur_index = np.where(unstable != 0)[0]
    bifur_c = c[bifur_index]
    bifur_stable = stable[bifur_index]
    bifur_unstable = unstable[bifur_index]

    unstable_positive = unstable[unstable != 0]
 
    plot_index1 = (np.exp(np.linspace(np.log(bifur_index[2]), np.log(bifur_index[-2]), 8)[1:3])).astype(int)
    plot_index1 = (np.exp(np.linspace(np.log(bifur_index[2]), np.log(bifur_index[-2]), 6)[-3:-1])).astype(int)
    
    plot_index2 = (np.exp(np.linspace(np.log(bifur_index[-1]), np.log(np.size(c)), 3)[1:-1])).astype(int)
    if bifur_index[0] != 0:
        plot_index3 = (np.exp(np.linspace(0, np.log(bifur_index[0]), 10)[-2:-1])).astype(int)
        #plot_arrow(plot_index3, c, stable, [], upper, lower, 1)
    
    #plot_arrow(plot_index1, c, stable, unstable, upper, lower, noise)
    #plot_arrow(plot_index2, c, stable, [], upper, lower, 1)

    color = ['#66c2a5', '#fc8d62', '#8da0cb']
    #plt.loglog(c[bifur_index[0]:], stable[bifur_index[0]:, 1], color[1])
    plt.loglog(c[:bifur_index[-1]], stable[:bifur_index[-1], 1], color[1])

    plt.loglog(bifur_c, bifur_stable[:, 0], color[2])
    plt.loglog(c[unstable != 0], unstable_positive, '--', color = color[0])

    plt.xlabel('$\\beta$', fontsize = fs)
    plt.ylabel('$x$', fontsize=fs)
    plt.subplots_adjust(left=0.18, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    # plt.title('Bifurcation of ' + dynamics, fontsize=fs)
    plt.ylim(lower, upper)
    #plt.xlim(1.6,2.7)
    #plt.xlim(0.8,7.7)
    plt.xlim(2.5,3.7)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)

    #plt.savefig('../manuscript/figure/'+dynamics + '.svg', format="svg") 
    plt.show()

def plot_transition_line(x, y, minima, maxima, fraction, linewidth, radius, up_down, plot, color='slateblue'):
    nth = int(fraction*minima + (1-fraction)*maxima)
    nth_index = np.where(abs(y - y[nth])<1e-1)[0]
    nth_one = nth_index[nth_index > minima][0]
    nth_two = nth_index[(nth_index > maxima) & (nth_index < minima)][0]
    nth_three = nth_index[nth_index < maxima][0]
    if plot:
        plt.scatter(y[nth], x[nth_two], s= radius, color='brown', alpha=0.5)
        plt.scatter(y[nth], x[nth_three], s= radius, color='brown')
        plt.scatter(y[nth], x[nth_one], s= radius, color='brown')
        if up_down == 1:
            plt.annotate("", xy=(y[nth], x[nth_one]), xytext=(y[nth], x[nth_two]), arrowprops=dict(arrowstyle="-|>, head_width=0.3, head_length=0.5", color='darkorange', lw = 1.8))
            plt.annotate("", xy=(y[nth], x[nth_two]), xytext=(y[nth], x[nth_three]), arrowprops=dict(arrowstyle="->", color=color, lw = linewidth, ls='-'))
        else:
            plt.annotate("", xytext=(y[nth], x[nth_one]), xy=(y[nth], x[nth_two]), arrowprops=dict(arrowstyle="->", color=color, lw = linewidth))
            plt.annotate("", xytext=(y[nth], x[nth_two]), xy=(y[nth], x[nth_three]), arrowprops=dict(arrowstyle="-|>, head_width=0.3, head_length=0.5", color='darkorange', lw = 1.8, ls='-'))
        
    return nth_one, nth_two, nth_three, nth

def bifur_illu():
    fig = plt.figure()
    ax = plt.subplot()
    # bifurcation function
    c = 5
    x1= 1
    x2 = 2
    x3 =4
    f = lambda x: c * (x-x1) * (x-x2) * (x-x3)
    x = np.arange(0.4, 4.3, 0.001)
    y = f(x)
    # minima and maxima of bifurcation 
    minima = argrelextrema(y, np.less)[0][0]
    maxima = argrelextrema(y, np.greater)[0][0]
    y_min = y[minima]
    y_max = y[maxima]
    minima_index = np.where(abs(y - y_min)< 1e-1)[0]
    maxima_index = np.where(abs(y - y_max)< 1e-1)[0]
    minima_two = minima_index[minima_index < maxima]
    maxima_two = maxima_index[maxima_index > maxima]

    # plot bifurcation curves with three sections 
    plt.plot(y[:maxima], x[:maxima], color='r')
    plt.plot(y[maxima:minima], x[maxima:minima], '--', color='y')
    plt.plot(y[minima:], x[minima:], color='g')
    # plt.plot(y_min * np.ones(10), np.linspace(x[minima_two], x[minima],10), '--',  color='k')
    plt.plot(y_min * np.ones(10), np.linspace(0, x[minima],10), '--',  color='k')
    # plt.plot(y_max * np.ones(10), np.linspace(x[maxima_two], x[maxima],10), '--',  color='k')
    plt.plot(y_max * np.ones(10), np.linspace(0, x[maxima_index[-1]],10), '--',  color='k')

    six_one, six_two, six_three, x_six = plot_transition_line(x, y, minima, maxima, 1/6, 2, 80, 1, 1)
    three_one, three_two, three_three, x_three = plot_transition_line(x, y, minima, maxima, 1/3, 5, 80, 1, 1)
    five_one, five_two, five_three, x_five = plot_transition_line(x, y, minima, maxima, 5/6, 2, 80, 0, 1)
    two_one, two_two, two_three, x_two = plot_transition_line(x, y, minima, maxima, 2/3, 5, 80, 0, 1)
    middle_one, middle_two, middle_three, x_middle = plot_transition_line(x, y, minima, maxima, 1/2, 5, 80, 0, 0)

    xs1 = np.linspace(y[six_one]+.1, y[maxima]+2, 151)
    ys1 = 0.03 * np.sin(10* xs1 + 3)  
    ys1 += np.linspace(x[int((six_two+six_three)/2)], x[maxima]-1, 151) 
    ax.plot(xs1, ys1, color="gray", lw="2")
    xs2 = np.linspace(y[three_one]+.1, y[maxima]+2, 151)
    ys2 = 0.03 * np.sin(10* xs2 + 3)  
    ys2 += np.linspace(x[int((six_two+six_three)/2)], x[maxima]-1, 151) 
    ax.plot(xs2, ys2, color="gray", lw="2")

    ax.text(y[maxima], x[maxima]-1.4, 'noise', size=18)
    ax.text(y[maxima]+ 2, x[int((six_one + six_two)/2)]-0.1, '$F(x)$', size=18)
    plt.annotate("", xytext=(y[maxima]+2, x[int((six_one + six_two)/2)]), xy=(y[int(six_one )], x[int(1/3*six_two+2/3*six_one)]), arrowprops=dict(arrowstyle='->', color='slategrey', lw = 1.8, ls='-', connectionstyle="arc3, rad=0",))
    plt.annotate("", xytext=(y[maxima]+2, x[int((six_one + six_two)/2)]), xy=(y[int(three_one)], x[int(1/3*six_one + 2/3*six_two)]), arrowprops=dict(arrowstyle='->', color='slategrey', lw = 1.8, ls='-', connectionstyle="arc3, rad=0",))
    
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height
    plt.axis('off')
    
    ax.arrow(f(x[0]-0.05), 0, (f(x[-1]+0.05) - f(x[0]-0.05)), 0., fc='k', ec='k', lw = 1, head_width=0.1, head_length=1, overhang = 0.3, length_includes_head= True, clip_on = False) 
    ax.arrow(f(x[0]-0.05), 0, 0, x[-1]-x[0] + 1, fc='k', ec='k', lw = 1, head_width=0.1/(x[-1]-x[0] + 1) * (f(x[-1]+0.05) - f(x[0]-0.05))* height/width, head_length=1*(x[-1]-x[0] + 1) / (f(x[-1]+0.05) - f(x[0]-0.05)) * width/height, overhang = 0.3, length_includes_head= True, clip_on = False) 

    plt.annotate("", xy=(y[middle_one], x[middle_one]), xytext=(y[middle_one]-3, x[middle_one]+0.5), arrowprops=dict(arrowstyle='->', color='slategrey', lw = 1.8, ls='-', connectionstyle="angle3, angleA=0, angleB=90",))
    ax.text(y[middle_one]-5, x[middle_one]+0.5, '$x_H$',size=18)
    plt.annotate("", xy=(y[middle_three], x[middle_three]), xytext=(y[middle_three]-3, x[middle_three]-0.5), arrowprops=dict(arrowstyle='->', color='slategrey', lw = 1.8, ls='-', connectionstyle="angle3, angleA=0, angleB=90",))
    ax.text(y[middle_three]-5, x[middle_three]-0.5, '$x_L$',size=18)
    plt.annotate("", xy=(y[middle_two], x[middle_two]), xytext=(y[middle_two]-1, x[middle_two]-0.3), arrowprops=dict(arrowstyle='->', color='slategrey', lw = 1.8, ls='-', connectionstyle="angle3, angleA=0, angleB=90",))
    ax.text(y[middle_two]-3, x[middle_two]-0.3, '$x_u$',size=18)

    ax.text(f(x[-1]), -0.4, '$\\beta$',size=fs)
    ax.text(f(x[0] - 0.09), x[-1], '$x$',size=fs)
    ax.text(y_min, -0.4, '$\\beta_{c1}$',size=fs)
    ax.text(y_max, -0.4, '$\\beta_{c2}$',size=fs)
    plt.xlabel('$\\beta$', fontsize=22)
    plt.ylabel('$x$',fontsize=22)
    plt.show()

def bifur_illu_control():
    fig = plt.figure()
    ax = plt.subplot()
    # bifurcation function
    c = 5
    x1= 1
    x2 = 2
    x3 =4
    f = lambda x: c * (x-x1) * (x-x2) * (x-x3)
    x = np.arange(0.4, 4.3, 0.001)
    y = f(x)
    # minima and maxima of bifurcation 
    minima = argrelextrema(y, np.less)[0][0]
    maxima = argrelextrema(y, np.greater)[0][0]
    y_min = y[minima]
    y_max = y[maxima]
    minima_index = np.where(abs(y - y_min)< 1e-1)[0]
    maxima_index = np.where(abs(y - y_max)< 1e-1)[0]
    minima_two = minima_index[minima_index < maxima]
    maxima_two = maxima_index[maxima_index > maxima]

    # plot bifurcation curves with three sections 
    plt.plot(y[:maxima], x[:maxima], color='tab:blue')
    plt.plot(y[maxima:minima], x[maxima:minima], '--', color='tab:green')
    plt.plot(y[minima:], x[minima:], color='tab:red')
    plt.plot(y_min * np.ones(10), np.linspace(0, x[minima],10), '--',  color='k')
    plt.plot(y_max * np.ones(10), np.linspace(0, x[maxima_index[-1]],10), '--',  color='k')

    plt.xticks([], []) 
    plt.yticks([], []) 
    plt.ylim(0.3, 4.4)
    ax.text(y_min, 0.0, '$\\beta_{c_1}$',size=21)
    ax.text(y_max, 0.0, '$\\beta_{c_2}$',size=21)
    ax.set_xlabel('$\\beta$', fontsize=fs)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.set_ylabel('$x$', fontsize=fs)
    ax.yaxis.set_label_coords(-0.1, 0.5)
    plt.subplots_adjust(left=0.18, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    plt.show()

def bifur_illu_two():
    fig = plt.figure()
    ax = plt.subplot()
    # bifurcation function
    c = 5
    x1= 1
    x2 = 2
    x3 =4
    f = lambda x: c * (x-x1) * (x-x2) * (x-x3)
    x = np.arange(0.4, 4.3, 0.001)
    y = f(x)
    # minima and maxima of bifurcation 
    minima = argrelextrema(y, np.less)[0][0]
    maxima = argrelextrema(y, np.greater)[0][0]
    y_min = y[minima]
    y_max = y[maxima]
    minima_index = np.where(abs(y - y_min)< 1e-1)[0]
    maxima_index = np.where(abs(y - y_max)< 1e-1)[0]
    minima_two = minima_index[minima_index < maxima]
    maxima_two = maxima_index[maxima_index > maxima]

    # plot bifurcation curves with three sections 
    plt.rc('text', usetex=True)
    plt.rc('font', family='arial')

    plt.plot(y[:maxima], x[:maxima], color='k')
    plt.plot(y[maxima:minima], x[maxima:minima], '--', color='k')
    plt.plot(y[minima:], x[minima:], color='k')
    # plt.plot(y_min * np.ones(10), np.linspace(x[minima_two], x[minima],10), '--',  color='k')
    plt.plot(y_min * np.ones(10), np.linspace(0, x[minima],10), '--',  color='blueviolet')
    plt.plot(y_max * np.ones(10), np.linspace(0, x[maxima_index[-1]],10), '--',  color='blueviolet')

    six_one, six_two, six_three, x_six = plot_transition_line(x, y, minima, maxima, 1/6, 2, 80, 1, 1, 'tab:blue')
    three_one, three_two, three_three, x_three = plot_transition_line(x, y, minima, maxima, 1/3, 2, 80, 1, 1,'tab:red')
    middle_one, middle_two, middle_three, x_middle = plot_transition_line(x, y, minima, maxima, 1/2, 5, 80, 0, 0)

    plt.plot(y[x_six] * np.ones(10), np.linspace(0, x[six_three],10), '--',  color='blueviolet')
    plt.plot(y[x_three] * np.ones(10), np.linspace(0, x[three_three],10), '--',  color='blueviolet')

    xs1 = np.linspace(y[six_one]+.1, y[maxima]+2, 151)
    ys1 = 0.03 * np.sin(10* xs1 + 3)  
    ys1 += np.linspace(x[int((six_two+six_three)/2)], x[maxima]-1, 151) 
    ax.plot(xs1, ys1, color="gray", lw="2")
    xs2 = np.linspace(y[three_one]+.1, y[maxima]+2, 151)
    ys2 = 0.03 * np.sin(10* xs2 + 3)  
    ys2 += np.linspace(x[int((six_two+six_three)/2)], x[maxima]-1, 151) 
    ax.plot(xs2, ys2, color="gray", lw="2")

    ax.text(y[maxima], x[maxima]-1.4, 'noise', size=18)
    ax.text(y[maxima]+ 2, x[int((six_one + six_two)/2)]-0.1, '$F(x)$', size=18)
    plt.annotate("", xytext=(y[maxima]+2, x[int((six_one + six_two)/2)]), xy=(y[int(six_one )], x[int(1/3*six_two+2/3*six_one)]), arrowprops=dict(arrowstyle='->', color='slategrey', lw = 1.8, ls='-', connectionstyle="arc3, rad=0",))
    plt.annotate("", xytext=(y[maxima]+2, x[int((six_one + six_two)/2)]), xy=(y[int(three_one)], x[int(1/3*six_one + 2/3*six_two)]), arrowprops=dict(arrowstyle='->', color='slategrey', lw = 1.8, ls='-', connectionstyle="arc3, rad=0",))
    
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height
    plt.axis('off')
    
    ax.arrow(f(x[0]-0.05)+1, 0, (f(x[-1]+0.05) - f(x[0]-0.05) - 3), 0., fc='k', ec='k', lw = 1, head_width=0.1, head_length=1, overhang = 0.3, length_includes_head= True, clip_on = False) 
    ax.arrow(f(x[0]-0.05)+1, 0, 0, x[-1]-x[0] + 1, fc='k', ec='k', lw = 1, head_width=0.1/(x[-1]-x[0] + 1) * (f(x[-1]+0.05) - f(x[0]-0.05))* height/width, head_length=1*(x[-1]-x[0] + 1) / (f(x[-1]+0.05) - f(x[0]-0.05)) * width/height, overhang = 0.3, length_includes_head= True, clip_on = False) 

    plt.annotate("", xy=(y[middle_one], x[middle_one]), xytext=(y[middle_one]-3, x[middle_one]+0.5), arrowprops=dict(arrowstyle='->', color='slategrey', lw = 1.8, ls='-', connectionstyle="angle3, angleA=0, angleB=90",))
    ax.text(y[middle_one]-5, x[middle_one]+0.5, '$x_H$',size=18)
    plt.annotate("", xy=(y[middle_three], x[middle_three]), xytext=(y[middle_three]-3, x[middle_three]-0.5), arrowprops=dict(arrowstyle='->', color='slategrey', lw = 1.8, ls='-', connectionstyle="angle3, angleA=0, angleB=90",))
    ax.text(y[middle_three]-5, x[middle_three]-0.5, '$x_L$',size=18)
    plt.annotate("", xy=(y[middle_two], x[middle_two]), xytext=(y[middle_two]-1, x[middle_two]-0.3), arrowprops=dict(arrowstyle='->', color='slategrey', lw = 1.8, ls='-', connectionstyle="angle3, angleA=0, angleB=90",))
    ax.text(y[middle_two]-3, x[middle_two]-0.3, '$x_u$',size=18)

    ax.text(y[x_six]-1.2, -0.4, '$\\beta_2$',size=fs, color='tab:blue')
    ax.text(y[x_three]-1.2, -0.4, '$\\beta_1$',size=fs, color='tab:red')
    ax.text(f(x[-1])-0.1, -0.4, '$\\beta$',size=fs)
    ax.text(f(x[0] - 0.09), x[-1], '$x$',size=fs)
    ax.text(y_min, -0.4, '$\\beta_{c_1}$',size=fs)
    ax.text(y_max, -0.4, '$\\beta_{c_2}$',size=fs)
    plt.xlabel('$\\beta$', fontsize=fs+1)
    plt.ylabel('$x$',fontsize=fs+1)
    plt.subplots_adjust(left=0.1, right=0.98, wspace=0.25, hspace=0.25, bottom=0.06, top=0.88)
    plt.savefig("../summery/F1a.svg", format="svg") 
    plt.show()

def x_t_illu():
    fig = plt.figure()
    ax = plt.subplot()
    bifurcation_file = '../data/eutrophication4/bifurcation.csv'
    fixed = np.array(pd.read_csv(bifurcation_file, header=None).iloc[:, :])
    c = fixed[0]
    lower = fixed[1]
    higher = fixed[2]
    unstable = fixed[3]
    c_small = 6
    c_large = 6.35

    index_small = np.where(c == c_small)[0][0]
    lower_small = lower[index_small]
    higher_small = higher[index_small]
    unstable_small = unstable[index_small]

    index_large = np.where(abs(c-c_large) < 1e-10)[0][0]
    lower_large = lower[index_large]
    higher_large = higher[index_large]
    unstable_large = unstable[index_large]

    t = np.linspace(0, 30, 3001)
    data_small = np.load('../data/eutrophication4/size1/c6/strength=0.03_R0/ave/realization6_T_0_100.npy')[:np.size(t)]
    data_large = np.load('../data/eutrophication4/size1/c6.35/strength=0.03_R0/ave/realization6_T_0_100.npy')[:np.size(t)]
    times = 1
    constant = 0.008 * times
    x_L = 0 + constant
    x_H = 1 * times  + constant 
    rho_small = (data_small - lower_small)/(higher_small - lower_small) * times   + constant 
    xu_small = (unstable_small - lower_small)/(higher_small - lower_small) * times + constant
    rho_small_log = np.log10(rho_small)

    rho_large = (data_large - lower_large)/(higher_large - lower_large) * times + constant 
    xu_large = (unstable_large - lower_large)/(higher_large - lower_large) * times + constant + 0.003
    rho_large_log = np.log10(rho_large)

    x_L_log = np.log10(x_L)
    x_H_log = np.log10(x_H)
    x_half_log = (x_L_log + x_H_log)/2

    rho_small_log[rho_small_log<x_L_log] = (rho_small_log[rho_small_log<x_L_log] - x_L_log) * 0.30 + x_L_log
    rho_large_log[rho_large_log<x_L_log] = (rho_large_log[rho_large_log<x_L_log] - x_L_log)* 0.75 + x_L_log
    xu_small_log = np.log10(xu_small)
    exceed_small = np.where(rho_small_log>x_H_log)[0][0]
    xu_large_log = np.log10(xu_large)
    exceed_large = np.where(rho_large_log>x_H_log)[0][0]
    fluctuation = 0.005

    rho_small_log[exceed_small:] = (rho_small_log[exceed_small:]- x_H_log) * 5 + x_H_log
    rho_large_log[exceed_large:] = (rho_large_log[exceed_large:]- x_H_log) * 5 + x_H_log


    transition_index_small = np.where(abs(rho_small_log-x_half_log)<1e-2)[0][0]
    transition_index_large = np.where(abs(rho_large_log-x_half_log)<1e-2)[0][0]
    plt.plot(t[transition_index_small] * np.ones(10), np.linspace(x_L_log-0.4, x_half_log, 10), '--', color='blueviolet')

    plt.plot(t[transition_index_large] * np.ones(10), np.linspace(x_L_log-0.4, x_half_log, 10), '--', color='blueviolet')
    plt.plot(t, rho_small_log, color='r', label='$\\beta_1$')
    plt.plot(t, rho_large_log, color='b', label='$\\beta_2$')

    end = 2200
    plt.plot(t[:end], xu_small_log * np.ones(np.size(t[:end])), '--', color='r')
    plt.plot(t[:end], xu_large_log * np.ones(np.size(t[:end])), '--', color='b')

    plt.plot(t[:end], x_L_log * np.ones(np.size(t[:end])), '--', color='k')
    plt.plot(t[:end], x_H_log * np.ones(np.size(t[:end])), '--', color='k')
    plt.plot(t[:end], x_half_log * np.ones(np.size(t[:end])), '--', color='k')

    plt.xlabel('$t$', fontsize=fs)
    plt.ylabel('$x$', fontsize=fs)

    plt.text(0 - 3, xu_small_log, '$x_{u1}$', fontsize=fs, color='r')
    plt.text(0 - 3, xu_large_log, '$x_{u2}$', fontsize=fs, color='b')
    plt.text(0 - 3, x_L_log, '$x_{L}$', fontsize=fs)
    plt.text(0 - 3, x_H_log, '$x_{H}$', fontsize=fs)
    plt.text(0 - 4.9, x_half_log, '$\\frac{x_L + x_H}{2}$', fontsize=fs)
    plt.text(t[transition_index_large], x_L_log - 0.6, '$\\tau_{2}$', fontsize = fs, color='b')
    plt.text(t[transition_index_small], x_L_log - 0.6, '$\\tau_{1}$', fontsize = fs, color='r')

    plt.text(0 - 4, x_H_log + 0.3, '$x$', fontsize=22)
    plt.text(28, x_L_log-0.7, '$t$', fontsize=22)

    plt.axis('off')
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    ax.arrow(-0.2, x_L_log-0.4, 30, 0., fc='k', ec='k', lw = 1, head_width=0.1, head_length=1, overhang = 0.3, length_includes_head= True, clip_on = False) 
    ax.arrow(-0.2, x_L_log-0.4, 0, 3, fc='k', ec='k', lw = 1, head_width=0.1 * 10 * height/width, head_length=1 * 0.1 * width/height, overhang = 0.3, length_includes_head= True, clip_on = False) 

    plt.legend(loc='best', bbox_to_anchor=(0.5, -0.01, 0.5, 0.5), fontsize=18 )
    plt.show()

def evolution(fixed, c, data, constant, adjustment_unstable, decrease, increase, t, end, color, label, tau_label, linestyle, x_L=0, x_H=1):

    c_bifur = fixed[0]
    lower = fixed[1]
    higher = fixed[2]
    unstable = fixed[3]
    index = np.where(abs(c_bifur-c) < 1e-10)[0][0]
    lower = lower[index]
    higher = higher[index]
    unstable = unstable[index]

    x_L = x_L + constant 
    x_H = x_H + constant 
    x_L_log = np.log10(x_L)
    x_H_log = np.log10(x_H)
    x_half_log = (x_L_log + x_H_log)/2

    rho = (data - lower)/(higher - lower)    + constant 
    xu = (unstable - lower)/(higher - lower)  + constant + adjustment_unstable
    rho_log = np.log10(rho)
    rho_log[rho_log<x_L_log] = (rho_log[rho_log<x_L_log] - x_L_log) * decrease + x_L_log
    xu_log = np.log10(xu)
    exceed = np.where(rho_log>x_H_log)[0][0]
    rho_log[exceed:] = (rho_log[exceed:]- x_H_log) * increase + x_H_log
    transition_index = np.where(abs(rho_log-x_half_log)<5e-2)[0][0]

    plt.plot(t[transition_index] * np.ones(10), np.linspace(x_L_log-0.4, x_half_log, 10), '--', linewidth=lw-1, alpha = alpha, color=color)
    plt.plot(t, rho_log, color=color, linewidth=lw, alpha = alpha, label=label, linestyle = linestyle)
    plt.text(t[transition_index], x_L_log - 0.6, tau_label, fontsize = fs, color=color)

    return x_L_log, x_H_log, x_half_log, xu_log

def x_t_beta():
    fig = plt.figure()
    ax = plt.subplot()
    bifurcation_file = '../data/eutrophication4/bifurcation.csv'
    fixed = np.array(pd.read_csv(bifurcation_file, header=None).iloc[:, :])
    c_small = 6
    c_large = 6.35

    t = np.linspace(0, 30, 3001)
    data_small = np.load('../data/eutrophication4/size1/c6/strength=0.03_R0/ave/realization6_T_0_100.npy')[:np.size(t)]
    data_large = np.load('../data/eutrophication4/size1/c6.35/strength=0.03_R0/ave/realization6_T_0_100.npy')[:np.size(t)]

    end = 2200
    constant = 0.008
    x_L_log, x_H_log, x_half_log, xu_small_log = evolution(fixed, c_small, data_small, constant, 0, 0.3, 5, t, end, 'tab:red', '$\\beta_1$ small', '$\\tau_1$', '-.')
    x_L_log, x_H_log, x_half_log, xu_large_log = evolution(fixed, c_large, data_large, constant, 0.003, 0.75, 5, t, end, 'tab:blue', '$\\beta_2$ large', '$\\tau_2$', '--')


    plt.rc('text', usetex=True)
    plt.rc('font', family='arial')
    plt.plot(t[:end], x_L_log * np.ones(np.size(t[:end])), '--', color='k', linewidth=lw-1, alpha = alpha-0.2)
    plt.plot(t[:end], x_H_log * np.ones(np.size(t[:end])), '--', color='k', linewidth=lw-1, alpha = alpha-0.2)
    plt.plot(t[:end], x_half_log * np.ones(np.size(t[:end])), '--', color='k', linewidth=lw-1, alpha = alpha-0.2)
    plt.plot(t[:end], xu_small_log * np.ones(np.size(t[:end])), '--', color='tab:red', linewidth=lw-1, alpha = alpha)
    plt.plot(t[:end], xu_large_log * np.ones(np.size(t[:end])), '--', color='tab:blue', linewidth=lw-1, alpha = alpha)

    plt.axis('off')
    # plt.text(0 - 3, xu_small_log, '$x_{u1}$', fontsize=fs, color='r')
    # plt.text(0 - 3, xu_large_log, '$x_{u2}$', fontsize=fs, color='b')
    # plt.text(0 - 3, x_L_log, '$x_{L}$', fontsize=fs)
    # plt.text(0 - 3, x_H_log, '$x_{H}$', fontsize=fs)
    # plt.text(0 - 4.9, x_half_log, '$\\frac{x_L + x_H}{2}$', fontsize=fs)
    # plt.text(0 - 4, x_H_log + 0.3, '$x$', fontsize=22)
    plt.text(0 - 3, xu_small_log, '$\\rho_{u_1}$', fontsize=fs, color='tab:red')
    plt.text(0 - 3, xu_large_log, '$\\rho_{u_2}$', fontsize=fs, color='tab:blue')
    plt.text(0 - 2, x_L_log - 0.08, '$0$', fontsize=fs)
    plt.text(0 - 2, x_H_log-0.08, '$1$', fontsize=fs)
    plt.text(0 - 3.5, x_half_log-0.08, '$0.5$', fontsize=fs)

    plt.text(0 - 3, x_H_log + 0.3, '$\\rho$', fontsize=fs+1)
    plt.text(28, x_L_log-0.7, '$t$', fontsize=fs+1)

    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height
    ax.arrow(-0.2, x_L_log-0.4, 30, 0., fc='k', ec='k', lw = 1, head_width=0.1, head_length=1, overhang = 0.3, length_includes_head= True, clip_on = False) 
    ax.arrow(-0.2, x_L_log-0.4, 0, 3, fc='k', ec='k', lw = 1, head_width=0.1 * 10 * height/width, head_length=1 * 0.1 * width/height, overhang = 0.3, length_includes_head= True, clip_on = False) 
    plt.legend(loc='best', bbox_to_anchor=(0.5, -0.2, 0.5, 0.5), fontsize=18 )
    plt.subplots_adjust(left=0.1, right=0.98, wspace=0.25, hspace=0.25, bottom=0.06, top=0.88)
    #plt.savefig("../summery/F1b.svg", format="svg") 
    plt.show()

def x_t_noise():
    fig = plt.figure()
    ax = plt.subplot()
    bifurcation_file = '../data/eutrophication4/bifurcation.csv'
    fixed = np.array(pd.read_csv(bifurcation_file, header=None).iloc[:, :])
    c = 6

    t = np.linspace(0, 50, 5001)
    data_large = np.load('../data/eutrophication4/size1/c6/strength=0.03_R0/ave/realization6_T_0_100.npy')[:np.size(t)]
    data_small = np.load('../data/eutrophication4/size1/c6/strength=0.01_R0/ave/realization6_T_0_100.npy')[:np.size(t)]

    end = 3500
    constant = 0.008
    x_L_log, x_H_log, x_half_log, xu_small_log = evolution(fixed, c, data_small, constant, 0, 0.8, 5, t, end,  (252/255, 141/255, 98/255), '$\\sigma_1$ small', '$\\tau_1$', '-.')
    x_L_log, x_H_log, x_half_log, xu_large_log = evolution(fixed, c, data_large, constant, 0, 0.8, 5, t, end, (141/255, 160/255, 203/255), '$\\sigma_2$ large', '$\\tau_2$', '--')

    plt.plot(t[:end], x_L_log * np.ones(np.size(t[:end])), '--', color='k', linewidth=lw-1, alpha = alpha-0.2)
    plt.plot(t[:end], x_H_log * np.ones(np.size(t[:end])), '--', color='k', linewidth=lw-1, alpha = alpha-0.2)
    plt.plot(t[:end], x_half_log * np.ones(np.size(t[:end])), '--', color='k', linewidth=lw-1, alpha = alpha-0.2)
    plt.plot(t[:end], xu_large_log * np.ones(np.size(t[:end])), '--', color='k', linewidth=lw-1, alpha = alpha-0.2)
    plt.axis('off')
    # plt.text(0 - 5, xu_large_log, '$x_{u}$', fontsize=fs, color='k')
    # plt.text(0 - 5, x_L_log, '$x_{L}$', fontsize=fs)
    # plt.text(0 - 5, x_H_log, '$x_{H}$', fontsize=fs)
    # plt.text(0 - 7.9, x_half_log, '$\\frac{x_L + x_H}{2}$', fontsize=fs)
    # plt.text(0 - 5.5, x_H_log + 0.3, '$x$', fontsize=22)
    plt.text(48, x_L_log-0.7, '$t$', fontsize=fs+1)

    plt.text(0 - 5.5, x_H_log + 0.3, '$\\rho$', fontsize=fs+1)
    plt.text(0 - 4, xu_large_log, '$\\rho_{u}$', fontsize=fs, color='k')
    plt.text(0 - 3, x_L_log -0.08, '$0$', fontsize=fs)
    plt.text(0 - 3, x_H_log -0.08, '$1$', fontsize=fs)
    plt.text(0 - 4.9, x_half_log -0.08, '$0.5$', fontsize=fs)

    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    ax.arrow(-0.2, x_L_log-0.4, 50, 0., fc='k', ec='k', lw = 1, head_width=0.1, head_length=1, overhang = 0.3, length_includes_head= True, clip_on = False) 
    ax.arrow(-0.2, x_L_log-0.4, 0, 3, fc='k', ec='k', lw = 1, head_width=0.1 * 10 * height/width, head_length=1 * 0.1 * width/height, overhang = 0.3, length_includes_head= True, clip_on = False) 

    plt.legend(loc='best', frameon=False, bbox_to_anchor=(0.5, 0.2, 0.5, 0.5), fontsize=15 )
    plt.subplots_adjust(left=0.06, right=0.99, wspace=0.25, hspace=0.25, bottom=0.06, top=0.88)
    plt.savefig("../F1c.svg", format="svg") 
    plt.show()

def plot_figure_table(plot_list, index, sigma_set, realization_index1, realization_index2):
    degree = 4
    arguments = []
    dynamics_set = ['mutual', 'harvest', 'eutrophication', 'vegetation']
    c_set = [4, 1.8, 6, 2.7]
    R_set = [0.02]

    des = '../summery/summery062420/' 
    #evolution heatmap:
    if 1 in plot_list:
        N_set = [100]
        plot_type = 'heatmap'
        sigma_set = sigma_set1
        fig = plot_P_rho(dynamics_set[index], c_set[index], R_set, plot_type, N_set, sigma_set, realization_index=realization_index1) 
        plt.close(fig)
    if 2 in plot_list:
        N_set = [10000]
        plot_type = 'heatmap'
        sigma_set = sigma_set1
        fig = plot_P_rho(dynamics_set[index], c_set[index], R_set, plot_type, N_set, sigma_set, realization_index=realization_index2) 
        plt.close(fig)

    # compare 100 realizations of \rho, N =100, 10000, \sigma = 0.2
    if 3 in plot_list:
        plot_type = 'rho'
        N_set = [100]
        sigma_set = sigma_set1
        fig = plot_P_rho(dynamics_set[index], c_set[index], R_set, plot_type, N_set, sigma_set, plot_num = 100, t=np.arange(0, 100, 0.01), log=0, ave =0, color='tab:red', fit =0)
        fig.savefig(des + dynamics_set[index] + '_'+ plot_type + '_' + f'N_{N_set[0]}' + '_sigma_' + str(sigma_set[0]).replace('.', '') + '_R' + str(R_set[0]).replace('.', '') + '.png')
 
        N_set = [10000]
        sigma_set = sigma_set1
        fig = plot_P_rho(dynamics_set[index], c_set[index], R_set, plot_type, N_set, sigma_set, plot_num = 50, t=np.arange(0, 100, 0.01), log=0, ave =0, color='tab:red', fit =0)
        fig.savefig(des + dynamics_set[index] + '_'+ plot_type + '_' + f'N_{N_set[0]}' + '_sigma_' + str(sigma_set[0]).replace('.', '') + '_R' + str(R_set[0]).replace('.', '') + '.png')

    if 4 in plot_list:
        mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:blue','tab:green', 'tab:red',  'tab:orange', 'tab:grey']) * cycler(linestyle=['-', '-']))
        plot_type = 'P'
        N_set = [100]
        sigma_set = sigma_set2
        plot_range = [0, 100]
        bins = np.logspace(np.log10(plot_range[0] + 0.1), np.log10(plot_range[1]), 1001)

        fig = plot_P_rho(dynamics_set[index], c_set[index], R_set, plot_type, N_set, sigma_set, plot_num = 10000, bins = bins, log=1, fit =1)
        fig.savefig(des + dynamics_set[index] + '_'+ plot_type + '_' + f'N_{N_set[0]}' + 'R' + str(R_set[0]) + '.png')
 
        mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:blue', 'tab:orange', 'tab:brown', 'tab:pink', 'grey', 'tab:cyan', 'tab:green', 'tab:red']) * cycler(linestyle=['-', '-']))

        plot_type = 'P'
        N_set = [9, 16, 25, 36, 49, 64, 81, 100]
        sigma_set = sigma_set1
        fig = plot_P_rho(dynamics_set[index], c_set[index], R_set, plot_type, N_set, sigma_set, plot_num = 10000, bins = bins, log=1, fit =1)
        fig.savefig(des + dynamics_set[index] + '_'+ plot_type + '_' + 'sigma_' +str(sigma_set[0]).replace('.', '')  + 'R' + str(R_set[0]) + '.png')

    if 5 in plot_list:
        mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:blue', 'tab:green', 'tab:red', 'tab:orange', 'tab:grey']) * cycler(linestyle=['-', '-']))
        N_set = [100]
        sigma_set = sigma_set2
        plot_type = 'rho'
        fig = plot_P_rho(dynamics_set[index], c_set[index], R_set, plot_type, N_set, sigma_set, plot_num = 1, t=np.arange(0,10, 0.01), log=1, ave = 1, xlim=(-1, 150), ylim=(1e-3, 1.1), fit =1, interval=[10, 1])
        fig.savefig(des + dynamics_set[index] + '_'+ plot_type + '_' + f'N_{N_set[0]}' + '_R' + str(R_set[0]).replace('.', '') + '.png')

    if 6 in plot_list:
        mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:grey']) * cycler(linestyle=['-', '-']))
        N_set = [10000]
        sigma_set = sigma_set3
        plot_type = 'rho'
        fig = plot_P_rho(dynamics_set[index], c_set[index], R_set, plot_type, N_set, sigma_set, plot_num = 100, t=np.arange(0,100, 0.01), log=1, ave = 1, xlim=(-1, 11), ylim=(1e-3, 1.1), fit =1, interval=[10, 1])
        fig.savefig(des + dynamics_set[index] + '_'+ plot_type + '_' + f'N_{N_set[0]}' + '_R' + str(R_set[0]).replace('.', '') + '.png')

        mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:grey']) * cycler(linestyle=['-']))
        plot_type = 'P'
        plot_range = [0, 80]
        bins = np.logspace(np.log10(plot_range[0] + 0.1), np.log10(plot_range[1]), 1001)

        fig = plot_P_rho(dynamics_set[index], c_set[index], R_set, plot_type, N_set, sigma_set, plot_num = 10000, bins=bins, log=0, ave = 1, fit =0)
        fig.savefig(des + dynamics_set[index] + '_'+ plot_type + '_' + f'N_{N_set[0]}'  + '_R' + str(R_set[0]).replace('.', '') + '.png')

    if 7 in plot_list:
        plot_type = 'tn_N'
        N_set = [9, 16, 25, 36, 49, 64, 81, 100]
        sigma_set = sigma_set1
        plot_range = [0, 100]
        bins = np.logspace(np.log10(plot_range[0] + 0.1), np.log10(plot_range[1]), 1001)
        fig = plot_P_rho(dynamics_set[index], c_set[index], R_set, plot_type, N_set, sigma_set, bins=bins)
        fig.savefig(des + dynamics_set[index] + '_'+ plot_type + '_' + 'sigma_' +str(sigma_set[0]).replace('.', '') + '_R' + str(R_set[0]).replace('.', '') + '.png')

    if 8 in plot_list:
        mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:grey']) * cycler(linestyle=['-', '-']))
        plot_type = 'P'
        N_set = [9, 100, 900, 2500, 10000]
        sigma_set = sigma_set4
        plot_range = [0, 1000]
        bins = np.logspace(np.log10(plot_range[0] + 0.1), np.log10(plot_range[1]), 1001)
        fig = plot_P_rho(dynamics_set[index], c_set[index], R_set, plot_type, N_set, sigma_set, plot_num=10000, bins=bins, log=1, fit=1)
        fig.savefig(des + dynamics_set[index] + '_'+ plot_type + '_' + 'sigma_' +str(sigma_set[0]).replace('.', '') + '_R' + str(R_set[0]).replace('.', '') + '.png')

    if 9 in plot_list:
        mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:grey']) * cycler(linestyle=['-']))
        plot_type = 'rho'
        N_set = [100, 900, 2500, 10000]
        sigma_set = sigma_set5
        plot_range = [0, 10]
        bins = np.logspace(np.log10(plot_range[0] + 0.1), np.log10(plot_range[1]), 1001)
        fig = plot_P_rho(dynamics_set[index], c_set[index], R_set, plot_type, N_set, sigma_set, plot_num=100, t=np.arange(0,30, 0.01), ave=1, log = 0, fit=0 )
        fig.savefig(des + dynamics_set[index] + '_'+ plot_type + '_' + 'sigma_' +str(sigma_set[0]).replace('.', '')  + '_R' + str(R_set[0]).replace('.', '') + '.png')
        plot_type = 'P'
        plot_range = [0, 20]
        fig = plot_P_rho(dynamics_set[index], c_set[index], R_set, plot_type, N_set, sigma_set, plot_num=1000, bins=bins, log=0 )
        fig.savefig(des + dynamics_set[index] + '_'+ plot_type + '_' + 'sigma_' +str(sigma_set[0]).replace('.', '')  + '_R' + str(R_set[0]).replace('.', '') + '.png')

    if 10 in plot_list:
        mpl.rcParams['axes.prop_cycle'] = (cycler(color=['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:grey']) * cycler(linestyle=['-']))
        plot_type = 'heatmap'
        ticksize = 13
        legendsize=14
        N_set = [9, 100, 900, 2500, 10000][::-1]
        sigma_set = sigma_set6
        arguments = []
        plot_range = [0, 10000]
        bins = np.logspace(np.log10(plot_range[0] + 0.1), np.log10(plot_range[1]), plot_range[-1] + 1)
        criteria = 1
        endpoint = 1000
        fit = 0
        powerlaw = 1
        fig = tau_all(dynamics_set[index], N_set, sigma_set, R_set, c_set[index], arguments, bins, criteria, fit, powerlaw, plot_type)
        fig.savefig(des + dynamics_set[index] + '_'+ plot_type  + '_R' + str(R_set[0]).replace('.', '')+ '_' + 'tau_sigma_N.png')

        plot_type = 'curve'
        N_set = [9, 100, 900, 2500, 10000]
        sigma_set = sigma_set7
        ticksize = 15
        fig = tau_all(dynamics_set[index], N_set, sigma_set, R_set, c_set[index], arguments, bins, criteria, endpoint, fit, powerlaw, plot_type)
        fig.savefig(des + dynamics_set[index] + '_'+ plot_type + '_' + 'tau_sigma_N.png')

def bifur_may():
    dynamics = 'harvestmay'
    bifurcation_file = '../data/' + dynamics + str(degree) + '/' + 'bifurcation.csv'
    data = np.array(pd.read_csv(bifurcation_file, header=None).iloc[:, :])
    # extract data of bifurcation parameter and fixed points 
    c = data[0]
    stable = data[1:3].transpose()
    unstable = data[-1]
 
    bifur_index = np.where(unstable != 0)[0]
    bifur_c = c[bifur_index]
    bifur_stable = stable[bifur_index]
    bifur_one_stable1 = stable[bifur_index[0]-7:bifur_index[0]]
    bifur_one_stable2 = stable[bifur_index[-1]:bifur_index[-1] + 7]
    bifur_one_c1 = c[bifur_index[0]-7:bifur_index[0]]
    bifur_one_c2 = c[bifur_index[-1]:bifur_index[-1] + 7]

    bifur_unstable = unstable[bifur_index]
    unstable_positive = unstable[unstable != 0]
    bifur_together = np.hstack((bifur_one_stable1[:, 0], bifur_stable[:, 1], unstable_positive[::-1], bifur_stable[:, 0], bifur_one_stable2[:, 0] ))
    c_together = np.hstack((bifur_one_c1, bifur_c, bifur_c[::-1], bifur_c, bifur_one_c2))

    fig = plt.figure()
    ax = plt.subplot()
    # bifurcation function
    x = bifur_together
    y = c_together
    # minima and maxima of bifurcation 
    minima = bifur_index[-1]
    maxima = bifur_index[0]
    y_min = y[minima]
    y_max = y[maxima]
    # plot bifurcation curves with three sections 
    plt.plot(np.hstack((bifur_one_c1, bifur_c)), np.hstack((bifur_one_stable1[:, 0], bifur_stable[:, 1])),color='k')
    plt.plot( bifur_c, unstable_positive, '--', color='k')
    plt.plot(np.hstack((bifur_c, bifur_one_c2)), np.hstack((bifur_stable[:, 0], bifur_one_stable2[:, 0])), color='k')
    plt.plot(bifur_c[0] * np.ones(10), np.linspace(0, bifur_stable[0, 1],10), '--',  color='purple')
    plt.plot(bifur_c[-1] * np.ones(10), np.linspace(0, bifur_stable[-1, 1],10), '--',  color='purple')

    
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height
    plt.axis('off')
    
    ax.arrow(0.16, 0, 0, 0.9, fc='k', ec='k', width = 0.001/3, lw = 0.1, head_width=0.03/0.9*0.12* height/width, head_length=0.008*0.9/0.12 * width/height, overhang = 0.3, length_includes_head= True, clip_on = False) 
    ax.arrow(0.16, 0, 0.12, 0., fc='k', ec='k', width = 0.001, lw = 1, head_width=0.03, head_length=0.008, overhang = 0.3, length_includes_head= True, clip_on = False) 


    ax.text(0.15, 0.8, '$x$',size=fs)
    ax.text(0.27, -0.1, '$\\beta$',size=fs)
    ax.text(y_min, -0.4, '$\\beta_{c1}$',size=fs)
    ax.text(y_max, -0.4, '$\\beta_{c2}$',size=fs)
    plt.xlabel('$\\beta$', fontsize=22)
    plt.ylabel('$x$',fontsize=22)
    plt.show()

def single_mutual(sigma_set, N=1):
    """TODO: Docstring for single_mutual.

    :sigma: TODO
    :N: TODO
    :returns: TODO

    """
    dynamics = 'mutual'
    degree = 4
    c = 4
    plt.rc('text', usetex=True)
    plt.rc('font', family='arial', weight='bold')
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams['axes.labelweight'] = 'bold'
    tau = np.zeros(np.size(sigma_set))
    for sigma, i in zip(sigma_set, range(np.size(sigma_set))):
        des = '../data/' + dynamics + str(degree) + '/size' + str(N) + '/c' + str(c) + '/strength=' + str(sigma) + '/'
        plot_range = [0, 10000]
        bins = np.logspace(np.log10(plot_range[0] + 0.1), np.log10(plot_range[1]), plot_range[-1] + 1)
        criteria = 1
     
        tau[i] = tau_combine(des, bins, criteria)
    plt.semilogy(1/sigma_set**2, tau, 'o', markersize = 10, color='tab:red')
    z = np.polyfit(1/sigma_set**2, np.log(tau), 1, full=True)
    k, b = z[0]
    error = z[1]
    plt.semilogy(1/sigma_set**2, np.exp(k * 1/sigma_set**2) *np.exp(b), '--', linewidth = lw, alpha = alpha-0.2, color='k')
    plt.subplots_adjust(left=0.15, right=0.99, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('lifetime $\\langle \\tau \\rangle$', fontsize=fs)
    plt.xlabel('$1/\\sigma^2$', fontsize=fs)

    plt.savefig("../summery/F1d.svg", format="svg") 
    plt.show()

def bifur_nonlinear():
    fig = plt.figure()
    ax = plt.subplot()
    # bifurcation function
    c = 5
    x1= 1
    x2 = 2
    x3 =4
    f = lambda x: c * (x-x1) * (x-x2) * (x-x3)
    x = np.arange(0.4, 4.3, 0.001)
    y = f(x)
    # minima and maxima of bifurcation 
    minima = argrelextrema(y, np.less)[0][0]
    maxima = argrelextrema(y, np.greater)[0][0]
    y_min = y[minima]
    y_max = y[maxima]
    minima_index = np.where(abs(y - y_min)< 1e-1)[0]
    maxima_index = np.where(abs(y - y_max)< 1e-1)[0]
    minima_two = minima_index[minima_index < maxima]
    maxima_two = maxima_index[maxima_index > maxima]

    # plot bifurcation curves with three sections 
    plt.plot(y[:maxima], x[:maxima], color='r')
    plt.plot(y[maxima:minima], x[maxima:minima], '--', color='y')
    plt.plot(y[minima:], x[minima:], color='g')
    # plt.plot(y_min * np.ones(10), np.linspace(x[minima_two], x[minima],10), '--',  color='k')
    plt.plot(y_min * np.ones(10), np.linspace(0, x[minima],10), '--',  color='k')
    # plt.plot(y_max * np.ones(10), np.linspace(x[maxima_two], x[maxima],10), '--',  color='k')
    plt.plot(y_max * np.ones(10), np.linspace(0, x[maxima_index[-1]],10), '--',  color='k')

    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height
    plt.axis('off')
    
    ax.arrow(f(x[0]-0.05), 0, (f(x[-1]+0.05) - f(x[0]-0.05)), 0., fc='k', ec='k', lw = 1, head_width=0.1, head_length=1, overhang = 0.3, length_includes_head= True, clip_on = False) 
    ax.arrow(f(x[0]-0.05), 0, 0, x[-1]-x[0] + 1, fc='k', ec='k', lw = 1, head_width=0.1/(x[-1]-x[0] + 1) * (f(x[-1]+0.05) - f(x[0]-0.05))* height/width, head_length=1*(x[-1]-x[0] + 1) / (f(x[-1]+0.05) - f(x[0]-0.05)) * width/height, overhang = 0.3, length_includes_head= True, clip_on = False) 


    ax.text(f(x[-1]), -0.4, '$\\gamma$',size=fs)
    ax.text(f(x[0] - 0.09), x[-1], '$x$',size=fs)
    ax.text(y_min, -0.4, '$\\gamma_{c_1}$',size=fs)
    ax.text(y_max, -0.4, '$\\gamma_{c_2}$',size=fs)
    plt.xlabel('$\\gamma$', fontsize=22)
    plt.ylabel('$x$',fontsize=22)
    plt.show()

def illustrate_selfdynamics():
    """TODO: Docstring for illustrate_selfdynamics.

    :arg1: TODO
    :returns: TODO

    """

    N = 10
    G=nx.grid_2d_graph(N,N)
    pos = dict( (n, n) for n in G.nodes() )
    labels = dict( ((i, j), i + (N-1-j) * N +1) for i, j in G.nodes() )
    nx.draw_networkx(G, pos=pos, node_size=300, node_color=[(150/250, 200/250, 250/250)], labels=labels, alpha=1)

    #plt.title(f'{N} $\\times$ {N} lattice graph')
    #plt.xlim(-0.5, 2.5)
    #plt.ylim(-0.5, 2.5)
    plt.axis('off')
    plt.savefig("../lattice.svg", format="svg") 
    plt.show()

     

# x_t_illustrate(u
#landscape()
#bifur_illu_two()
# bifurcation('eutrophication', 20, 0.3, 1)
# x_t_beta
plot_list = np.arange(12)
plot_list = [3, 4, 5]
index = 2

sigma_set3 = [0.08, 0.09, 0.1]
sigma_set4 = [0.06]
sigma_set5 = [0.1]
sigma_set7 = [0.055, 0.06, 0.07, 0.08, 0.09, 0.1]
sigma_set6 = [0.055, 0.06, 0.07, 0.08, 0.09, 0.1]


sigma_set1 = [0.01]
sigma_set2 = [0.01, 0.009, 0.008]
sigma_set3 = [0.01, 0.02, 0.03]
sigma_set4 = [0.007]
sigma_set5 = [0.1]
sigma_set7 = [0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2]
sigma_set6 = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.2, 0.5, 1]
sigma_set6 = [0.055, 0.06, 0.07, 0.08, 0.09, 0.1, 0.3, 0.5]
sigma_set6 = [0.007, 0.008, 0.009, 0.01, 0.02, 0.05, 0.1]


realization_index2 = 1000
realization_index1 = 1000
sigma_set = [sigma_set1, sigma_set2, sigma_set3, sigma_set4, sigma_set5, sigma_set6, sigma_set7]
#plot_figure_table(plot_list, index, sigma_set, realization_index1, realization_index2 )



illustrate_selfdynamics()

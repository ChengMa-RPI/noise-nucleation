import os
import numpy as np
import ast
import shutil

def search_dir(des):
    sub_dir = []
    for _, dirnames, _ in os.walk(des):
        sub_dir.extend(dirnames)
        break  # only add the first level directory
    des_sub = []
    for i in sub_dir:
        des_sub.append(des + i + '/')
    return des_sub

def all_dir(degree):
    """list all folder any level of '../data/gridi{degree}/'

    :degree: fold name
    :returns: all folders 

    """
    des = f'../data/grid{degree}/'
    des_sub = search_dir(des)
    des_subsubsub = []
    for i in des_sub:
        des_subsub = search_dir(i)
        for j in des_subsub:
            des_subsubsub.extend(search_dir(j))
    return des_subsubsub

def extract_info(des):
    seperate = ['size', 'beta', 'strength=', 'T=', '/']
    evalulate = []
    for i in range(3):
        evalulate.append(ast.literal_eval(des[des.find(seperate[i]) + len(seperate[i]): des.rfind(seperate[i+1]) -1]))
    evalulate.append(ast.literal_eval(des[des.find(seperate[3]) + len(seperate[3]): des.rfind(seperate[4]) ]))

    return evalulate

def rm_rho_lifetime(degree):
    """TODO: Docstring for rm_rho_lifetime.

    :degree: TODO
    :returns: TODO

    """
    des_subsubsub = all_dir(degree)
    for i in des_subsubsub:
        filename_set = ['lifetime.csv', 'rho.csv']
        # filename_set = ['heatmap']
        for j in filename_set:
            filename = i + j
            if os.path.exists(filename):
                os.remove(filename)
                # shutil.rmtree(filename)

def rm_data(des, file_type, realization):
    """TODO: Docstring for rm_evolution_data.

    :des: TODO
    :file_type: TODO
    :realization: TODO
    :returns: TODO

    """
    if file_type == 'realization':
        for i in range(realization):
            os.remove(des + file_type + f'{i}.h5')
    elif file_type == 'rho' or 'lifetime':
        os.remove(des + file_type + '.csv')

    return None

def file_range(des):
    """find file 'realization.h5' range

    :des: destination where files are stored
    :returns: realization range, start_index and end_index

    """
    realization = []
    for filename in os.listdir(des):
        if filename.endswith('.h5'):
            realization.append(ast.literal_eval(filename[filename.find('realization') + len('realization'): filename.rfind('.h5') ]))

    if np.size(realization) == 0:
        realization_start = 0
        realization_end = -1
    else:
        realization_start = np.min(realization)
        realization_end = np.max(realization)

    if realization_end - realization_start - np.size(realization) + 1 != 0:
        print('realization not continuous!')
        return None
    else:
        return realization_start, realization_end 
'''
degree = 4
beta_fix = 4
N_set = [25, 100, 400]
sigma_set = [0.1]
T = 100

file_type = 'lifetime'
realization = 100
for N in N_set:
    for sigma in sigma_set:
        des = f'../data/grid{degree}/size{N}/beta{beta_fix}/strength={sigma}_T={T}/'
        rm_data(des,file_type, realization)
'''

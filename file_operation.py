import os
import numpy as np
import ast
import shutil
import pandas as pd
import csv

def search_dir(des):
    sub_dir = []
    for _, dirnames, _ in os.walk(des):
        sub_dir.extend(dirnames)
        break  # only add the first level directory
    des_sub = []
    for i in sub_dir:
        des_sub.append(des + i + '/')
    return des_sub

def all_dir(des):
    """list all folder any level of '../data/gridi{degree}/'

    :degree: fold name
    :returns: all folders 

    """
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

def csv_convert_h5(des, T):
    """TODO: Docstring for csv_convert_h5.

    :filename: TODO
    :returns: TODO

    """
    t_size = T*100
    with open(des+'rho.csv') as f:
        f_csv = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC) # header=None, so don't need to next(f_csv)
        for row in f_csv:
            data = pd.DataFrame(row)
            data.to_hdf(des +'rho.h5', key='data', mode='a', append=True)


def remove_dir(path):
    """remove directory

    :des: TODO
    :returns: TODO

    """
    if os.path.isfile(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)

def clear_file(N_set, sigma_set, R_set, des_type, realization):
    for N in N_set:
        for sigma in sigma_set:
            for R in R_set:
                if R == 0.2:
                    des = f'../data/' + dynamics + f'{degree}/size{N}/c{c}/strength={sigma}/'
                else:
                    des = f'../data/' + dynamics + f'{degree}/size{N}/c{c}/strength={sigma}_R{R}/'
                des_file = os.listdir(des+des_type)
                for filename in des_file:
                    if des_type == 'high/':
                        realization_num = ast.literal_eval(filename[filename.find('realization') + len('realization'): filename.find('.')])
                    else:
                        realization_num = ast.literal_eval(filename[filename.find('realization') + len('realization'): filename.find('_')])

                    if realization_num in realization:
                        remove_dir(des+ des_type+ filename)
                        print(realization_num)

degree = 4
beta_fix = 4
c_set = [4, 1.8, 6, 2.7]
dynamics_set = ['mutual', 'harvest', 'eutrophication', 'vegetation']
index = 2
c = c_set[index]
dynamics = dynamics_set[index]
N_set = [100]
sigma_set = [0.07]
R_set = [0.2]
des_type = 'evolution/'
realization = np.arange(1000) + 4000

clear_file(N_set, sigma_set, R_set, des_type, realization)

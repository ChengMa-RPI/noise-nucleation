import os
import ast

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
        for j in filename_set:
            filename = i + j
            if os.path.exists(filename):
                os.remove(filename)

rm_rho_lifetime(4)

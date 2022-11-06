import numpy as np 
import matplotlib.pyplot as plt 

def  f(x, A, x1, x2, x3):
    """artificial dynamics

    :x: TODO
    :A: TODO
    :x1: TODO
    :x2: TODO
    :x3: TODO
    :returns: TODO

    """
    f = A * x * (x - x1) * (x - x2) * (x -x3)
    return f

A = 2
x1 = 1
x2 = 1.1 
x3 = 4

x = np.arange(0, 5, 0.1)
y = f(x, A, x1, x2, x3)

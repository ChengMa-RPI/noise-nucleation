import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy import linalg as LA
fs = 18
def network_eigenvalue(N):
    num_col = int(np.sqrt(N))
    G = nx.grid_graph(dim=[num_col,int(N/num_col)], periodic=True)
    A = np.array(nx.adjacency_matrix(G).todense())
    L = A - 4 * np.diag(np.ones(N))
    w, v = LA.eig(L)
    w_r = np.real(w)
    second_eigenvalue = np.sort(w_r)[-2]
    return second_eigenvalue

eigen_set = []
N_set = np.arange(3, 51)
for i in N_set:
    N = i**2
    eigen_set.append( network_eigenvalue(N))
eigen_set = np.array(eigen_set)
plt.plot(N_set**2, -1/eigen_set, '--*')
plt.xlabel('network size $N$', fontsize=fs)
plt.ylabel('$-\\frac{1}{\\lambda}$', fontsize=fs)
plt.title('Second largest eigenvalue of $L$', fontsize = fs)

plt.show()




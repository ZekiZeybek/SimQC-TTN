import numpy as np
import copy
import string
import opt_einsum as oe
from collections import deque

def rand_gate():
    i,j,k,l = 2,2,2,2
    real_part = np.random.randn(i,j,k,l)
    imag_part=  np.random.randn(i,j,k,l)
    M_ijkl = real_part + 1j * imag_part
    M_ij_kl = np.reshape(M_ijkl, (i*j, k*l))
    Q, _ = np.linalg.qr(M_ij_kl)
    M_ijkl = np.reshape(Q, (i,j,k,l))
    return M_ijkl

def iso_ten(rank_tuple,m):
    real_part = np.random.randn(*rank_tuple)
    imag_part=  np.random.randn(*rank_tuple)
    A = real_part + 1j * imag_part
    Q,_ = qr_dec(A,m)
    return Q

def rand_norm_ten(rank_tuple):
    real_part = np.random.randn(*rank_tuple)
    imag_part=  np.random.randn(*rank_tuple)
    A = real_part + 1j * imag_part
    A = A/np.linalg.norm(A)
    return A

def qr_dec(T,m):
    T = np.moveaxis(T,m,-1)
    shape = T.shape
    T = T.reshape(-1,T.shape[-1])
    Q,R = np.linalg.qr(T)
    Q = Q.reshape(shape)
    Q = np.moveaxis(Q,-1,m)
    return Q,R

def is_T_QR(T,Q,R,m):
    letters = list(string.ascii_lowercase)
    Q_indices = letters[:len(Q.shape)]

    R_indices = letters[len(Q.shape):len(Q.shape)+len(R.shape)]

    R_indices[0]=Q_indices[m]

    res_indices=copy.deepcopy(Q_indices)
    res_indices[m] = R_indices[1]

    Q_str = ''.join(Q_indices)
    R_str = ''.join(R_indices)
    res_str = ''.join(res_indices)
    ein_str = f"{Q_str},{R_str}->{res_str}"
    print('T=QR',ein_str)
    x = oe.contract(ein_str,Q,R)
    print("T=QR? ",np.allclose(x, T, atol=1e-10))

def is_Q_isom(Q,m):

    if len(Q.shape )==2 and m>1:
        return 0

    letters = list(string.ascii_lowercase)
    Q_indices = letters[:len(Q.shape)]
    Q_conj_indices = copy.deepcopy(Q_indices)
    Q_conj_indices[m] = letters[len(Q.shape)]
    res_indices = []
    res_indices.extend([Q_indices[m],Q_conj_indices[m]])
    
    Q_str = ''.join(Q_indices)
    Q_conj_str = ''.join(Q_conj_indices)
    res_str = ''.join(res_indices)
    ein_str = f"{Q_str},{Q_conj_str}->{res_str}"

    x = oe.contract(ein_str,Q,np.conjugate(Q))
    idt =  np.eye(x.shape[0])
    print("Id=QQconj? ",ein_str)

    print("Id=QQconj? ",np.allclose(x, idt, atol=1e-10))

def all_shortest_paths(graph):
    """
    Computes the shortest path from every node to every other reachable node in the graph.
    """
    
    all_paths = {}
    for start in graph:
        paths = {start: [start]}
        queue = deque([start])
        
        while queue:
            current = queue.popleft()            
            for neighbor in graph.get(current, []):
                if neighbor not in paths:
                    paths[neighbor] = paths[current] + [neighbor]
                    queue.append(neighbor)
                    
        all_paths[start] = paths
    
    return all_paths
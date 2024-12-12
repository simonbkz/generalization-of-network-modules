import time
import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
from jax import jit, grad
import jax.numpy as jnp

def svs(A, U, VT, num_svds, k2):
    # Gets singular values of A using A's singular vectors U and VT (keeps singular values aligned for plotting, unlike np.linalg.svd)
    S = np.dot(U.T, np.dot(A,VT.T))
    small_length = np.min([S.shape[0], S.shape[1]])
    s = np.array([S[i,i] for i in range(small_length)])
    if k2 == 0:
        s = np.sort(s)[::-1]
    s = s[:num_svds]
    return U, s, VT

#TODO: test this and reference the literature if everything is code correctly
#TODO: add following component and ensure it works with the above 
#TODO: how are we separating dense from sparse network

def gen_binary_patterns(num_features):
    # Generates every unique binary digit possible with num_feature bits
    data = np.ones((2**num_features, num_features))*-1.0
    for j in np.arange(0, num_features, 1):
        step = 2**(j+1)
        idx = [list(range(i,i+int(step/2))) for i in np.arange(int(step/2),2**num_features,step)]
        idx = np.concatenate(idx)
        data[idx,j] = 1
    data = np.flip(data, axis=1)
    return data
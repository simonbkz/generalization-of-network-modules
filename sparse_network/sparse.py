import time
import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
from jax import jit, grad
import jax.numpy as jnp

def svs(A, U, VT, num_svds, k2):
    # Gets singular values of A using A's singular vectors U and VT (keeps singular values aligned for plotting, unlike np.linalg.svd)
    # input params
    # : A --> matrix needed to decompose
    # : U --> left orthogonal matrix
    # : VT --> right orthogonal matrix
    # : num_svds --> number of singular values needed to retain
    # : k2 --> not yet clear
    S = np.dot(U.T, np.dot(A,VT.T)) #SDV
    small_length = np.min([S.shape[0], S.shape[1]]) #lower-rank sub-structure, from S return fewest columns or rows and that is the lowest rank
    s = np.array([S[i,i] for i in range(small_length)]) # construct a list of singular values, sorted from highest to lowest
    if k2 == 0:
        s = np.sort(s)[::-1] #only sorting eigenvalues if k2 == 0, otherwise not going to sort it
    s = s[:num_svds] #under what conditions are these not sorted, k2 == 1 and means will not be sorted
    return U, s, VT

#TODO: test this and reference the literature if everything is code correctly (svs function make sense)
#TODO: add following component and ensure it works with the above 
#TODO: how are we separating dense from sparse network

def gen_binary_patterns(num_features):
    # This generates compositional features 
    # only applicable for compositional features, non compositional we take all features
    # assumption is num rows will always be 2**columns (definition of compositional input feature space)
    data = np.ones((2**num_features, num_features))*-1.0 #generate data placeholder matrix, ensure all entries are -1
    # below methodology is adopted from where (paper ?)
    for j in np.arange(0, num_features, 1):
        step = 2**(j+1)
        idx = [list(range(i,i+int(step/2))) for i in np.arange(int(step/2),2**num_features,step)]
        idx = np.concatenate(idx)
        data[idx,j] = 1
    data = np.flip(data, axis=1)
    return data

# TODO: we will need to unpack the dynamic_formular which generates the data that we need
def dynamic_formular(n1,n2, k1, k2, r, a_init, num_time_steps, num_svds):
    """
    return predictions of training dynamics of the linear network using formular for the Singular Vector Decomposition for the first generation of training
    sets up some constant matrices consistent with those used in the appendix of specialisation of network modules for the svd equations
    """
    # nx - number of bits in compositional matrix, rows will be 2**nx
    n1 = 3 #num sys inputs, compositional
    n2 = 1 #num sys outputs, compositional
    k1 = 3 #num non-sys inputs
    k2 = 1 #num non-sys outputs

    # extracting compositional matrix
    # understand from literature why we are reversing the order of data to create Y and X
    A = np.dot(np.dot(Y[:n2], X[:n1].T).T, np.dot(Y[:n2],X[:n1].T)) #this is 
    B = np.dot(np.dot(Y[:n2].T, Y[:n2]), X[:n1].T)
    C = np.dot(np.dot(X[:n1], X[:n1].T))
    AX = np.dot(np.dot(X[:n1], X[:n1].T).T, np.dot(X[:n1], X[:n1].T))

    # Now we are doing the matrices for non-compositional features (k1, k2)

    return sv_trajectory_plots, predicted_sys_norm, predicted_non_sys_norm, quad_norms, U_preds, V_T_preds, sv_inidces

# TODO: Questions to ask
# TODO: Why are we specifying number of singular values to return if we have a function we use to return lower rank of the matrix
# TODO: where is the methodology of generating binary patterns adopted from
# TODO: Why are we reversing the order of X and Y from the data

if __name__ =='__main__':
  
      # Data Hyper-parameters
  n1 = 3 #n1 num sys inputs
  n2 = 1 #n2 num sys outputs
  k1 = 3 #k1 num nonsys reps input
  k2 = 1  #k2 num nonsys reps output
  r = 1 #r scale

  # sample values 
  A = np.array([[1,2,3],[4,5,6],[7,8,9]])
  U = A*A.T
  VT = (A.T*A).T
  num_svds = k2

  U, s, VT = svs(A, U, VT, num_svds, k2)
  print(f"U is: {U}, s is: {s}, VT is: {VT}")
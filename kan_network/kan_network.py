import time
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from jax import jit
import jax.numpy as jnp
import torch
import torch.nn as nn

def piece_wise_f(x):
  if x < 0:
    return -2*x
  if x < 1:
    return -0.5*x
  return 2*x - 2.5

def update(params, batch):
  #function to update paramaters of the network
  # weights replaced by function that is learnable

  return params

def gen_binary_patterns(num_features):
    # This generates compositional features 
    # only applicable for compositional features, non compositional features will be on the diagonal (yes/no) and not featured in this method
    data = np.ones((2**num_features, num_features))*-1.0 #generate data placeholder matrix, ensure all entries are -1
    # below methodology is adopted from where (paper ?)
    for j in np.arange(0, num_features, 1):
        step = 2**(j+1)
        idx = [list(range(i,i+int(step/2))) for i in np.arange(int(step/2),2**num_features,step)]
        idx = np.concatenate(idx)
        data[idx,j] = 1
    data = np.flip(data, axis=1)
    return data

#TODO: define the update function, does the function update relu function or the weights
#TODO: sparsity of the network
#TODO: do we have both the weights and learnable functions on the edges?
#TODO: grid for piecewise function being the same, what does this mean
#TODO: define mapping of KANs

if __name__ == '__main__':

  r"""We will maintain same logic as in the network specialization paper"""
  n1 = 3 #n1 num sys inputs
  n2 = 1 #n2 num sys outputs
  k1 = 3 #k1 num nonsys reps input
  k2 = 1  #k2 num nonsys reps output
  r = 1 #r scale
  a_init = 0
  num_time_steps = 3
  num_svds = 3
  
  k = 3 # Grid size, number of piece wise defined domains
  inp_size = 5
  out_size = 7
  batch_size = 10
  batch_size_y = 7
  # X = torch.randn(batch_size, inp_size) # Our input
  # Y = torch.randn(batch_size_y, out_size) # Our output

  num_hidden = 50
  layer_sizes = [n1 + k1*2**n1, int(num_hidden), n2 + k2*2**n1]
  step_size = 0.02
  num_epochs = 300
  param_scale = 0.01/float(num_hidden)

  if k2 > 0:
    num_svds = 2**n1
  else:
    num_svds = n2

  #TODO: initializing a 
  a = np.ones(num_svds)
  a[:n1] = a[:n1]*7e-7
  a[n1:] = a[n1:]*7e-7
  traj_real = a

  seed = np.random.randint(0,100000)
  trainings = np.zeros((num_epochs, num_svds, 1))
  predictions = np.zeros((num_epochs, num_svds, 1))
  sys_norms = np.zeros((num_epochs, 1))
  non_norms = np.zeros((num_epochs, 1))
  sys_sys_norms = np.zeros((num_epochs, 1))
  sys_non_norms = np.zeros((num_epochs, 1))
  non_sys_norms = np.zeros((num_epochs, 1))
  non_non_norms = np.zeros((num_epochs, 1))
  preds_sys_norms = np.zeros((num_epochs, 1))
  preds_non_norms = np.zeros((num_epochs, 1))
  preds_sys_sys_norms = np.zeros((num_epochs, 1))
  preds_sys_non_norms = np.zeros((num_epochs, 1))
  preds_non_sys_norms = np.zeros((num_epochs, 1))
  preds_non_non_norms = np.zeros((num_epochs, 1))
  losses = np.zeros((num_epochs,1))

  #create dataset for training, this will give us k1 identity matrices
  X = np.flip(gen_binary_patterns(n1).T, axis = 1)
  for i in range(k1):
      X = np.vstack([X, r*np.eye(2**n1)])

  #create dataset for labels
  Y = np.flip(gen_binary_patterns(n1).T, axis = 1)
  for i in range(k2):
      Y = np.vstack([Y, r*np.eye(2**n1)])

  Y_keep_feat = np.arange(Y.shape[0])
  Y_delete = np.random.choice(n1, n1 - n2, replace = False)
  Y_keep_feat = np.delete(Y_keep_feat, Y_delete)
  Y = Y[Y_keep_feat]
  print("input data is: \n",X)
  print("initial labels are: \n", Y)
  batch_size = X.shape[1]
  start_labels = np.copy(Y)
  step_size = 0.02

  linear = nn.Linear(inp_size*k, out_size)  # kan architecture when k > 1 as each spline will just be an activation function where we can customize different activation functions for different parts of the input space

  repeated = X.unsqueeze(1).repeat(1,k,1) #we are repeating the input k times, transforming KAN to MLP
  shifts = torch.linspace(-1, 1, k).reshape(1,k,1) #c constants for each piecewise function

  shifted = repeated + shifts
  # grid is shared, we use the same relu for C2 and C3
  intermediate = torch.cat([shifted[:,:1,:], torch.relu(shifted[:,1:,:])], dim=1).flatten(1) #investigate more here

  outputs = linear(intermediate) #propagate forward, no back propagation yet

#   X = torch.linspace(-2, 2, 100)
#   plt.plot(X, [piece_wise_f(x) for x in X])
#   plt.grid()
#   plt.show()
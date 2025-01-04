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
  #this function will be looking into the shallow strucutre of the network
  #function to update paramaters of the network
  # weights replaced by function that is learnable, which are relu functions for simplicity
  #The following happens for each batch
  #TODO: perform forward pass here
  #TODO: compute the loss
  #TODO: perform backward pass using backpropagation to compute gradients
  #TODO: parameters are then updated using these gradients via optimization algorithm like SGD, Adam, etc
  sigma_xx = (1/batch[0].shape[1]) * np.dot(batch[0], batch[0].T) #is this derived from literature?
  sigma_xy = (1/batch[0].shape[1]) * np.dot(batch[0], batch[1].T)
  #TODO: We need to compute W2W1 input-ouput mapping for KANs
  return params

#TODO: we need to define an update function for sparse KAN networks

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

def create_architecture(X, Y, input_size, output_size, k, num_hidden):
   r"This function creates a random mapping between the edge (relu) to the next neuron. "
  #  linear = nn.Linear(input_size*k, output_size)  # kan architecture when k > 1 as each spline will just be an activation function where we can customize different activation functions for different parts of the input space

   repeated = X.unsqueeze(1).repeat(1,k,1) #we are repeating the input k times, transforming KAN to MLP
   shifts = torch.linspace(-1, 1, k).reshape(1,k,1) #c constants for each piecewise function

   shifted = repeated + shifts
   # grid is shared, we use the same relu for C2 and C3
   X_conf = torch.cat([shifted[:,:1,:], torch.relu(shifted[:,1:,:])], dim=1).flatten(1) #proxy of weights, bias applied to X
   return X_conf

@jit
def predict(linear_l, params):
  preds = linear_l(params)
  return jnp.array(preds)

@jit
def loss(params, batch, linear_l):
   inputs, targets = batch
   preds = predict(linear_l, inputs)
   return np.mean(np.sum(jnp.power((preds - targets)),2), axis = 1)

#TODO: universal approximation theorem underpins the neural network logic, any feedforward neural network can approximated any continuous function under certain conditions 
#TODO: Komogorov Arnold theorem states that any multivariate continuous function can be replicated by adding univariate functions or feeding one into the other 
#TODO: check the difference between the network configs on shallow, and deep networks coded on specialization
#TODO: define the update function, does the function update relu function or the weights, weights are replaced by learnable functions (https://arxiv.org/pdf/2407.11075v4)
#TODO: sparsity of the network
#TODO: We only have learnable functions on edge instead of weights, this is the key difference between KAN and MLP
#TODO: grid for piecewise function being the same, what does this mean
#TODO: define mapping of KANs
#TODO: objective of learnable functions, decide which learnable function is the best between two neurons
#TODO: How are we going to do the exact solution of KANs
#TODO: are we able to specify the function we will be approximating with the KANs
#TODO: 

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
  X_ = torch.randn(batch_size, inp_size) # Our input
  Y_ = torch.randn(batch_size_y, out_size) # Our output

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

  X = create_architecture(X_, Y_, inp_size, out_size, k, num_hidden) # we will update this function for subsequent layers
  #traditionally we would have had, x*w
  # outputs = linear(intermediate) #propagate forward, no back propagation yet

  for epoch in range(num_epochs):
     #We only have two batches
     for batch_start in range(0, X.shape[1], batch_size):
        batch = (X[:,batch_start:batch_start+batch_size], Y[:,batch_start:batch_start+batch_size])

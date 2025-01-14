import time
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from jax import jit
import jax.numpy as jnp
import torch
import torch.nn as nn

def piece_wise_f(x):
  #This is a test function 
  if x < 0:
    return -2*x
  if x < 1:
    return -0.5*x
  return 2*x - 2.5

def svs(A, U, VT, num_svds, k2):
   #This function takes matrix A, left singular vector U, right singular conjugate transpose vector VT, number of singular values num_svds, and number of non-systematic outputs (identity matrices)
   #This function use both U and VT to decompose matrix A into systematic and non-systematic components
   #This function also ensures that we control the number of singular values to return
   S = np.dot(U.T, np.dot(A, VT.T))
   small_length = min(S.shape[0], S.shape[1])
   s = np.array([S[i,i] for i in range(small_length)])
   if k2 == 0: # no identity matrices
    s = np.sort(s)[::-1]
   s = s[:num_svds]
   return s

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
  #TODO: shallow is different to deep network in this update function, see where this is referenced in the paper
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
   #TODO: this function needs to be written in jax instead of pytorch
   #FIXME: we need to get this written in jax instead of pytorch
  #  linear = nn.Linear(input_size*k, output_size)  # kan architecture when k > 1 as each spline will just be an activation function where we can customize different activation functions for different parts of the input space

   repeated = X.unsqueeze(1).repeat(1,k,1) #we are repeating the input k times, transforming KAN to MLP
   shifts = torch.linspace(-1, 1, k).reshape(1,k,1) #c constants for each piecewise function

   shifted = repeated + shifts
   # grid is shared, we use the same relu for C2 and C3
   X_conf = torch.cat([shifted[:,:1,:], torch.relu(shifted[:,1:,:])], dim=1).flatten(1) #proxy of weights, bias applied to X
   return X_conf

#TODO: we need to get the update function for KANs
#TODO: do we use weights or not, where do we apply them?

#Now we want to break the input-output data into systematic and non-systematic components
#We will use the same method dynamic formular
def dynamic_formular(X,Y,n1,n2, k1, k2, r, a_init, num_time_steps, num_svds):
  """
  return predictions of training dynamics of the linear network using formular for the Singular Vector Decomposition for the first generation of training
  sets up some constant matrices consistent with those used in the appendix of specialisation of network modules for the svd equations
  """
  # nx - number of bits in compositional matrix, rows will be 2**nx
  n1 = 3 #num sys inputs, compositional
  n2 = 1 #num sys outputs, compositional
  k1 = 3 #num non-sys inputs
  k2 = 1 #num non-sys outputs

  #coding X and Y for ease of testing in this 
  # X = np.flip(gen_binary_patterns(n1).T, axis=1) #flip to maintain original structure of dataset
  # for i in range(k1):
  #     X = np.vstack([X, r*np.eye(2**n1)])

  # # Create Dataset labels
  # Y = np.flip(gen_binary_patterns(n1).T, axis=1)
  # for i in range(k2):
  #     Y = np.vstack([Y, r*np.eye(2**n1)])
  # Y_keep_feat = np.arange(Y.shape[0])
  # Y_delete = np.random.choice(n1, n1-n2, replace=False)
  # Y_keep_feat = np.delete(Y_keep_feat, Y_delete)
  # Y = Y[Y_keep_feat]
  # print("Input Data: \n", X)
  # print("Initial Labels: \n", Y)
  # batch_size = X.shape[1]
  # start_labels = np.copy(Y)
  # step_size = 0.02
  # tau = 1/(X.shape[1]*step_size)

  # num_svds = 2**n1
  # a = np.ones(num_svds)
  # a[:n1] = a[:n1]*7e-7
  # a[n1:] = a[n1:]*8e-7

  # below matrices come from definitions derived from the paper
  A = np.dot(np.dot(Y[:n2], X[:n1].T).T, np.dot(Y[:n2], X[:n1].T)) #this is 
  B = np.dot(np.dot(Y[:n2].T, Y[:n2]), X[:n1].T)
  C = np.dot(np.dot(X[:n1].T, X[:n1]),X[:n1].T)
  AX = np.dot(np.dot(X[:n1], X[:n1].T).T, np.dot(X[:n1], X[:n1].T))

  if k1 > 0 and k2 > 0:
      XZ = (1/(k1*k2))**0.5*np.eye(2**n1) - (1/((k1*k2)**0.5*2**n1))*np.dot(X[:n1].T, X[:n1])
      T, H, P = np.linalg.svd(XZ)
      T = (1/(k2))**0.5*T[:, :(2**n1-n1)+3]
      P = (1/(k1))**0.5*P[:2**n1-n1]
  else:
      XZ = (1/(k1))**0.5*np.eye(2**n1) - (1/((k1)**0.5*2**n1))*np.dot(X[:n1].T,X[:n1])
      T, H, P = np.linalg.svd(XZ)
      T = (1/(k2))**0.5*T[:, :2**n1-n1]
      P = (1/(k1))**0.5*P[:2**n1-n1]
  # predicting the input-output covariance and input SV formula and storing in diagonal matrix
  # to get correct order Singular Values
  sv1 = ((k1*r**2+2**n1)*(k2*r**2+2**n1)/2**(6*n1))**0.5
  sv2 = ((k1*r**2+2**n1)*(k2*r**2)/2**(2*n1))**0.5
  sv3 = ((k1*k2)**0.5*r**2)/(2**n1)
  sxv1 = (k1*r**2+2**n1)/2**(n1)
  sxv3 = (k1*r**2)/(2**n1)

  sv_preds = sv1*A + sv2*(np.eye(n1) - (1/2**(2*n1))*A)
  sxv_preds_inv = sxv1*np.eye(n1)

  sxv_preds_inv = [1/sxv_preds_inv[i,i] for i in range(sxv_preds_inv.shape[0]) if sv_preds[i,i] > 0]
  sv_preds = [sv_preds[i,i] for i in range(sv_preds.shape[0]) if sv_preds[i,i] > 0]

  if k2 > 0:
      for _ in range(2**n1 - n1): sv_preds.append(sv3)
      for _ in range(2**n1 -n1): sxv_preds_inv.append(1/sxv3)
  #When you have non-compositional structure, comp structure will co-exist with non-comp struct
  unique_svs = np.unique(sv_preds[:n1])[::-1]
  SV11_i = np.argmax(sv_preds == unique_svs[0])
  if n1 > n2 and n2 > 0 and k2 > 0:
      SV12_i = np.argmax(sv_preds == unique_svs[1])
  else:
      SV21_i = np.argmax(sv_preds = unique_svs[0])
  SV21_i = -1
  sv_indices = [SV11_i, SV12_i, SV21_i]

  # predicting the left singular vectors. Used to get correctly ordered SVs from the simulated network mapping
  U1 = (1/2**n1)**0.5*(1/(k2*r**2+2**n1))**0.5
  U2 = (1/2**(3*n1))**0.5*((r**2)/(k2*r**2+2**n1))**0.5
  U3 = (1/2**(3*n1))**0.5*((r**2)/(k2*r**2))**0.5

  U_preds = U1*np.dot(Y[:n2],X[:n1].T)
  for _ in range(k2):
      U_preds = np.vstack([U_preds, U2*B + U3*(C-B)])

  if k2 > 0:
      U_nonsys_part = np.zeros((n2, 2**n1))
      for _ in range(k2):
          U_nonsys_part = np.vstack([U_nonsys_part, T])
      U_preds = np.hstack([U_preds, U_nonsys_part])
      U_preds = np.hstack([U_preds, np.zeros((U_preds.shape[0], n2+(k2*2**n1) - (2**n1)))])

  #predicting the transpose right singular vectors. Used to get correctly ordered SVs from the simulated network mapping
  VT1 = ((2**n1)/(k1*r**2+2**n1))**0.5
  VT2 = ((r**2)/((2**n1)*(k1*r**2+2**n1)))**0.5

  V_T_preds = VT1*np.eye(n1)
  for _ in range(k1):
      V_T_preds = np.hstack([V_T_preds, VT2*X[:n1]])

  V_T_nonsys_part = np.zeros((2**n1 - n1, n1))
  for _ in range(k1):
      V_T_nonsys_part = np.hstack([V_T_nonsys_part, P])

  V_T_preds = np.vstack([V_T_preds, V_T_nonsys_part])
  V_T_preds = np.vstack([V_T_preds, np.zeros((n1+(k1*2**n1) - (2**n1), V_T_preds.shape[1]))])

  #getting SVs in numpy arrays, setting up time indices array and exponential components of SV dynamics formula
  s = np.array(sv_preds)
  d_inv = np.array(sxv_preds_inv)
  taus = (np.arange(0,num_time_steps,1).reshape(num_time_steps,1)/tau)
  exp_s = np.exp(-2*s[:num_svds]*taus)

  #predict SV trajectories using dynamic formula
  numerator = s*d_inv
  denominator = 1 - (1 - (s*d_inv/a_init))*exp_s
  sv_trajectory_plots = np.zeros(denominator.shape)
  num_sig_svs = numerator[numerator > 0 ].shape[0]
  sv_trajectory_plots[:,:num_sig_svs] = numerator[:num_sig_svs].reshape(1,num_sig_svs)/denominator[:,:num_sig_svs]

  #predict Frobenius norm trajectories using SV dynamics (partitioned by output only, i.e. full input used for mappings)
  predicted_sys_norm = ((n2*2**n1)/(k2*r**2+2**n1)*sv_trajectory_plots[:SV11_i]**2)**0.5
  if k2 > 0:
      predicted_non_norm = ((k2*n2*r**2/(k2*r**2+2**n1))*sv_trajectory_plots[:,SV11_i]**2 \
                            + (n1-n2)*sv_trajectory_plots[:,SV12_i]**2 \
                              + (2**n1-n1)*sv_trajectory_plots[:,SV21_i]**2)**0.5
  else:
      predicted_non_norm = np.zeros(predicted_sys_norm)

  # predict Frobenius norm trajectories using SV dynamics (partition by input and output)
  predicted_sys_sys_norm = ((n2*2**(2*n1))/((k1*r**2+2**n1)*(k2*r**2+2**n1))*sv_trajectory_plots[:,SV11_i]**2)**0.5
  predicted_non_sys_norm = ((n2*2**n1*(k1*r**2))/((k1*r**2+2**n1)*(k2*r**2+2**n1))*sv_trajectory_plots[:,SV11_i]**2)**0.5

  if k2 > 0:
      predicted_sys_non_norm = ((k2*n2*r**2*2**n1)/((k1*r**2+2**n1)*(k2*r**2+2**n1))*sv_trajectory_plots[:, SV11_i]**2 \
                                  + ((n1-n2)*2**n1)/(k1*r**2+2**n1)*sv_trajectory_plots[:,SV12_i]**2)**0.5
      
      predicted_non_non_norm = ((k2*n2*r**2*(k1*r**2))/((k1*r**2+2**n1)*(k2*r**2+2**n1))*sv_trajectory_plots[:,SV11_i]**2 \
                                + ((n1-n2)*(k1*r**2))/(k1*r**2+2**n1)*sv_trajectory_plots[:,SV12_i]**2 \
                                  + (2**n1-n1)*sv_trajectory_plots[:,SV21_i]**2)**0.5
  else:
      predicted_sys_non_norm = np.zeros(predicted_sys_norm.shape)
      predicted_non_non_norm = np.zeros(predicted_sys_norm)

  quad_norms = [predicted_sys_sys_norm, predicted_non_sys_norm,predicted_sys_non_norm,predicted_non_non_norm]

  return sv_trajectory_plots, predicted_sys_norm, predicted_non_norm,quad_norms,U_preds,V_T_preds,sv_indices
  #when done with comp and non-comp apply formulas from the paper

@jit
def predict(linear_l, params):
  preds = linear_l(params)
  return jnp.array(preds)

@jit
def loss(batch, linear_l):
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
#TODO: if there are no weights then we can only update the relu functions which are static in our case, thus no update function required


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

  #TODO: initializing a, these should be sufficiently small
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
  tau = 1/(X.shape[1]*step_size) #TODO: why we are doing this calculation

  predictions, preds_sys_norms, preds_non_norms, preds_quad_norms, U, VT, sv_indices =\
        dynamic_formular(X,Y,n1, n2, k1, k2, r, a.reshape(1,num_svds), num_epochs, num_svds)
  
  preds_sys_sys_norms = preds_quad_norms[0]
  preds_non_sys_norms = preds_quad_norms[1]
  preds_sys_non_norms = preds_quad_norms[2] 
  preds_non_non_norms = preds_quad_norms[3]

  # X = create_architecture(X_, Y_, inp_size, out_size, k, num_hidden) # we will update this function for subsequent layers
  #traditionally we would have had, x*w
  # outputs = linear(intermediate) #propagate forward, no back propagation yet

  for epoch in range(num_epochs):
     #We only have two batches
     for batch_start in range(0, X.shape[1], batch_size):
        batch = (X[:,batch_start:batch_start+batch_size], Y[:,batch_start:batch_start+batch_size])
        params = update(params, batch)
        lossr = loss(batch, linear)
        losses[epoch] = lossr
        print(f"Epoch: {epoch}, Loss: {lossr}")



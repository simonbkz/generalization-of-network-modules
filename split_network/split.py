import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
from jax import jit, grad
import jax.numpy as jnp

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=200)
matplotlib.rcParams.update({'font.family':'Times New Roman', 'font.size': 15})
plt.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

def svs(A, U, VT, num_svds, k2):
    # Gets singular values of A using A's singular vectors U and VT (keeps singular values aligned for plotting, unlike np.linalg.svd)
    S = np.dot(U.T, np.dot(A,VT.T))
    small_length = np.min([S.shape[0], S.shape[1]])
    s = np.array([S[i,i] for i in range(small_length)])
    if k2 == 0:
         s = np.sort(s)[::-1]
    s = s[:num_svds]
    return U, s, VT

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

def dynamic_formula(n1, n2, k1, k2, r, a_init, num_time_steps, num_svds):
  # Returns predictions of the training dynamics of the linear net using the formulas for the SVs for the first generation of training.

  # Sets up some constant matrices consistent with those used in the appendix for the SVD equations
  A = np.dot(np.dot(Y[:n2], X[:n1].T).T, np.dot(Y[:n2], X[:n1].T))
  B = np.dot(np.dot(Y[:n2].T,Y[:n2]), X[:n1].T)
  C = np.dot(np.dot(X[:n1].T,X[:n1]),X[:n1].T)
  Ax = np.dot(np.dot(X[:n1], X[:n1].T).T, np.dot(X[:n1], X[:n1].T))

  if k1>0 and k2>0:
    XZ = (1/(k1*k2))**0.5*np.eye(2**n1) - (1/((k1*k2)**0.5*2**n1))*np.dot(X[:n1].T, X[:n1])
    T,H,P = np.linalg.svd(XZ)
    T = (1/(k2))**0.5*T[:, :2**n1-n1]
    P = (1/(k1))**0.5*P[:2**n1-n1]
  else:
    XZ = (1/(k1))**0.5*np.eye(2**n1) - (1/((k1)**0.5*2**n1))*np.dot(X[:n1].T, X[:n1])
    T,H,P = np.linalg.svd(XZ)
    T = T[:, :2**n1-n1]
    P = (1/(k1))**0.5*P[:2**n1-n1]

  # Predicting the input-output covariance and input covariance SV formulas and storing in diagonal-matrix form
  # to get correct order of SVs
  sv1 = ((k1*r**2+2**n1)*(k2*r**2+2**n1)/2**(6*n1))**0.5
  sv2 = ((k1*r**2+2**n1)*(k2*r**2)/2**(2*n1))**0.5
  sv3 = ((k1*k2)**0.5*r**2)/(2**n1)
  sxv1 = (k1*r**2+2**n1)/2**(n1)
  sxv3 = (k1*r**2)/(2**n1)

  sv_preds = sv1*A + sv2*(np.eye(n1) - (1/2**(2*n1))*A)
  sxv_preds_inv = sxv1*np.eye(n1)

  # Flattening SV formulas back into a list now that the correct order is obtained
  sxv_preds_inv = [1/sxv_preds_inv[i,i] for i in range(sxv_preds_inv.shape[0]) if sv_preds[i,i] > 0]
  sv_preds = [sv_preds[i,i] for i in range(sv_preds.shape[0]) if sv_preds[i,i] > 0]

  if k2 > 0:
      for _ in range(2**n1 - n1): sv_preds.append(sv3)
      for _ in range(2**n1 - n1): sxv_preds_inv.append(1/sxv3)

  # Getting unique SV positions in the list (used to calculate the norm trajectories using unique SV trajectories)
  unique_svs = np.unique(sv_preds[:n1])[::-1]
  SV11_i = np.argmax(sv_preds == unique_svs[0])
  if n1 > n2 and n2 > 0 and k2 > 0:
      SV12_i = np.argmax(sv_preds == unique_svs[1])
  else:
      SV12_i = np.argmax(sv_preds == unique_svs[0])
  SV21_i = -1
  sv_indices = [SV11_i, SV12_i, SV21_i]

  # Predicting the left singular vectors. Used to get correctly ordered SVs from the simulated network mapping.
  U1 = (1/2**n1)**0.5*(1/(k2*r**2+2**n1))**0.5
  U2 = (1/2**(3*n1))**0.5*((r**2)/(k2*r**2+2**n1))**0.5
  if k2 > 0:
      U3 = (1/2**(3*n1))**0.5*((r**2)/(k2*r**2))**0.5
      U4 = (1/(k1*k2))**0.25
      U5 = (1/(k1*k2))**0.125*(1/2**(1*n1))

  U_preds = U1*np.dot(Y[:n2],X[:n1].T)
  for _ in range(k2):
      U_preds = np.vstack([U_preds, U2*B + U3*(C-B)])

  if k2 > 0:
      U_nonsys_part = np.zeros((n2, 2**n1 - n1))
      for _ in range(k2):
          U_nonsys_part = np.vstack([U_nonsys_part, T])
      U_preds = np.hstack([U_preds, U_nonsys_part])
      U_preds = np.hstack([U_preds, np.zeros((U_preds.shape[0], n2+(k2*2**n1) - (2**n1)))])

  # Predicting the transposed right singular vectors. Used to get correctly ordered SVs from the simulated network mapping.
  VT1 = (2**n1)**0.5*(1/(k1*r**2+2**n1))**0.5
  VT2 = (1/2**n1)**0.5*((r**2)/(k1*r**2+2**n1))**0.5

  V_T_preds = VT1*np.eye(n1)
  for _ in range(k1):
      V_T_preds = np.hstack([V_T_preds, VT2*X[:n1]])

  V_T_nonsys_part = np.zeros((2**n1 - n1, n1))
  for _ in range(k1):
      V_T_nonsys_part = np.hstack([V_T_nonsys_part, P])

  V_T_preds = np.vstack([V_T_preds, V_T_nonsys_part])
  V_T_preds = np.vstack([V_T_preds, np.zeros((n1+(k1*2**n1) - (2**n1), V_T_preds.shape[1]))])

  # Getting SVs in numpy arrays, setting up time indices array and exponential components of SV dynamics formula
  s = np.array(sv_preds)
  d_inv = np.array(sxv_preds_inv)
  taus = (np.arange(0,num_time_steps,1).reshape(num_time_steps,1)/tau)
  exp_s = np.exp(-2*s[:num_svds]*taus)

  # Predict SV Trajectories using dynamics formula
  numerator = s*d_inv
  denominator = 1 - (1 - (s*d_inv/a_init))*exp_s
  sv_trajectory_plots = np.zeros(denominator.shape)
  num_sig_svs = numerator[numerator > 0].shape[0]
  sv_trajectory_plots[:,:num_sig_svs] = numerator[:num_sig_svs].reshape(1, num_sig_svs)/denominator[:,:num_sig_svs]

  # Predict Frobenius Norm Trajectories using SV dynamics (partitioned by output only, ie: full input used for mappings) 
  predicted_sys_norm = ( (n2*2**n1)/(k2*r**2+2**n1)*sv_trajectory_plots[:,SV11_i]**2)**0.5

  if k2 > 0:
      predicted_non_norm = ( (k2*n2*r**2/(k2*r**2+2**n1))*sv_trajectory_plots[:,SV11_i]**2 \
                               + (n1-n2)*sv_trajectory_plots[:,SV12_i]**2 \
                               + (2**n1-n1)*sv_trajectory_plots[:,SV21_i]**2 )**0.5
  else:
      predicted_non_norm = np.zeros(predicted_sys_norm.shape)

  # Predict Frobenius Norm Trajectories using SV dynamics (partitioned by input and output)
  predicted_sys_sys_norm = ( (n2*2**(2*n1))/((k1*r**2+2**n1)*(k2*r**2+2**n1))*sv_trajectory_plots[:,SV11_i]**2)**0.5
  predicted_non_sys_norm = ( (n2*2**n1*(k1*r**2))/((k1*r**2+2**n1)*(k2*r**2+2**n1))*sv_trajectory_plots[:,SV11_i]**2)**0.5

  if k2 > 0:
      predicted_sys_non_norm = ( (k2*n2*r**2*2**n1)/((k1*r**2+2**n1)*(k2*r**2+2**n1))*sv_trajectory_plots[:,SV11_i]**2 \
                               + ((n1-n2)*2**n1)/(k1*r**2+2**n1)*sv_trajectory_plots[:,SV12_i]**2 )**0.5
      predicted_non_non_norm = ( (k2*n2*r**2*(k1*r**2))/((k1*r**2+2**n1)*(k2*r**2+2**n1))*sv_trajectory_plots[:,SV11_i]**2 \
                               + ((n1-n2)*(k1*r**2))/(k1*r**2+2**n1)*sv_trajectory_plots[:,SV12_i]**2 \
                               + (2**n1-n1)*sv_trajectory_plots[:,SV21_i]**2 )**0.5
  else:
      predicted_sys_non_norm = np.zeros(predicted_sys_norm.shape)
      predicted_non_non_norm = np.zeros(predicted_sys_norm.shape)

  quad_norms = [predicted_sys_sys_norm,predicted_non_sys_norm,predicted_sys_non_norm,predicted_non_non_norm]

  return sv_trajectory_plots, predicted_sys_norm, predicted_non_norm, quad_norms, U_preds, V_T_preds, sv_indices

def init_random_params(scale, layer_sizes, seed):
  # Returns a list of tuples where each tuple is the weight matrix and bias vector for a layer
  np.random.seed(seed)
  return [np.random.normal(0.0, scale, (n, m)) for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

@jit
def predict(params, inputs):
  # Propagate data forward through the network
  return jnp.dot(params[1], jnp.dot(params[0], inputs))

@jit
def loss(params, batch):
  # Loss over a batch of data
  inputs, targets = batch
  preds = predict(params, inputs)
  return np.mean(np.sum(jnp.power((preds - targets),2), axis=1))

def startup(n1,n2,k1,k2,r):
  # Returns number of non-zero SVs and initializes lists for each branch/module of network
  if k2 > 0:
      num_svds = 2**n1
  else:
      num_svds = n2

  # Holds the SV trajectories to be plotted
  trainings = np.zeros((num_epochs, num_svds, 1))
  predictions = np.zeros((num_epochs, num_svds, 1))
  norms = np.zeros((num_epochs, 1))
  preds_norms = np.zeros((num_epochs, 1))
  quad_norm_1 = np.zeros((num_epochs, 1))
  quad_norm_2 = np.zeros((num_epochs, 1))
  pred_quad_norm_1 = np.zeros((num_epochs, 1))
  pred_quad_norm_2 = np.zeros((num_epochs, 1))

  return trainings, predictions, norms, preds_norms, quad_norm_1, quad_norm_2, pred_quad_norm_1, pred_quad_norm_2, num_svds

if __name__ == "__main__":

  # Function which updates model parameters
  def updater(params, batch):
    sigma_xx = (1/batch[0].shape[1])*np.dot(batch[0], batch[0].T)
    sigma_yx = (1/batch[0].shape[1])*np.dot(batch[1], batch[0].T)
    W2W1 = np.dot(params[1], params[0])
    return [params[0] + ((1/tau) * np.dot(params[1].T, sigma_yx - np.dot(W2W1, sigma_xx))),\
            params[1] + ((1/tau) * np.dot(sigma_yx - np.dot(W2W1, sigma_xx), params[0].T))]

  # Data Hyper-parameters
  n1 = 3 #n1 num sys inputs
  n2 = 1 #n2 num sys outputs
  k1 = 3 #k1 num nonsys reps input
  k2 = 1  #k2 num nonsys reps output
  r = 1 #r scale

  # Training hyper-parameters
  num_hidden = 50.0
  sys_layer_sizes = [n1 + k1*2**n1,int(num_hidden), n2]
  non_layer_sizes = [n1 + k1*2**n1,int(num_hidden), k2*2**n1]
  step_size = 0.02
  num_epochs = 300
  param_scale = 0.1/float(num_hidden)
  seed = np.random.randint(0,100000) # can set seed here, for now it is random. The only randomness is in the network init.
  losses = np.zeros((num_epochs, 1))  

  sys_trainings, sys_predictions, sys_norms, preds_sys_norms, sys_sys_norms, sys_non_norms,preds_sys_sys_norms, preds_sys_non_norms, sys_num_svds = startup(n1,n2,k1,0,r)
  non_trainings, non_predictions, non_norms, preds_non_norms, non_sys_norms, non_non_norms,preds_non_sys_norms, preds_non_non_norms, non_num_svds = startup(n1,0,k1,k2,r)

  # Create Dataset training data
  X = np.flip(gen_binary_patterns(n1).T, axis=1)
  for i in range(k1):
      X = np.vstack([X, r*np.eye(2**n1)])

  # Create Dataset labels
  Y = np.flip(gen_binary_patterns(n1).T, axis=1)
  for i in range(k2):
      Y = np.vstack([Y, r*np.eye(2**n1)])
  Y_keep_feat = np.arange(Y.shape[0])
  Y_delete = np.random.choice(n1, n1-n2, replace=False)
  Y_keep_feat = np.delete(Y_keep_feat, Y_delete)
  Y = Y[Y_keep_feat]
  print("Input Data: \n", X)
  print("Initial Labels: \n", Y)
  batch_size = X.shape[1]
  start_labels = np.copy(Y)

  tau = 1/(X.shape[1]*step_size)

  # Init params, plot initial loss and do SVD on the input-output covariance matrix for the initial weights
  sys_params = init_random_params(param_scale, sys_layer_sizes, seed)
  non_params = init_random_params(param_scale, non_layer_sizes, seed) 
  sys_a = np.ones(sys_num_svds)*8e-5
  non_a = np.ones(non_num_svds)*8e-5
  sys_traj_real = sys_a
  non_traj_real = non_a

  # Get ground truth trajectories and save them for plotting at end
  sys_predictions[:,:,i], preds_sys_norms[:,i], _, preds_sys_quad_norms, sys_U, sys_VT, sys_svd_indices =\
          dynamic_formula(n1, n2, k1, 0, r, sys_a.reshape(1,sys_num_svds), num_epochs, sys_num_svds)
  preds_sys_sys_norms[:,i] = preds_sys_quad_norms[0]
  preds_non_sys_norms[:,i] = preds_sys_quad_norms[1]
  non_predictions[:,:,i], _, preds_non_norms[:,i], preds_non_quad_norms, non_U, non_VT, non_svd_indices =\
          dynamic_formula(n1, 0, k1, k2, r, non_a.reshape(1,non_num_svds), num_epochs, non_num_svds)
  preds_sys_non_norms[:,i] = preds_non_quad_norms[2]
  preds_non_non_norms[:,i] = preds_non_quad_norms[3]

  # Run generations of IL. We track the SVs of the input-output covariance matrix from the network at the end of each epoch
  print("Generation: ", i)
  for epoch in range(num_epochs):
      for batch_start in range(0,X.shape[1],batch_size):
          sys_params = updater(sys_params, (X[:,batch_start:batch_start+batch_size],Y[:n2,batch_start:batch_start+batch_size]))
          non_params = updater(non_params, (X[:,batch_start:batch_start+batch_size],Y[n2:,batch_start:batch_start+batch_size]))
      sys_lossr = loss(sys_params, (X,Y[:n2]))
      non_lossr = loss(non_params, (X,Y[n2:]))
      losses[epoch,i] = sys_lossr + non_lossr
      print(sys_lossr + non_lossr)

      # Full input-output mapping of the trained network and its sinular values for each module of the network
      sys_full_map = np.dot(sys_params[1], sys_params[0])
      _, sys_a, _ = svs(sys_full_map, sys_U, sys_VT, sys_num_svds, 0)
      sys_a = sys_a[:sys_num_svds]
      sys_traj_real = np.vstack([sys_traj_real, sys_a])

      non_full_map = np.dot(non_params[1], non_params[0])
      _, non_a, _ = svs(non_full_map, non_U, non_VT, non_num_svds, k2)
      non_a = non_a[:non_num_svds]
      non_traj_real = np.vstack([non_traj_real, non_a])

      # Get systenatic and non-systematic frobenius norms
      sys_norms[epoch,i] = np.linalg.norm(sys_full_map,'fro')
      non_norms[epoch,i] = np.linalg.norm(non_full_map,'fro')

      # Get Quadrant frobenius norms
      sys_sys_norms[epoch,i] = np.linalg.norm(sys_full_map[:,:n1],'fro')
      sys_non_norms[epoch,i] = np.linalg.norm(non_full_map[:,:n1],'fro')
      non_sys_norms[epoch,i] = np.linalg.norm(sys_full_map[:,n1:],'fro')
      non_non_norms[epoch,i] = np.linalg.norm(non_full_map[:,n1:],'fro')
  
  sys_trainings[:,:,0] = sys_traj_real[1:]
  non_trainings[:,:,0] = non_traj_real[1:]

  # Plot Quadrant norms together
  plt.plot(sys_sys_norms, color='red', label=r'$\Omega_x\Omega_y$-Norm')
  plt.plot(non_sys_norms, color='blue', label=r'$\Gamma_x\Omega_y$-Norm')
  plt.plot(sys_non_norms, color='purple', label=r'$\Omega_x\Gamma_y$-Norm')
  plt.plot(non_non_norms, color='navy', label=r'$\Gamma_x\Gamma_y$-Norm')
  plt.plot(preds_sys_sys_norms, color='green', label=r'Predicted $\Omega_x\Omega_y$-Norm', linestyle='dashed')
  plt.plot(preds_non_sys_norms, color='orange', label=r'Predicted $\Gamma_x\Omega_y$-Norm', linestyle='dashed')
  plt.plot(preds_sys_non_norms, color='lime', label=r'Predicted $\Omega_x\Gamma_y$-Norm', linestyle='dashed')
  plt.plot(preds_non_non_norms, color='cyan', label=r'Predicted $\Gamma_x\Gamma_y$-Norm', linestyle='dashed')
  #plt.title("Deep Dynamics of Quadrants Frobenius Norms")
  plt.ylim([-0.05, np.max([sys_sys_norms,non_sys_norms,sys_non_norms,non_non_norms])+0.55])
  plt.xlim([-10, num_epochs])
  plt.axhline(0, color='black')
  plt.axvline(0, color='black')
  plt.ylabel("Frobenius Norm")
  plt.xlabel("Epoch number")
  plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.52, 0.5, 0.5),ncol=2)
  plt.grid()
  plt.savefig('s_quad_norms_plot.pdf')
  plt.close()

  # Plot each trajectory together for comparison
  colours_arr = [('red','green',r'Prediction $\pi_1$', r'Real $\pi_1$'),('blue','orange',r'Prediction $\pi_3$', r'Real $\pi_3$'),
                 ('cyan','purple',r'Prediction $\pi_2$', r'Real $\pi_2$')]
  predictions = np.append(sys_predictions, non_predictions, axis=1)
  trainings = np.append(sys_trainings, non_trainings, axis=1)
  unique_svs,unique_indices = np.unique(predictions[-1,:,0], return_index=True)
  unique_indices = unique_indices[::-1]
  unique_svs = np.round(unique_svs,2)[::-1]
  colours = {unique_svs[i]:colours_arr[i] for i in range(unique_svs.shape[0])}
  print("Num Plots: ", trainings.shape[2])
  for j in [unique_indices[0], unique_indices[2], unique_indices[1]]:
      identifier = colours[np.round(predictions[-1,j,0],2)]
      labelr_real = identifier[3]
      plt.plot(np.arange(num_epochs), trainings[:,j,0], color=identifier[1], label=labelr_real)
  for j in [unique_indices[0], unique_indices[2], unique_indices[1]]:
      identifier = colours[np.round(predictions[-1,j,0],2)]
      labelr = identifier[2]
      plt.plot(np.arange(num_epochs), predictions[:,j,0], color=identifier[0], label=labelr, linestyle='dashed')
  #plt.title("Deep Dynamics of Singular Values")
  plt.ylabel("Singular Value")
  plt.xlabel("Epoch number")
  plt.grid()
  plt.axhline(0, color='black')
  plt.axvline(0, color='black')
  plt.legend(loc='center', bbox_to_anchor=(0.55, 0.81),ncol=2)
  plt.savefig("sSVs.pdf")
  plt.close()

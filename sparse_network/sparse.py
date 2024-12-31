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
    # : A --> matrix that is a full mapping of the network
    # : U --> left orthogonal matrix
    # : VT --> right transpose orthogonal matrix
    # : num_svds --> number of singular values needed to retain
    # : k2 --> number of non-systematic outputs
    #output the rank of A - this determines the number of singular values that are returned
    # use rank of A and num_svds and check if there are any discrepancies between the two
    S = np.dot(U.T, np.dot(A,VT.T)) #SDV
    small_length = np.min([S.shape[0], S.shape[1]]) #lower-rank sub-structure, from S return fewest columns or rows and that is the lowest rank
    s = np.array([S[i,i] for i in range(small_length)]) # construct a list of singular values, sorted from highest to lowest
    if k2 == 0: 
        s = np.sort(s)[::-1] #only sorting eigenvalues if k2 == 0, otherwise not going to sort it
    s = s[:num_svds] #under what conditions are these not sorted, k2 == 1 and means will not be sorted
    # s is not sorted here
    return U, s, VT

#TODO: test this and reference the literature if everything is code correctly (svs function make sense)
#TODO: add following component and ensure it works with the above 
#TODO: how are we separating dense from sparse network
#TODO:   dense and sparse network are separated by the number of singular values that are returned?
#TODO: test svs with examples from the internet and see how it is meant to work


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
#TODO: what has inspired the creation of the gen_binary_patterns function, is this meant to mimick the examples from SCAN dataset and how has it been adapted 


# TODO: we will need to unpack the dynamic_formular which generates the data that we need
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
def init_random_params(scale, layer_sizes, seed):
    #Returns a list of tuples where each tuple is the weight matrix bias vector for a layer
    np.random.seed(seed)
    #
    return [np.random.normal(0.0,scale, (n,m)) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

@jit
def predict(params, inputs):
    # propagate data forward through the network
    return jnp.dot(params[1], jnp.dot(params[0], inputs))

@jit
def loss(params, batch):
    # loss over a batch of data
    inputs, targets = batch
    preds = predict(params, inputs)
    return np.mean(np.sum(jnp.power((preds - targets), 2), axis = 1))

    # Now we are doing the matrices for non-compositional features (k1, k2)

# TODO: Questions to ask
# TODO: Why are we specifying number of singular values to return if we have a function we use to return lower rank of the matrix
# TODO: where is the methodology of generating binary patterns adopted from
# TODO: Why are we reversing the order of X and Y from the data, X and Y flipped to effectively slice the data

if __name__ =='__main__':
  
      # Data Hyper-parameters
  n1 = 3 #n1 num sys inputs
  n2 = 1 #n2 num sys outputs
  k1 = 3 #k1 num nonsys reps input
  k2 = 1  #k2 num nonsys reps output
  r = 1 #r scale
  a_init = 0
  num_time_steps = 3
  num_svds = 3

#   sv_preds, sxv_preds_inv = dynamic_formular(n1,n2,k1,k2,r,a_init,num_time_steps,num_svds)
#   print(sv_preds,sxv_preds_inv)

  def updater(params, batch):
      # function that updates model parameters
      sigma_xx = (1/batch[0].shape[1])*np.dot(batch[0],batch[0].T)
      sigma_yx = (1/batch[0].shape[1])*np.dot(batch[1], batch[0].T)
      W2W1 = np.dot(params[1], params[0])
      return [params[0] + ((1/tau) * np.dot(params[1].T, sigma_yx - np.dot(W2W1, sigma_xx))), \
              params[1] + ((1/tau) * np.dot(sigma_yx - np.dot(W2W1, sigma_xx), params[0].T))]
  num_hidden = 50
  layer_sizes = [n1 + k1*2**n1, int(num_hidden), n2 + k2*2**n1]
  step_size = 0.02
  num_epochs = 300
  param_scale = 0.01/float(num_hidden)
  # if there is non systematic output (identity matrix), we need to have 2**n1 singular values, otherwise we need systematic output
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
  #variable defined the same as in the paper, used in dynamics formula
  tau = 1/(X.shape[1]*step_size)

  #Initial network parameters and initial network SVs to be small
  #returns weight and bias matrices for each layer
  params = init_random_params(param_scale, layer_sizes, seed)

  #Get ground truth trajectories and same them for plotting at the end, as well as the matrices and SV indices for next generation
  predictions, preds_sys_norms, preds_non_norms, preds_quad_norms, U, VT, sv_indices =\
        dynamic_formular(X,Y,n1, n2, k1, k2, r, a.reshape(1,num_svds), num_epochs, num_svds)
  preds_sys_sys_norms = preds_quad_norms[0]
  preds_non_sys_norms = preds_quad_norms[1]
  preds_sys_non_norms = preds_quad_norms[2] 
  preds_non_non_norms = preds_quad_norms[3]

  # we track the SVs of the input-output covariance matrix from the network at the end of each epoch
  for epoch in range(num_epochs):
      for batch_start in range(0, X.shape[1], batch_size):
          batch = (X[:,batch_start:batch_start+batch_size], Y[:,batch_start:batch_start+batch_size])
          params = updater(params, (X[:,batch_start:batch_start+batch_size], Y[:,batch_start:batch_start+batch_size]))
      lossr = loss(params, (X,Y))
      losses[epoch] = lossr
      print(lossr)
      full_map = np.dot(params[1], params[0]) #multiply weight and bias matrices to get full mapping
      _, a, _ = svs(full_map, U, VT, num_svds, k2)
      traj_real = np.vstack([traj_real, a])

      #Get systematic and non-systematic Frobenius norms 
      sys_norms[epoch] = np.linalg.norm(full_map[:n2], 'fro')
      non_norms[epoch] = np.linalg.norm(full_map[n2:], 'fro') 

      #Get quadrant Frobenius norms 
      sys_sys_norms[epoch] = np.linalg.norm(full_map[:n2,:n1], 'fro')
      sys_non_norms[epoch] = np.linalg.norm(full_map[n2:,:n1], 'fro')
      non_sys_norms[epoch] = np.linalg.norm(full_map[:n2,n1:], 'fro')
      non_non_norms[epoch] = np.linalg.norm(full_map[n2:,n1:], 'fro')

  trainings[:,:,0] = traj_real[1:]
  colours_arr = [('red','green',r'Prediction $\pi_1$', r'Real $\pi_1$'),('blue','orange',r'Prediction $\pi_2$', r'Real $\pi_2$'),
                 ('cyan','purple',r'Prediction $\pi_3$', r'Real $\pi_3$')]
  unique_svs, unique_indices = np.unique(predictions[-1,:], return_index = True)
  unique_indices = unique_indices[::-1]
  unique_svs = np.round(unique_svs, 2)[::-1]
  colours = {unique_svs[i]: colours_arr[i] for i in range(unique_svs.shape[0])} 
  print("Num Plots: ", trainings.shape[2])
  for j in [unique_indices[0], unique_indices[1], unique_indices[2]]:
      identifier = colours[np.round(predictions[-1,j],2)]
      labelr_real = identifier[3]
      plt.plot(np.arange(num_epochs), trainings[:,j], color = identifier[1],label=labelr_real)
  for j in [unique_indices[0], unique_indices[1], unique_indices[2]]:
      identifier = colours[np.round(predictions[-1,j],2)]
      labelr_pred = identifier[2]
      plt.plot(np.arange(num_epochs), predictions[:,j], color = identifier[1], linestyle = 'dashed', label = labelr_pred)

  plt.ylabel('Singular Value')
  plt.xlabel('Epoch number')
  plt.grid()
  plt.axhline(y = 0, color = 'black')
  plt.axvline(x = 0, color = 'black')
  plt.legend(loc = 'center', bbox_to_anchor = (0.55, 0.79), ncol = 2)
  plt.savefig('dSVs.pdf')
  plt.close()

  #Plot Quadrant norms together
  plt.plot(sys_sys_norms, color = 'red', label = r'$\Omega_x\Omega_y$-Norm')
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
  plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.52, 0.5, 0.5), ncol=2)
  plt.grid()
  plt.savefig('d_quad_norms_plot.pdf')
  plt.close() 

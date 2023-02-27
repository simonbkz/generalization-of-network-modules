import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
from jax import jit, grad
import jax.numpy as jnp
import sys

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=200)
matplotlib.rcParams.update({'font.family':'Times New Roman', 'font.size': 15})
plt.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

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

@jit
def split_loss(params, batch):
  # Loss over a batch of data
  inputs, targets = batch
  preds = predict(params, inputs)
  return np.mean(np.sum(jnp.power((preds[:n2] - targets[:n2]),2), axis=1)), np.mean(np.sum(jnp.power((preds[n2:] - targets[n2:]),2), axis=1))

if __name__ == "__main__":

  def updater(params, batch):
    # Function which updates model parameters
    sigma_xx = (1/batch[0].shape[1])*np.dot(batch[0], batch[0].T)
    sigma_yx = (1/batch[0].shape[1])*np.dot(batch[1], batch[0].T)
    W2W1 = np.dot(params[1], params[0])
    return [params[0] + ((1/tau) * np.dot(params[1].T, sigma_yx - np.dot(W2W1, sigma_xx))),\
            params[1] + ((1/tau) * np.dot(sigma_yx - np.dot(W2W1, sigma_xx), params[0].T))]

  # Data Hyper-parameters
  data_params = sys.argv[1]
  data_params = data_params.split(' ')
  n1 = int(data_params[0]) #n1 num sys inputs
  n2 = int(data_params[1]) #n2 num sys outputs
  k1 = int(data_params[2]) #k1 num nonsys reps input
  k2 = int(data_params[3]) #k2 num nonsys reps output
  r =  float(data_params[4]) #r scale
  if n1 > 0:
      n1_effective = n1
  else:
      n1_effective = int(data_params[5])

 # Create Dataset training data
  X = np.flip(gen_binary_patterns(n1_effective).T, axis=1)
  for i in range(k1):
      X = np.vstack([X, r*np.eye(2**n1_effective)])
  if n1 == 0:
      X = X[n1_effective:]

  # Create Dataset labels
  Y = np.flip(gen_binary_patterns(n1_effective).T, axis=1)
  for i in range(k2):
      Y = np.vstack([Y, r*np.eye(2**n1_effective)])
  Y_keep_feat = np.arange(Y.shape[0])
  Y_delete = np.random.choice(n1_effective, n1_effective-n2, replace=False)
  Y_keep_feat = np.delete(Y_keep_feat, Y_delete)
  Y = Y[Y_keep_feat]
 
  print("Input Data: \n", X)
  print("Initial Labels: \n", Y)
  print(sys.argv)

  X_train = X[:,:3]
  Y_train = Y[:,:3]
  X_test = X[:,3:]
  Y_test = Y[:,3:]

  # Training hyper-parameters
  num_hidden = 50.0
  layer_sizes = [n1 + k1*2**n1_effective,int(num_hidden), n2 + k2*2**n1_effective]
  step_size = 0.02
  num_epochs = 500 #300
  num_trainings = int(sys.argv[2])
  param_scale = 0.001
  seed = np.random.randint(0,100000) # can set seed here, for now it is random. The only randomness is in the network init

  # Variable defined the same as in the paper, used in dynamics formula
  tau = 1/(X.shape[1]*step_size)

  # Holds the SV trajectories to be plotted
  sys_train_losses = np.zeros((num_epochs, num_trainings))
  non_train_losses = np.zeros((num_epochs, num_trainings))
  sys_test_losses = np.zeros((num_epochs, num_trainings))
  non_test_losses = np.zeros((num_epochs, num_trainings))
  
  for training_num in range(num_trainings):

    # Initialize network parameters and initial network SVs set to be small
    params = init_random_params(param_scale, layer_sizes, seed)
    
    # We track the SVs of the input-output covariance matrix from the network at the end of each epoch
    print('Training Num: ', training_num)
    for epoch in range(num_epochs):
      params = updater(params, (X_train,Y_train))
      sys_lossr, non_lossr = split_loss(params, (X_train,Y_train))
      sys_train_losses[epoch, training_num] = sys_lossr
      non_train_losses[epoch, training_num] = non_lossr
      #print(sys_lossr, ' ', non_lossr)
      sys_lossr, non_lossr = split_loss(params, (X_test,Y_test))
      sys_test_losses[epoch, training_num] = sys_lossr
      non_test_losses[epoch, training_num] = non_lossr
      #print(sys_lossr, ' ', non_lossr)      
  
  # Save our logs and plot and initial graph
  np.savetxt('train_accs_sys.txt', sys_train_losses)
  np.savetxt('train_accs_non.txt', non_train_losses)
  np.savetxt('test_accs_sys.txt', sys_test_losses)
  np.savetxt('test_accs_non.txt', non_test_losses)

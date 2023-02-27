import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
from jax import jit, grad
import jax.numpy as jnp
import sys

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=200)
matplotlib.rcParams.update({'font.size': 13})
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

if __name__ == "__main__":

  # Data Hyper-parameters
  n1 = int(sys.argv[1])

  # Create Dataset training data
  #X = np.flip(gen_binary_patterns(n1).T, axis=1)
  X = gen_binary_patterns(n1)
  #print(X @ X.T)
  
  num_samples = 5000
  print("Checking the probability of sampling a full-rank covaiance matrix for compositional structure with " + str(n1) + " binary features:")
  print("Num Samples | Probability Full-rank")
  for j in range(n1,2**n1):
    count = 0
    examples = []
    for i in range(num_samples):
        sample_idx = np.random.choice(np.arange(X.shape[0]), j, replace=False)
        sample = X[sample_idx]
        mat = jnp.dot(sample.T, sample)
        examples.append(jnp.abs(mat))
        if jnp.linalg.det(jnp.abs(mat)) > 0.1:
            count = count + 1
    if j < 10:
        print("    ",j, "     |      ", count/num_samples)
    else:
        print("    ",j, "    |      ", count/num_samples)
    unique, counts = np.unique(examples,axis=0,return_counts=True)

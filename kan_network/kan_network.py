import time
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from jax import jit
import jax.numpy as jnp
import torch

def piece_wise_f(x):
  if x < 0:
    return -2*x
  if x < 1:
    return -0.5*x
  return 2*x - 2.5

if __name__ == '__main__':
  
  k = 3 # Grid size
  inp_size = 5
  out_size = 7
  batch_size = 10
  X = torch.randn(batch_size, inp_size) # Our input

  linear = nn.Linear(inp_size*k, out_size)  # Weights

  repeated = X.unsqueeze(1).repeat(1,k,1)
  shifts = torch.linspace(-1, 1, k).reshape(1,k,1)

  shifted = repeated + shifts
  intermediate = torch.cat([shifted[:,:1,:], torch.relu(shifted[:,1:,:])], dim=1).flatten(1)

  outputs = linear(intermediate)

#   X = torch.linspace(-2, 2, 100)
#   plt.plot(X, [piece_wise_f(x) for x in X])
#   plt.grid()
#   plt.show()
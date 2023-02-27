import time
import itertools

import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Conv, GeneralConv, Relu, Sigmoid, Flatten, LogSoftmax
from jax.nn.initializers import variance_scaling, normal, glorot_normal, he_normal
from jax.tree_util import tree_map
from jax.tree_util import tree_multimap
import datasets

matplotlib.rcParams.update({'font.family':'Times New Roman', 'font.size': 15})

# Standard quadratic loss
@jit
def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return jnp.mean(jnp.sum(jnp.power((preds - targets),2), axis=1))

# Quadratic loss separated by systematic and non-systematic components (normalized in training loop)
@jit
def loss_display(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  elem_loss = jnp.power((preds - targets),2)
  sys_loss = jnp.mean(jnp.sum(elem_loss[:,:30], axis=1))
  non_loss = jnp.mean(jnp.sum(elem_loss[:,30:], axis=1))
  return sys_loss, non_loss, jnp.mean(jnp.sum(elem_loss, axis=1))

# Dense network architecture
def Dense_Net():
    return stax.serial(
    Conv(32, (9, 27), W_init = normal(init_var), b_init = normal(init_var), padding='VALID'), Relu,
    Conv(32, (7, 21), W_init = normal(init_var), b_init = normal(init_var), padding='VALID'), Relu,
    Conv(32, (5, 15), W_init = normal(init_var), b_init = normal(init_var), padding='VALID'), Relu,
    Flatten,
    Dense(1030, W_init = normal(init_var), b_init = normal(init_var)))

if __name__ == "__main__":
  rng = random.PRNGKey(np.random.randint(1000))

  # Hyper-parameters
  step_size = 2e-3
  num_epochs = 300
  num_trainings = 3
  batch_size = 16
  metric_batch_size = 1028
  init_var = 0.01
  momentum_mass = 0.0

  # Init our network
  init_random_params, predict = Dense_Net()
  opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)

  # Get Compositional-MNIST dataset
  train_images, train_labels, test_images, test_labels, num_batches, batches = datasets.composed_mnist(batch_size, 10.0)
 
  @jit
  def update_standard(i, opt_state, batch):
    # Use SGD Learning rule
    params = get_params(opt_state)
    grads = grad(loss)(params, batch)
    return opt_update(i, grads, opt_state)
 
  # Arrays for holding training progress statistics
  train_accs = np.zeros((num_trainings, num_epochs + 1))
  train_accs_sys = np.zeros((num_trainings, num_epochs + 1))
  train_accs_non = np.zeros((num_trainings, num_epochs + 1))
  test_accs = np.zeros((num_trainings, num_epochs + 1))
  test_accs_sys = np.zeros((num_trainings, num_epochs + 1))
  test_accs_non = np.zeros((num_trainings, num_epochs + 1))

  print("\nStarting training...")
  for training_idx in range(num_trainings):
      # Initialize the network
      _, student_init_params = init_random_params(rng, (-1, 28, 84, 1))
      student_opt_state = opt_init(student_init_params)

      itercount = itertools.count()
      itercount_IL = itertools.count()
      
      # Get loss of untrained model
      student_params = get_params(student_opt_state)
      train_sample = np.random.choice(np.arange(train_images.shape[0]), metric_batch_size)
      test_sample = np.random.choice(np.arange(test_images.shape[0]), metric_batch_size)
      sys_acc, non_acc, train_acc = loss_display(student_params, (train_images[train_sample], train_labels[train_sample]))
      train_accs[training_idx, 0] = train_acc
      train_accs_sys[training_idx, 0] = sys_acc/3
      train_accs_non[training_idx, 0] = non_acc/100
      sys_acc_test, non_acc_test, test_acc = loss_display(student_params, (test_images[test_sample], test_labels[test_sample]))
      test_accs[training_idx, 0] = test_acc
      test_accs_sys[training_idx, 0] = sys_acc_test/3
      test_accs_non[training_idx, 0] = non_acc_test/100
 
      for epoch in range(num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
            student_opt_state = update_standard(next(itercount), student_opt_state, next(batches))
        epoch_time = time.time() - start_time

        # Track training and test performance at the end of every epoch
        student_params = get_params(student_opt_state)
        train_sample = np.random.choice(np.arange(train_images.shape[0]), metric_batch_size)
        test_sample = np.random.choice(np.arange(test_images.shape[0]), metric_batch_size)
        
        sys_acc, non_acc, train_acc = loss_display(student_params, (train_images[train_sample], train_labels[train_sample]))
        train_accs[training_idx, epoch+1] = train_acc
        train_accs_sys[training_idx, epoch+1] = sys_acc/3
        train_accs_non[training_idx, epoch+1] = non_acc/100
        sys_acc_test, non_acc_test, test_acc = loss_display(student_params, (test_images[test_sample], test_labels[test_sample]))
        test_accs[training_idx, epoch+1] = test_acc
        test_accs_sys[training_idx, epoch+1] = sys_acc_test/3
        test_accs_non[training_idx, epoch+1] = non_acc_test/100
        
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set loss {}".format(train_acc))
        print(" Sys loss normalized {}".format(train_accs_sys[training_idx, epoch]))
        print(" Non loss normalized {}".format(train_accs_non[training_idx, epoch]))
        print(" Test set loss {}".format(test_acc))
        print(" Sys loss normalized {}".format(test_accs_sys[training_idx, epoch]))
        print(" Non loss normalized {}".format(test_accs_non[training_idx, epoch]))

  # Save our logs and plot and initial graph
  np.savetxt('train_accs.txt', train_accs)
  np.savetxt('train_accs_sys.txt', train_accs_sys)
  np.savetxt('train_accs_non.txt', train_accs_non)
  np.savetxt('test_accs.txt', test_accs)
  np.savetxt('test_accs_sys.txt', test_accs_sys)
  np.savetxt('test_accs_non.txt', test_accs_non)


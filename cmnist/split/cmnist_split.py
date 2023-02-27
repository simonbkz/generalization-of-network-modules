import time
import itertools

import numpy as np
import numpy.random as npr
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

# Standard quadratic loss
@jit
def loss_left(params, batch):
  inputs, targets = batch
  preds = predict_left(params, inputs)
  return jnp.mean(jnp.sum(jnp.power((preds - targets),2), axis=1))

# Standard quadratic loss
@jit
def loss_right(params, batch):
  inputs, targets = batch
  preds = predict_right(params, inputs)
  return jnp.mean(jnp.sum(jnp.power((preds - targets),2), axis=1))

# Quadratic loss separated by systematic and non-systematic components (normalized in training loop)
@jit
def loss_display_left(params, batch):
  inputs, targets = batch
  preds = predict_left(params, inputs)
  elem_loss = jnp.power((preds - targets),2)
  return jnp.mean(jnp.sum(elem_loss, axis=1))

# Quadratic loss separated by systematic and non-systematic components (normalized in training loop)
@jit
def loss_display_right(params, batch):
  inputs, targets = batch
  preds = predict_right(params, inputs)
  elem_loss = jnp.power((preds - targets),2)
  return jnp.mean(jnp.sum(elem_loss, axis=1))

# Left module for split network architecture
def Left_Module():
    return stax.serial(
    Conv(16, (9, 27), W_init = normal(init_var), b_init = normal(init_var), padding='VALID'), Relu,
    Conv(16, (7, 21), W_init = normal(init_var), b_init = normal(init_var), padding='VALID'), Relu,
    Conv(16, (5, 15), W_init = normal(init_var), b_init = normal(init_var), padding='VALID'), Relu,
    Flatten,
    Dense(30, W_init = normal(init_var), b_init = normal(init_var)))

# Right module for split network architecture
def Right_Module():
    return stax.serial(
    Conv(16, (9, 27), W_init = normal(init_var), b_init = normal(init_var), padding='VALID'), Relu,
    Conv(16, (7, 21), W_init = normal(init_var), b_init = normal(init_var), padding='VALID'), Relu,
    Conv(16, (5, 15), W_init = normal(init_var), b_init = normal(init_var), padding='VALID'), Relu,
    Flatten,
    Dense(1000, W_init = normal(init_var), b_init = normal(init_var)))

# Split network architecture
def Siamese_Net():
    left_init, left_predict = Left_Module()
    right_init, right_predict = Right_Module()
    return left_init, left_predict, right_init, right_predict

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
  init_random_params_left, predict_left, init_random_params_right, predict_right = Siamese_Net()
  opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)

  # Get Compositional-MNIST dataset
  train_images, train_labels, test_images, test_labels, num_batches, batches = datasets.composed_mnist(batch_size, 10.0)
 
  @jit
  def update_standard_left(i, opt_state, batch):
    # Use SGD Learning rule
    params = get_params(opt_state)
    grads = grad(loss_left)(params, batch)
    return opt_update(i, grads, opt_state)

  @jit
  def update_standard_right(i, opt_state, batch):
    # Use SGD Learning rule
    params = get_params(opt_state)
    grads = grad(loss_right)(params, batch)
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
      _, student_init_params_left = init_random_params_left(rng, (-1, 28, 84, 1))
      _, student_init_params_right = init_random_params_right(rng, (-1, 28, 84, 1))
      student_opt_state_left = opt_init(student_init_params_left)
      student_opt_state_right = opt_init(student_init_params_right)

      itercount = itertools.count()
      itercount_IL = itertools.count()

      # Get loss of untrained model
      student_params_left = get_params(student_opt_state_left)
      student_params_right = get_params(student_opt_state_right)
      train_sample = np.random.choice(np.arange(train_images.shape[0]), metric_batch_size)
      test_sample = np.random.choice(np.arange(test_images.shape[0]), metric_batch_size)
      sys_acc = loss_display_left(student_params_left, (train_images[train_sample], train_labels[train_sample,:30]))
      non_acc = loss_display_right(student_params_right, (train_images[train_sample], train_labels[train_sample,30:]))
      train_accs_sys[training_idx, 0] = sys_acc/3
      train_accs_non[training_idx, 0] = non_acc/100
      train_accs[training_idx, 0] = (train_accs_sys[training_idx, 0] + train_accs_non[training_idx, 0])/2
      sys_acc_test = loss_display_left(student_params_left, (test_images[test_sample], test_labels[test_sample,:30]))
      non_acc_test = loss_display_right(student_params_right, (test_images[test_sample], test_labels[test_sample,30:]))
      test_accs_sys[training_idx, 0] = sys_acc_test/3
      test_accs_non[training_idx,0] = non_acc_test/100
      test_accs[training_idx, 0] = (test_accs_sys[training_idx, 0] + test_accs_non[training_idx, 0])/2 

      for epoch in range(num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
            next_inputs, next_labels = next(batches)
            student_opt_state_left = update_standard_left(next(itercount), student_opt_state_left, (next_inputs, next_labels[:,:30]))
            student_opt_state_right = update_standard_right(next(itercount),student_opt_state_right, (next_inputs, next_labels[:,30:]))
        epoch_time = time.time() - start_time

        # Track training and test accuracy performance at the end of every epoch
        student_params_left = get_params(student_opt_state_left)
        student_params_right = get_params(student_opt_state_right)
        train_sample = np.random.choice(np.arange(train_images.shape[0]), metric_batch_size)
        test_sample = np.random.choice(np.arange(test_images.shape[0]), metric_batch_size)
        
        sys_acc = loss_display_left(student_params_left, (train_images[train_sample], train_labels[train_sample,:30]))
        non_acc = loss_display_right(student_params_right, (train_images[train_sample], train_labels[train_sample,30:]))
        train_accs_sys[training_idx, epoch+1] = sys_acc/3
        train_accs_non[training_idx, epoch+1] = non_acc/100
        train_accs[training_idx, epoch+1] = (train_accs_sys[training_idx, epoch] + train_accs_non[training_idx, epoch])/2
        sys_acc_test = loss_display_left(student_params_left, (test_images[test_sample], test_labels[test_sample,:30]))
        non_acc_test = loss_display_right(student_params_right, (test_images[test_sample], test_labels[test_sample,30:]))
        test_accs_sys[training_idx, epoch+1] = sys_acc_test/3
        test_accs_non[training_idx, epoch+1] = non_acc_test/100
        test_accs[training_idx, epoch+1] = (test_accs_sys[training_idx, epoch] + test_accs_non[training_idx, epoch])/2
        
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set loss {}".format(train_accs[training_idx, epoch]))
        print(" Sys loss Percent {}".format(train_accs_sys[training_idx, epoch]))
        print(" Non loss Percent {}".format(train_accs_non[training_idx, epoch]))
        print(" Test set accuracy {}".format(test_accs[training_idx, epoch]))
        print(" Sys loss Percent {}".format(test_accs_sys[training_idx, epoch]))
        print(" Non loss Percent {}".format(test_accs_non[training_idx, epoch]))

  # Save our logs and plot and initial graph
  np.savetxt('train_accs.txt', train_accs)
  np.savetxt('train_accs_sys.txt', train_accs_sys)
  np.savetxt('train_accs_non.txt', train_accs_non)
  np.savetxt('test_accs.txt', test_accs)
  np.savetxt('test_accs_sys.txt', test_accs_sys)
  np.savetxt('test_accs_non.txt', test_accs_non)

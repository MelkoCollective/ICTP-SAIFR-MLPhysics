########## ICTP-SAIFR Minicourse on Machine Learning for Many-Body Physics ##########
### Roger Melko, Juan Carrasquilla, Lauren Hayward Sierens and Giacomo Torlai
### Tutorial 4: Training a Restricted Boltzmann Machine (RBM)
#####################################################################################

from __future__ import print_function
from pprint import pformat
import tensorflow as tf
import itertools as it
from random import randint
from rbm import RBM
import sys
import numpy as np
import os
import json
import math as m

#Input parameters:
L  = 4     #linear size of the system
T  = 2.269 #temperature (MC configuration training samples are supplied for
           #T=1.0,1.254,1.508,1.762,2.016, 2.269,2.524,2.778,3.032,3.286 and 3.540)
num_visible     = L*L      #number of visible nodes
num_hidden      = 4        #number of hidden nodes
nsteps          = 1000000  #number of training steps
learning_rate_b = 1e-3     #learning rate for optimization
bsize           = 100      #batch size
num_gibbs       = 10       #number of Gibbs iterations (steps of contrastive divergence)
num_samples     = 10       #number of chains in PCD

### Function to save weights and biases to a parameter file ###
def save_parameters(sess, results_dir, rbm, epochs,L,T):
    weights, visible_bias, hidden_bias = sess.run([rbm.weights, rbm.visible_bias, rbm.hidden_bias])
    parameter_file_path =  'data_ising2d/parameters/parameters_L' + str(L)
    parameter_file_path += '_T' + str(T)
    np.savez_compressed(parameter_file_path, weights=weights, visible_bias=visible_bias, hidden_bias=hidden_bias,
                        epochs=epochs) 
class Args(object):
    pass

class Placeholders(object):
    pass

class Ops(object):
    pass

weights=None                    # weights
visible_bias=None               # visible bias
hidden_bias=None                # hidden bias
bcount=0                        # counter
epochs_done=1                   # epochs counter

# Load the MC configuration data:
train_dir = 'data_ising2d/datasets/'  # Location of training data.
trainName = 'data_ising2d/datasets/ising2d_L'+str(L)+'_T'+str(T)+'_train.txt'
testName = 'data_ising2d/datasets/ising2d_L'+str(L)+'_T'+str(T)+'_test.txt'
xtrain = np.loadtxt(trainName)
xtest = np.loadtxt(testName)

ept=np.random.permutation(xtrain) # random permutation of training data
epv=np.random.permutation(xtest) # random permutation of test data
iterations_per_epoch = xtrain.shape[0] / bsize  

# Initialize the RBM class
rbm = RBM(num_hidden=num_hidden, num_visible=num_visible, weights=weights, visible_bias=visible_bias,hidden_bias=hidden_bias, num_samples=num_samples) 

# Initialize operations and placeholders classes
ops = Ops()
placeholders = Placeholders()
placeholders.visible_samples = tf.placeholder(tf.float32, shape=(None, num_visible), name='v') # placeholder for training data

total_iterations = 0 # starts at zero 
ops.global_step = tf.Variable(total_iterations, name='global_step_count', trainable=False)
learning_rate = tf.train.exponential_decay(
    learning_rate_b,
    ops.global_step,
    100 * xtrain.shape[0]/bsize,
    1.0 # decay rate =1 means no decay
)
  
cost = rbm.neg_log_likelihood_grad(placeholders.visible_samples, num_gibbs=num_gibbs)
optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-2)
ops.lr=learning_rate
ops.train = optimizer.minimize(cost, global_step=ops.global_step)
ops.init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
logZ = rbm.exact_log_partition_function()
placeholders.logZ = tf.placeholder(tf.float32) 
NLL = rbm.neg_log_likelihood(placeholders.visible_samples,placeholders.logZ)
path_to_distr = 'data_ising2d/boltzmann_distributions/distribution_ising2d_L4_T'+str(T)+'.txt'
boltz_distr=np.loadtxt(path_to_distr)
p_x = tf.exp(rbm.free_energy(placeholders.visible_samples))
all_v_states= np.array(list(it.product([0, 1], repeat=num_visible)), dtype=np.float32)

# Observer file
observer_file=open('data_ising2d/observables/training_observer.txt','w',0)
observer_file.write('#      O')
observer_file.write('         NLL')
observer_file.write('\n')

with tf.Session() as sess:
  sess.run(ops.init)
  
  for ii in range(nsteps):
    if bcount*bsize+ bsize>=xtrain.shape[0]:
     bcount=0
     ept=np.random.permutation(xtrain)

    batch=ept[ bcount*bsize: bcount*bsize+ bsize,:]
    bcount=bcount+1
    feed_dict = {placeholders.visible_samples: batch}
    
    _, num_steps = sess.run([ops.train, ops.global_step], feed_dict=feed_dict)

    if num_steps % iterations_per_epoch == 0:
      print ('Epoch = %d     ' % epochs_done,end='')
      lz = sess.run(logZ)
      nll = sess.run(NLL,feed_dict={placeholders.visible_samples: epv, placeholders.logZ: lz})
      px = sess.run(p_x,feed_dict={placeholders.visible_samples: all_v_states})
      
      Ov = 0.0
      for i in range(1<<num_visible):
        Ov += boltz_distr[i]*m.log(boltz_distr[i])
        Ov += -boltz_distr[i]*(m.log(px[i])-lz)
    
      # Print to screen
      print ('Ov = %.6f     ' % Ov,end='')
      print ('NLL = %.6f     ' % nll,end='')
      
      # Save to file
      observer_file.write('%.6f   ' % Ov)
      observer_file.write('%.6f   ' % nll)
      observer_file.write('\n')

      print()
      #save_parameters(sess, rbm)
      epochs_done += 1

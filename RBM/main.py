from __future__ import print_function
from pprint import pformat

import tensorflow as tf

import time
from rbm import RBM
import sys
import numpy as np
import os
import json
from datetime import datetime

def save_parameters(sess, results_dir, rbm, epochs):
    weights, visible_bias, hidden_bias = sess.run([rbm.weights, rbm.visible_bias, rbm.hidden_bias])
    parameter_file_path = os.path.join(results_dir, 'parameters.npz_'+str(epochs))
    np.savez_compressed(parameter_file_path, weights=weights, visible_bias=visible_bias, hidden_bias=hidden_bias,
                        epochs=epochs) 

class Args(object):
    pass

class Placeholders(object):
    pass

class Ops(object):
    pass

def mnist_gibbs():


    # loading the data
    xtrain=np.loadtxt('./data/train.txt')
    xtest=np.loadtxt('./data/test.txt')

    nsteps=1000000 # number of training steps
    bsize=200 # batchsize
    bcount=0 # counter
    ept=np.random.permutation(xtrain) # random permutation of training data
    epv=np.random.permutation(xtest) # random permutation of test data
    iterations_per_epoch = xtrain.shape[0] / bsize  

    results_dir = os.path.join('results', 'run-{}'.format(datetime.now().isoformat()[:-7].replace(':', '-')))
    os.makedirs(results_dir) 
    weights=None
    visible_bias=None
    hidden_bias=None
    epochs_done=1   

    num_hidden = 8  # Number of hidden units.
    num_visible = 4  # Number of visible units. 
    learning_rate_b = 1e-3  # Learning rate used in training.
    num_gibbs = 10  # Number of gibbs iterations to perform.
    train_dir = 'data/'  # Location of training data.
    num_samples = 10 # number of chains
    
    rbm = RBM(num_hidden=num_hidden, num_visible=num_visible, weights=weights, visible_bias=visible_bias,
              hidden_bias=hidden_bias, num_samples=num_samples) # class defining the RBM
    
     
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

    # define operations
    ops.lr=learning_rate
    ops.train = optimizer.minimize(cost, global_step=ops.global_step)
    ops.init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())


    with tf.Session() as sess:
        sess.run(ops.init)

        #for batch in training_batches:
        print("nsteps",nsteps)  
        for ii in range(nsteps):
            
            #xtrain[:,(bcount-1)*bs+1:(bcount-1)*bs+bs]
            if bcount*bsize+ bsize>=xtrain.shape[0]:
               bcount=0
               ept=np.random.permutation(xtrain)


            batch=ept[ bcount*bsize: bcount*bsize+ bsize,:]
            bcount=bcount+1
            feed_dict = {placeholders.visible_samples: batch}
            
            _, num_steps = sess.run([ops.train, ops.global_step], feed_dict=feed_dict)

            #print(ops.lr.eval(),ops.global_step.eval())
            

            if num_steps % iterations_per_epoch == 0:
                print(ops.lr.eval(),ops.global_step.eval())
                print("saving parameters epoch ", epochs_done)
                save_parameters(sess, results_dir, rbm, epochs_done)
                epochs_done += 1


mnist_gibbs()



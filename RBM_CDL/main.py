from __future__ import print_function
from pprint import pformat
import tensorflow as tf
import itertools as it
from random import randint
import argparse
from rbm import RBM
import numpy as np
import json

def main()
    
    # Initialize the command line parser
    parser = argparse.ArgumentParser()
    
    # Read command line arguments
    parser.add_argument('command',type=str,help='command to execute') 
    parser.add_argument('-nV',type=int,default=4,help='number of visible nodes')                
    parser.add_argument('-nH',type=int,default=4,help='number of hidden nodes')   
    parser.add_argument('-steps',type=int,default=1000000,help='training steps')  
    parser.add_argument('-lr',type=float,default=1e-3,help='learning rate')   
    parser.add_argument('-bs',type=int,default=100,help='batch size')   
    parser.add_argument('-CD',type=int,default=10,help='steps of contrastive divergence') 
    parser.add_argument('-nC',type=float,default=10,help='number of chains in PCD')  
    
    # Parse the arguments
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    
    if args.command == 'sample':
        sample(args)


class Args(object):
    pass

class Placeholders(object):
    pass

class Ops(object):
    pass

def train(args):
   
    # Simulation parameters
    num_visible = args.nV           # number of visible nodes
    num_hidden = args.nH            # number of hidden nodes
    nsteps = args.steps             # training steps
    bsize = args.bs                 # batch size
    learning_rate_b=args.lr         # learning rate
    num_gibbs = args.CD             # number of Gibbs iterations
    num_samples = args.nC           # number of chains in PCD
    weights=None                    # weights
    visible_bias=None               # visible bias
    hidden_bias=None                # hidden bias
    bcount=0                        # counter
    epochs_done=1                   # epochs counter
 
    # *************************************************************
    # INSERT HERE THE PATH TO THE TRAINING AND TESTING DATASETS
    trainName = '*******************'
    testName  = '*******************'
    
    # Loading the data
    xtrain = np.loadtxt(trainName)
    xtest = np.loadtxt(testName)
    
    ept=np.random.permutation(xtrain)               # random permutation of training data
    epv=np.random.permutation(xtest)                # random permutation of test data
    iterations_per_epoch = xtrain.shape[0] / bsize  # gradient iteration per epoch

    # Initialize RBM class
    rbm = RBM(num_hidden=num_hidden, num_visible=num_visible, weights=weights, visible_bias=visible_bias,hidden_bias=hidden_bias, num_samples=num_samples) 
    
    # Initialize operations and placeholders classes
    ops = Ops()
    placeholders = Placeholders()

    placeholders.visible_samples = tf.placeholder(tf.float32, shape=(None, num_visible), name='v') # placeholder for training data

    total_iterations = 0 # starts at zero 
    ops.global_step = tf.Variable(total_iterations, name='global_step_count', trainable=False)
    
    # Decaying learning rate
    learning_rate = tf.train.exponential_decay(
        learning_rate_b,
        ops.global_step,
        100 * xtrain.shape[0]/bsize,
        1.0 # decay rate =1 means no decay
    )

    cost = rbm.neg_log_likelihood_grad(placeholders.visible_samples, num_gibbs=num_gibbs)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-2)

    # Define operations
    ops.lr=learning_rate
    ops.train = optimizer.minimize(cost, global_step=ops.global_step)
    ops.init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    
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
                print ('Epoch = %d     ' % epochs_done)
                #save_parameters(sess, rbm)
                epochs_done += 1

def sample(args):
       
    num_visible = args.nV   # number of visible nodes
    num_hidden = args.nH    # number of hidden nodes
    
    # *************************************************************
    # INSERT HERE THE PATH TO THE PARAMETERS FILE
    path_to_params = '*******************'
   
    # Load the RBM parameters 
    params = np.load(path_to_params)
    weights = params['weights']
    visible_bias = params['visible_bias']
    hidden_bias = params['hidden_bias']
    hidden_bias=np.reshape(hidden_bias,(hidden_bias.shape[0],1))
    visible_bias=np.reshape(visible_bias,(visible_bias.shape[0],1))
  
    # Sampling parameters
    num_samples=1000   # how many independent chains will be sampled
    gibb_updates=100   # how many gibbs updates per call to the gibbs sampler
    nbins=1000         # number of calls to the RBM sampler      

    # Initialize RBM class
    rbm = RBM(num_hidden=num_hidden, num_visible=num_visible, weights=weights, visible_bias=visible_bias,hidden_bias=hidden_bias, num_samples=num_samples)
    hsamples,vsamples=rbm.stochastic_maximum_likelihood(gibb_updates)

    # Initialize tensorflow
    init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
   
    with tf.Session() as sess:
        sess.run(init)
    
        for i in range(nbins):
            print ('bin %d\t' %i)
            
            # Gibbs sampling
            _,samples=sess.run([hsamples,vsamples])


def save_parameters(sess,rbm,epochs):
    weights, visible_bias, hidden_bias = sess.run([rbm.weights, rbm.visible_bias, rbm.hidden_bias])
    
    # *************************************************************
    # INSERT HERE THE PATH TO THE PARAMETERS FILE
    parameter_file_path = '*******************'
    
    np.savez_compressed(parameter_file_path, weights=weights, visible_bias=visible_bias, hidden_bias=hidden_bias,
                        epochs=epochs)


if __name__=='__main__':
    main()

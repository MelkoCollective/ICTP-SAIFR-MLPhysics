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

def save_parameters(sess, results_dir, rbm, epochs,L,T):
    weights, visible_bias, hidden_bias = sess.run([rbm.weights, rbm.visible_bias, rbm.hidden_bias])
    parameter_file_path =  '../data/ising2d/parameters/parameters_L' + str(L)
    parameter_file_path += '_T' + str(T)
    np.savez_compressed(parameter_file_path, weights=weights, visible_bias=visible_bias, hidden_bias=hidden_bias,
                        epochs=epochs) 
class Args(object):
    pass

class Placeholders(object):
    pass

class Ops(object):
    pass

def train(args):
   
    # Simulation parameters
    T = args.t                      # temperature
    num_visible = args.L*args.L     # number of visible nodes
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
 
    # Loading the data
    train_dir = '../data/ising2d/datasets/'  # Location of training data.
    trainName = '../data/ising2d/datasets/ising2d_L'+str(args.L)+'_T'+str(T)+'_train.txt'
    testName = '../data/ising2d/datasets/ising2d_L'+str(args.L)+'_T'+str(T)+'_test.txt'
    xtrain = np.loadtxt(trainName)
    xtest = np.loadtxt(testName)
    
    ept=np.random.permutation(xtrain) # random permutation of training data
    epv=np.random.permutation(xtest) # random permutation of test data
    iterations_per_epoch = xtrain.shape[0] / bsize  

    # Initialize RBM class
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
    path_to_distr = '../data/ising2d/boltzmann_distributions/distribution_ising2d_L4_T'+str(T)+'.txt'
    boltz_distr=np.loadtxt(path_to_distr)
    p_x = tf.exp(rbm.free_energy(placeholders.visible_samples))
    all_v_states= np.array(list(it.product([0, 1], repeat=num_visible)), dtype=np.float32)
    
    # Observer file
    observer_file=open('../data/ising2d/observables/training_observer.txt','w',0)
    observer_file.write('#      O')
    observer_file.write('         NLL')
    observer_file.write('         <E>')
    observer_file.write('      <|M|>')
    observer_file.write('\n')
    gibb_updates=10
    observer_samples=rbm.observer_sampling(gibb_updates)
    nbins=100
    
    # Build lattice
    path_to_lattice = '../data/ising2d/lattice2d_L'+str(args.L)+'.txt'
    nn=np.loadtxt(path_to_lattice)
    e = np.zeros((num_samples))
    N = num_visible
    E=0.0
    M=0.0
    
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
                E = 0.0
                M = 0.0
                
                for i in range(1<<num_visible):
                    Ov += boltz_distr[i]*m.log(boltz_distr[i])
                    Ov += -boltz_distr[i]*(m.log(px[i])-lz)
                
                for i in range(nbins):
                    
                    # Gibbs sampling
                    samples=sess.run(observer_samples)
                    spins = np.asarray((2*samples-1))
                    
                    # Compute averages of magnetizations
                    m_avg = np.mean(np.absolute(np.sum(spins,axis=1)))
                    
                    # Compute averages of energies
                    e.fill(0.0)
                    for k in range(num_samples):
                        for i in range(N):
                            e[k] += -spins[k,i]*(spins[k,int(nn[i,0])]+spins[k,int(nn[i,1])])
                    e_avg = np.mean(e)
                    E += (e_avg-E)/float(i+1)
                    M += (m_avg-M)/float(i+1)
              
                # Print observer on screen
                print ('Ov = %.6f     ' % Ov,end='')
                print ('NLL = %.6f     ' % nll,end='')
                print ('<E> = %.6f     ' % (E/float(N)),end='')
                print ('<|M|> = %.6f     ' % (M/float(N)),end='')
                
                # Save observer on file
                observer_file.write('%.6f   ' % Ov)
                observer_file.write('%.6f   ' % nll)
                observer_file.write('%.6f   ' % (E/float(N)))
                observer_file.write('%.6f   ' % (M/float(N)))
                observer_file.write('\n')
 
                print()
                #save_parameters(sess, rbm)
                epochs_done += 1


def sample(args):
       
    T = args.t                      # temperature
    num_visible = args.L*args.L     # number of visible units 
    num_hidden = args.nH            # number of hidden units
    
    # Build lattice
    path_to_lattice = '../data/ising2d/lattice2d_L'+str(args.L)+'.txt'
    nn=np.loadtxt(path_to_lattice)
    
    # Load the RBM parameters
    path_to_params = '../data/ising2d/parameters/parameters_nH'+str(num_hidden) + '_L'+str(args.L)+'_T'+str(T)+'.npz'
    params = np.load(path_to_params)
    weights = params['weights']
    visible_bias = params['visible_bias']
    hidden_bias = params['hidden_bias']
    hidden_bias=np.reshape(hidden_bias,(hidden_bias.shape[0],1))
    visible_bias=np.reshape(visible_bias,(visible_bias.shape[0],1))
  
    # Sampling parameters
    num_samples=1000   # how many independent chains will be sampled
    gibb_updates=10    # how many gibbs updates per call to the gibbs sampler
    nbins=1000         # number of calls to the RBM sampler      

    # Initialize RBM class
    rbm = RBM(num_hidden=num_hidden, num_visible=num_visible, weights=weights, visible_bias=visible_bias,hidden_bias=hidden_bias, num_samples=num_samples)
    hsamples,vsamples=rbm.stochastic_maximum_likelihood(gibb_updates)

    # Initialize tensorflow
    init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
   
    N = num_visible
    e = np.zeros((num_samples))
    e2= np.zeros((num_samples))
 
    with tf.Session() as sess:
        sess.run(init)
    
        for i in range(nbins):
            print ('bin %d\t\t' %i,end='')
            
            # Gibbs sampling
            _,samples=sess.run([hsamples,vsamples])
            spins = np.asarray((2*samples-1))
            
            # Compute averages of magnetizations
            m_avg = np.mean(np.absolute(np.sum(spins,axis=1)))
            m2_avg = np.mean(np.multiply(np.sum(spins,axis=1),np.sum(spins,axis=1)))
            
            # Compute averages of energies
            e.fill(0.0)
            e2.fill(0.0)
            for k in range(num_samples):
                for i in range(N):
                    e[k] += -spins[k,i]*(spins[k,int(nn[i,0])]+spins[k,int(nn[i,1])])
                e2[k] = e[k]*e[k]
            e_avg = np.mean(e)
            e2_avg= np.mean(e2)
            
            # Compute specific heat and susceptibility
            c = (e2_avg-e_avg*e_avg)
            s = (m2_avg-m_avg*m_avg)

            # Print
            print ('<E> = %.6f     ' % (e_avg/float(N)),end='')
            print ('<|M|> = %.6f     ' % (m_avg/float(N)),end='')
            print ('<Cv> = %.6f     ' % (c/float(N*T**2)),end='')
            print ('<S> = %.6f' % (s/float(N*T)),end='')
            print()


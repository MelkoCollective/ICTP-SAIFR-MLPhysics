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

def save_parameters(sess, results_dir, rbm, epochs,L,B):
    weights, visible_bias, hidden_bias = sess.run([rbm.weights, rbm.visible_bias, rbm.hidden_bias])
    parameter_file_path =  '../data/tfim1d/parameters/parameters_L' + str(L)
    parameter_file_path += '_B' + str(B)
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
    B = args.t                      # magnetic field
    num_visible = args.L            # number of visible nodes
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
    train_dir = '../data/tfim1d/datasets/'  # Location of training data.
    trainName = '../data/tfim1d/datasets/tfim1d_L'+str(args.L)+'_B'+str(B)+'_train.txt'
    testName = '../data/tfim1d/datasets/tfim1d_L'+str(args.L)+'_B'+str(B)+'_test.txt'
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

    # define operations
    ops.lr=learning_rate
    ops.train = optimizer.minimize(cost, global_step=ops.global_step)
    ops.init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    
    logZ = rbm.exact_log_partition_function()
    placeholders.logZ = tf.placeholder(tf.float32) 
    NLL = rbm.neg_log_likelihood(placeholders.visible_samples,placeholders.logZ)
    path_to_wf='../data/tfim1d/wavefunctions/wavefunction_tfim1d_L'+str(args.L)+'_B'+str(B)+'.txt'
    wf=np.loadtxt(path_to_wf)
    psi_x = tf.exp(0.5*rbm.free_energy(placeholders.visible_samples))
    all_v_states= np.array(list(it.product([0, 1], repeat=num_visible)), dtype=np.float32)
    
    observer_file=open('../data/tfim1d/observables/training_observer.txt','w',0)
    observer_file.write('#      O')
    observer_file.write('        NLL')
    observer_file.write('         <H>')
    observer_file.write('     <|Sz|>')
    observer_file.write('       <Sx>')
    observer_file.write('\n')
    gibb_updates=10
    observer_samples=rbm.observer_sampling(gibb_updates)
    nbins=100
    
    e = np.zeros((num_samples))
    sX= np.zeros((num_samples))
    L = num_visible
    
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
                psix = sess.run(psi_x,feed_dict={placeholders.visible_samples: all_v_states}) 
 
                Ov = 0.0
                E = 0.0
                Sz = 0.0
                Sx = 0.0
                SzSz = np.zeros((L,L))

                for i in range(1<<num_visible): 
                    psix[i] /= m.exp(0.5*lz)
                    Ov += wf[i]*psix[i]
                
                for i in range(nbins):
                    
                    # Gibbs sampling
                    samples=sess.run(observer_samples)
                    spins = np.asarray((2*samples-1))
                    
                    # Compute average of longitudinal magnetizations
                    sZ_avg = np.mean(np.absolute(np.sum(spins,axis=1)))
                    
                    # Compute averages of energies
                    e.fill(0.0)
                    sX.fill(0.0)

                    for k in range(num_samples):
                        state = int(samples[k].dot(1 << np.arange(samples[k].size)[::-1]))
                        
                        # Compute the average of transverse magnetization
                        for i in range(L):
                            samples[k,i] = 1 - samples[k,i]
                            state_flip = int(samples[k].dot(1 << np.arange(samples[k].size)[::-1]))
                            sX[k] += float(psix[state_flip])/float(psix[state])
                            samples[k,i] = 1 - samples[k,i]
                        
                        # Compute the correlations ZZ
                        for i in range(L):
                            for j in range(L):
                                SzSz[i,j] += spins[k,i]*spins[k,j]/float(num_samples*nbins)

                        # Compute the Energy
                        for i in range(L-1):
                            e[k] += -spins[k,i]*spins[k,i+1]
                            samples[k,i] = 1 - samples[k,i]
                            state_flip = int(samples[k].dot(1 << np.arange(samples[k].size)[::-1]))
                            e[k] += -B*psix[state_flip]/psix[state]
                            samples[k,i] = 1 - samples[k,i]
                        e[k] += -spins[k,L-1]*spins[k,0]
                        samples[k,L-1] = 1 - samples[k,L-1]
                        state_flip = int(samples[k].dot(1 << np.arange(samples[k].size)[::-1]))
                        e[k] += -B*psix[state_flip]/psix[state]
                        samples[k,L-1] = 1 - samples[k,L-1]
                    sX_avg = np.mean(sX) 
                    e_avg = np.mean(e)
                    
                    E += (e_avg-E)/float(i+1)
                    Sz += (sZ_avg-Sz)/float(i+1)
                    Sx += (sX_avg-Sx)/float(i+1)
 
                               
                # Print observer on screen
                print ('Ov = %.6f     ' % Ov,end='')
                print ('NLL = %.6f     ' % nll,end='')
                print ('<H> = %.6f     ' % (E/float(L)),end='')
                print ('<|Sz|> = %.6f     ' % (Sz/float(L)),end='')
                print ('<Sx> = %.6f     ' % (Sx/float(L)),end='')
                print()
 
                # Save observer on file
                observer_file.write('%.6f   ' % Ov)
                observer_file.write('%.6f   ' % nll)
                observer_file.write('%.6f   ' % (E/float(L)))
                observer_file.write('%.6f   ' % (Sz/float(L)))
                observer_file.write('%.6f   ' % (Sx/float(L)))
                observer_file.write('\n')
                
                #observer_corr_file=open('../data/tfim1d/observables/training_observer_corr.txt','w')
                #for i in range(L):
                #    for j in range(L):
                #        observer_corr_file.write('%.6f  ' % SzSz[i,j])
                #    observer_corr_file.write('\n')

                epochs_done += 1


def sample(args):
       
    # Architecture
    B = args.t                      # magnetic field
    num_visible = args.L            # number of visible nodes
    num_hidden = args.nH            # number of hidden nodes
    
    # Load the RBM parameters
    path_to_params = '../data/tfim1d/parameters/parameters_nH'+str(num_hidden) + '_L'+str(args.L)+'_B'+str(B)+'.npz'
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
    placeholders = Placeholders()
    placeholders.visible_samples = tf.placeholder(tf.float32, shape=(None, num_visible), name='v') # placeholder for training data

    # Initialize tensorflow
    init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    logZ = rbm.exact_log_partition_function()
    placeholders.logZ = tf.placeholder(tf.float32) 
    psi_x = tf.exp(0.5*rbm.free_energy(placeholders.visible_samples))
    all_v_states= np.array(list(it.product([0, 1], repeat=num_visible)), dtype=np.float32)

    sX= np.zeros((num_samples))
    L = num_visible
    e = np.zeros((num_samples)) 
    with tf.Session() as sess:
        sess.run(init)
        lz = sess.run(logZ)
        psix = sess.run(psi_x,feed_dict={placeholders.visible_samples: all_v_states}) 
        #Ov=0.0
        for i in range(1<<num_visible): 
            psix[i] /= m.exp(0.5*lz)
        #    Ov += wf[i]*psix[i]
 
        for i in range(nbins):
            print ('bin %d\t\t' %i,end='')
            
            # Gibbs sampling
            _,samples=sess.run([hsamples,vsamples])
            spins = np.asarray((2*samples-1))
            
            # Compute average of longitudinal magnetizations
            sZ_avg = np.mean(np.absolute(np.sum(spins,axis=1)))
            
            # Compute averages of energies
            e.fill(0.0)
            sX.fill(0.0)
            for k in range(num_samples):
                state = int(samples[k].dot(1 << np.arange(samples[k].size)[::-1]))
                # Compute the average of transverse magnetization
                for i in range(L):
                    samples[k,i] = 1 - samples[k,i]
                    state_flip = int(samples[k].dot(1 << np.arange(samples[k].size)[::-1]))
                    sX[k] += float(psix[state_flip])/float(psix[state])
                    samples[k,i] = 1 - samples[k,i]
            
                # Compute the Energy
                for i in range(L-1):
                    e[k] += -spins[k,i]*spins[k,i+1]
                    samples[k,i] = 1 - samples[k,i]
                    state_flip = int(samples[k].dot(1 << np.arange(samples[k].size)[::-1]))
                    e[k] += -B*psix[state_flip]/psix[state]
                    samples[k,i] = 1 - samples[k,i]
                e[k] += -spins[k,L-1]*spins[k,0]
                samples[k,L-1] = 1 - samples[k,L-1]
                state_flip = int(samples[k].dot(1 << np.arange(samples[k].size)[::-1]))
                e[k] += -B*psix[state_flip]/psix[state]
                samples[k,L-1] = 1 - samples[k,L-1]
            
            e_avg = np.mean(e)
            sX_avg = np.mean(sX) 
 
            # Print observer on screen
            print ('<H> = %.6f     ' % (e_avg/float(L)),end='')
            print ('<|Sz|> = %.6f     ' % (sZ_avg/float(L)),end='')
            print ('<Sx> = %.6f     ' % (sX_avg/float(L)),end='')
            print()


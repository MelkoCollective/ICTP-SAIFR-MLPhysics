########## ICTP-SAIFR Minicourse on Machine Learning for Many-Body Physics ##########
### Roger Melko, Juan Carrasquilla, Lauren Hayward Sierens and Giacomo Torlai
### Tutorial 4: Sampling a Restricted Boltzmann Machine (RBM)
#####################################################################################

from __future__ import print_function
import tensorflow as tf
#import itertools as it
#from random import randint
from rbm import RBM
import numpy as np
#import math as m
#import argparse


# Input parameters:
L     = 4     #linear size of the system
num_visible         = L*L      #number of visible nodes
num_hidden          = 4        #number of hidden nodes

#temperature list for which there are trained RBM parameters stored in data_ising2d/parameters_solutions
T_list = [1.0,1.254,1.508,1.762,2.016,2.269,2.524,2.778,3.032,3.286,3.540]

# Read in nearest neighbours for the lattice lattice
path_to_lattice = 'data_ising2d/lattice2d_L'+str(L)+'.txt'
nn=np.loadtxt(path_to_lattice)

# Sampling parameters:
num_samples  = 500  # how many independent chains will be sampled
gibb_updates = 2    # how many gibbs updates per call to the gibbs sampler
nbins        = 1000 # number of calls to the RBM sampler

rbms = []
rbm_samples = []
for i in range(len(T_list)):
    T = T_list[i]
    path_to_params = 'data_ising2d/parameters/parameters_nH'+str(num_hidden) + '_L'+str(L)+'_T'+str(T)+'.npz'
    params = np.load(path_to_params)
    weights = params['weights']
    visible_bias = params['visible_bias']
    hidden_bias = params['hidden_bias']
    hidden_bias=np.reshape(hidden_bias,(hidden_bias.shape[0],1))
    visible_bias=np.reshape(visible_bias,(visible_bias.shape[0],1))

    # Initialize RBM class
    rbms.append(RBM(num_hidden=num_hidden, num_visible=num_visible, weights=weights, visible_bias=visible_bias,hidden_bias=hidden_bias, num_samples=num_samples))
    rbm_samples.append(rbms[i].stochastic_maximum_likelihood(gibb_updates))

# Initialize tensorflow
init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

# Thermodynamic observables
N = num_visible

with tf.Session() as sess:
    sess.run(init)

    for i in range(nbins):
        print ('bin %d\t' %i,end='')
        print() 
        fout = open('data_ising2d/observables/sampler_observer.txt','w')
 
        for t in range(len(T_list)):
            _,samples=sess.run(rbm_samples[t])
            spins = np.asarray((2*samples-1))

            m_avg = np.mean(np.absolute(np.sum(spins,axis=1)))
            e = np.zeros((num_samples))
            e2= np.zeros((num_samples))

            for k in range(num_samples):
                for i in range(N):
                    e[k] += -spins[k,i]*(spins[k,int(nn[i,0])]+spins[k,int(nn[i,1])])
                e2[k] = e[k]*e[k]
            e_avg = np.mean(e)
            e2_avg= np.mean(e2) 
            m2_avg = np.mean(np.multiply(np.sum(spins,axis=1),np.sum(spins,axis=1)))
            c = (e2_avg-e_avg*e_avg)
            s = (m2_avg-m_avg*m_avg)
            
            fout.write('%.6f  ' % (e_avg/float(N)))
            fout.write('%.6f  ' % (m_avg/float(N)))
            fout.write('%.6f  ' % (c/float(N*T_list[t]**2)))
            fout.write('%.6f\n' % (s/float(N*T_list[t])))

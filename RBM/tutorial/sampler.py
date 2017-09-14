from __future__ import print_function
from pprint import pformat
import tensorflow as tf
import itertools as it
from random import randint
from rbm import RBM
import numpy as np
import math as m
import argparse

#def main():
#    
#    # Initialize the command line parser
#    parser = argparse.ArgumentParser()
#    # Read command line arguments
#    parser.add_argument('model',type=str,help='model to sample') 
#    # Parse the arguments
#    
#    args = parser.parse_args()
#
#    if args.model == 'ising2d':
#        sample_ising2d()
#    
#    if args.model == 'tfim1d':
#        sample_tfim1d()


def ising2d():

    L=4
    temps=[1.0,1.254,1.508,1.762,2.016,
           2.269,2.524,2.778,3.032,3.286,3.540]
    num_visible = L*L     # number of visible nodes
    num_hidden = 4       # number of hidden nodes
    
    # Build lattice
    path_to_lattice = '../data/ising2d/lattice2d_L'+str(L)+'.txt'
    nn=np.loadtxt(path_to_lattice)
    
    # Sampling parameters
    num_samples=500 # how many independent chains will be sampled
    gibb_updates=2  # how many gibbs updates per call to the gibbs sampler
    nbins=1000      # number of calls to the RBM sampler      
    
    rbms = []
    rbm_samples = []
    for i in range(len(temps)):
        T = temps[i]
        path_to_params = '../data/ising2d/parameters/parameters_nH'+str(num_hidden) + '_L'+str(L)+'_T'+str(T)+'.npz'
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
            fout = open('../data/ising2d/observables/sampler_observer.txt','w')
     
            for t in range(len(temps)):
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
                fout.write('%.6f  ' % (c/float(N*temps[t]**2)))
                fout.write('%.6f\n' % (s/float(N*temps[t])))

def tfim1d():
    
    L=10
    fields = [0.2,0.4,0.6,0.8,1.0,
              1.2,1.4,1.6,1.8,2.0]
    
    num_visible = L     # number of visible nodes
    num_hidden = 10     # number of hidden nodes
    
    # Sampling parameters
    num_samples=500 # how many independent chains will be sampled
    gibb_updates=2 # how many gibbs updates per call to the gibbs sampler
    nbins=1000      # number of calls to the RBM sampler      
    
    class Placeholders(object):
        pass
    
    placeholders = Placeholders()
    placeholders.visible_samples = tf.placeholder(tf.float32, shape=(None, num_visible), name='v') # placeholder for training data
    
    rbms = []
    rbm_samples = []
    psi_list = []
    logZ = []
    all_v_states= np.array(list(it.product([0, 1], repeat=num_visible)), dtype=np.float32)
    for i in range(len(fields)):
        B = fields[i]
        path_to_params = '../data/tfim1d/parameters/parameters_nH'+str(num_hidden) + '_L'+str(L)+'_B'+str(B)+'.npz'
        params = np.load(path_to_params)
        weights = params['weights']
        visible_bias = params['visible_bias']
        hidden_bias = params['hidden_bias']
        hidden_bias=np.reshape(hidden_bias,(hidden_bias.shape[0],1))
        visible_bias=np.reshape(visible_bias,(visible_bias.shape[0],1))
    
        # Initialize RBM class
        rbms.append(RBM(num_hidden=num_hidden, num_visible=num_visible, weights=weights, visible_bias=visible_bias,hidden_bias=hidden_bias, num_samples=num_samples))
        rbm_samples.append(rbms[i].stochastic_maximum_likelihood(gibb_updates))
        logZ.append(rbms[i].exact_log_partition_function())
        psi_list.append(tf.exp(0.5*rbms[i].free_energy(placeholders.visible_samples)))
    
    # Initialize tensorflow
    init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    
    psix = []
    lz = []
    
    with tf.Session() as sess:
        sess.run(init)
        for i in range(len(fields)):
            lz.append(sess.run(logZ[i]))
            psix.append(sess.run(psi_list[i],feed_dict={placeholders.visible_samples: all_v_states})) 
            for j in range(1<<num_visible): 
                psix[i][j] /= m.exp(0.5*lz[i])
    
        for i in range(nbins):
            print ('bin %d\t' %i,end='')
            print() 
            fout = open('../data/tfim1d/observables/sampler_observer.txt','w')
            fcorr = open('../data/tfim1d/observables/sampler_observer_corr.txt','w') 
    
            for b in range(len(fields)):
                _,samples=sess.run(rbm_samples[b])
                spins = np.asarray((2*samples-1))
                
                # Compute average of longitudinal magnetizations
                sZ_avg = np.mean(np.absolute(np.sum(spins,axis=1)))
                
                # Compute averages of energies
                sX= np.zeros((num_samples))
                for k in range(num_samples):
                    state = int(samples[k].dot(1 << np.arange(samples[k].size)[::-1]))
                    # Compute the average of transverse magnetization
                    for i in range(L):
                        samples[k,i] = 1 - samples[k,i]
                        state_flip = int(samples[k].dot(1 << np.arange(samples[k].size)[::-1]))
                        sX[k] += float(psix[b][state_flip])/float(psix[b][state])
                        samples[k,i] = 1 - samples[k,i]
                    
                sX_avg = np.mean(sX) 
                
                if (b==4):
                    sZsZ = np.zeros((L,L))
                    for i in range(L):
                        for j in range(L):
                            for k in range(num_samples):
                                sZsZ[i,j] += spins[k,i]*spins[k,j]/float(num_samples)
                        
                            fcorr.write('%.10f   ' % sZsZ[i,j])
                        fcorr.write('\n')
                fout.write('%.6f  ' % (sZ_avg/float(L)))
                fout.write('%.6f  \n' % (sX_avg/float(L)))


if __name__ == '__main__':
    main()


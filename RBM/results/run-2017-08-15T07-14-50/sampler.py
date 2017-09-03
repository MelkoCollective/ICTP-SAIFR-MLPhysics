import numpy as np
import numpy as np
import sys, traceback
import itertools as it
from random import randint
import numpy, scipy.io
import tensorflow as tf
import sys
from rbm import RBM



def kron(x):
    return 1 * (np.sum(x,axis=1)==1)

# ['epochs', 'hidden_bias', 'weights', 'visible_bias']
 
b=np.load('parameters.npz_800.npz')
weights=b['weights']
visible_bias=b['visible_bias']
hidden_bias=b['hidden_bias']

# reading the parameters of the RBM 
#weights=np.load('weights.npy')
#hidden_bias=np.load('hidden_bias.npy')
#visible_bias=np.load('visible_bias.npy')
#weights=np.transpose(np.loadtxt('weights.txt',dtype=np.float32))
#hidden_bias=np.loadtxt('hidden_bias.txt',dtype=np.float32)
#visible_bias=np.loadtxt('visible_bias.txt',dtype=np.float32)

hidden_bias=np.reshape(hidden_bias,(hidden_bias.shape[0],1))
visible_bias=np.reshape(visible_bias,(visible_bias.shape[0],1))



num_hidden=hidden_bias.shape[0]
num_visible=visible_bias.shape[0]


num_samples=100 # how many independent chains will be sampled
gibb_updates=100 # how many gibbs updates per call to the gibbs sampler
nbins=100        # number of calls to the RBM sampler      

#defines the RBM based on the loaded parameters
rbm = RBM(num_hidden=num_hidden, num_visible=num_visible, weights=weights, visible_bias=visible_bias,
              hidden_bias=hidden_bias, num_samples=num_samples)

# samples the rbm using gibbs sampling
hsamples,vsamples=rbm.stochastic_maximum_likelihood(gibb_updates)

# getting amplitude of unormalized psi=sqrt(RBM(x)) for a matrix of configurations x
x=tf.placeholder(tf.float32, shape=(None, num_visible), name='v')

psi_x=tf.exp(0.5*rbm.free_energy(x))

init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

xm=np.eye(num_visible)


with tf.Session() as sess:
   sess.run(init)
   Ov=sess.run(psi_x,feed_dict={x: xm })  
   Ov=np.sum(Ov)
   Ovrbm=0.0
   Ovrbm2=0.0  
   for i in range(nbins):
       print 'bin ', i
       _,samples=sess.run([hsamples,vsamples])
       psix=sess.run(psi_x,feed_dict={x: samples }) 
       rr=np.sum(np.reshape(kron(samples),(samples.shape[0],1))/np.reshape(psix,(samples.shape[0],1))/num_visible)/num_samples 
       Ovrbm=Ovrbm+rr
       Ovrbm2=Ovrbm2+np.power(rr,2)

   Ovrbm2=Ovrbm2/nbins
   Ovrbm=Ovrbm/nbins   
   err=np.sqrt( np.abs(np.power(Ovrbm,2)- Ovrbm2 ))
  
   
   Od=np.sqrt(Ovrbm*Ov)
   print  Od, 0.5*np.sqrt(Ov/Ovrbm)*err  
                 




 

 






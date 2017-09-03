import tensorflow as tf
import input_data
import sys
import numpy as np
import matplotlib.pyplot as plt



#"linear size of the system "
lx=30

#parameters of the neural network and cost function
numberlabels=2 # Number of phases under consideration (2 for the Ising model on the square lattice) 
hiddenunits1=100 # number of hidden unites in the hidden layer
lamb=0.001 # regularization parameter 
beta=1.0 #``inverse temperature'' of the sigmoid neuron

#Parameters of the optimization
#batch size for the gradient descent 
bsize=1500
# number of iterations
niter=4000


# temperature list at which the training/test sets were generated
tlist=[1.0000000000000000,1.0634592657106510,1.1269185314213019,1.1903777971319529,1.2538370628426039,1.3172963285532548,1.3807555942639058,1.4442148599745568,1.5076741256852078,1.5711333913958587,1.6345926571065097,1.6980519228171607,1.7615111885278116,1.8249704542384626,1.8884297199491136,1.9518889856597645,2.0153482513704155,2.0788075170810667,2.1422667827917179,2.2057260485023691,2.3326445799236715,2.3961038456343227,2.4595631113449739,2.5230223770556250,2.5864816427662762,2.6499409084769274,2.7134001741875786,2.7768594398982298,2.8403187056088810,2.9037779713195322,2.9672372370301834,3.0306965027408346,3.0941557684514858,3.1576150341621370,3.2210742998727881,3.2845335655834393,3.3479928312940905,3.4114520970047417,3.4749113627153929,3.5383706284260401]

# Description of the input data 
Ntemp=40 # number of different temperatures used in the training and testing data
samples_per_T=250 # number of samples per temperature value in the testing set
Nord=20 # number of temperatures in the ordered phase

#reading the data in the directory txt 
mnist = input_data.read_data_sets(numberlabels,lx,'txt', one_hot=True)

print "reading sets ok"

#sys.exit("pare aqui")

# defining weighs and initlizatinon
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# defining the application of the sigmoid functions
def layers(x, W,b):
  return tf.nn.sigmoid(beta*tf.matmul(x, W)+b)

# defines the hidden layer argument for investigating what the neural net learns upon training         
def hidlay(x,W,b):
  return tf.matmul(x, W)+b

# defining the model: input(spin configuration) and label (Ferromagnet/paramagnet)

x = tf.placeholder("float", shape=[None, lx*lx]) # spin configuration
y_ = tf.placeholder("float", shape=[None, numberlabels]) # label in the form of a one hot vector

#first layer (hidden layer) 
#defining the weights and bias of the hidden layer
W_1 = weight_variable([lx*lx,hiddenunits1])
b_1 = bias_variable([hiddenunits1])

hl=hidlay(x,W_1,b_1)

#Apply a sigmoid

O1 = layers(x, W_1,b_1)

#second layer(output layer in this case)
W_2 = weight_variable([hiddenunits1,numberlabels])
b_2 = bias_variable([numberlabels])

O2=layers(O1,W_2,b_2)
y_conv=O2

#Train and Evaluate the Model

# cost function to minimize (with a small L2 regularization (lamb))

cross_entropy = tf.reduce_mean(-y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0) )-(1.0-y_)*tf.log(tf.clip_by_value(1.0-y_conv,1e-10,1.0))) +(lamb)*(tf.nn.l2_loss(W_1)+tf.nn.l2_loss(W_2))


#batch size for the gradient descent 
bsize=1500
# number of iterations
niter=4000 

#defining the optimizer to be used in the minimization of the cross entropy
optimizer= tf.train.AdamOptimizer(0.0005)

# train step amounts to minimizing the cross entropy using gradient descent computed by the optimizer
train_step = optimizer.minimize(cross_entropy)

#predictions
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) # checks the correct predictions by comparing the results of the neural net with the labels
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # measures the accuracy


# initializing the session 
sess = tf.Session()
sess.run(tf.initialize_all_variables())


# training
for i in range(niter):

  batch = mnist.train.next_batch(bsize)
  #batch=(mnist.train.images[:,:].reshape(bsize,lx*lx), mnist.train.labels[:,:].reshape((bsize,numberlabels)) )
  if i%100 == 0:
    train_accuracy = sess.run(accuracy,feed_dict={
        x:batch[0], y_: batch[1]})
    print "step %d, training accuracy %g"%(i, train_accuracy)
    print sess.run(cross_entropy,feed_dict={x: batch[0], y_: batch[1]})
    print "test accuracy %g"%sess.run(accuracy, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels})
  sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
print "test accuracy %g"%sess.run(accuracy, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels})


#producing plots of the results


f = open('nnout.dat', 'w')

# Average output of neural net over the test set
ii=0
for i in range(Ntemp):
  av=0.0
  for j in range(samples_per_T):
        batch=(mnist.test.images[ii,:].reshape((1,lx*lx)),mnist.test.labels[ii,:].reshape((1,numberlabels)))     
        res=sess.run(y_conv,feed_dict={x: batch[0], y_: batch[1]})
        av=av+res 
        ii=ii+1 
  av=av/samples_per_T
  f.write(str(i)+' '+str(tlist[i])+' '+str(av[0,0])+' '+str(av[0,1])+"\n")  
  #print i,av   
f.close()       

# Average accuracy vs temperature over the test set
f = open('acc.dat', 'w')
for ii in range(Ntemp):
  batch=(mnist.test.images[ii*samples_per_T:ii*samples_per_T+samples_per_T,:].reshape(samples_per_T,lx*lx), mnist.test.labels[ii*samples_per_T:ii*samples_per_T+samples_per_T,:].reshape((samples_per_T,numberlabels)) )
  train_accuracy = sess.run(accuracy,feed_dict={
        x:batch[0], y_: batch[1]}) 
  f.write(str(ii)+' '+str(tlist[ii])+' '+str(train_accuracy)+"\n") #


f.close()




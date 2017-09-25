import tensorflow as tf
import input_data
import sys


lx=16 # linear size of the lattice
training=4000 # size of training set
bsize=200  # batch size
Ntemp=2 # number of temperatures 0, infinity
samples_per_T_test=2500 # samples per each temperature in the test set


numberlabels=2 # number of labels (T=0, infinity)
mnist = input_data.read_data_sets(numberlabels,lx+1,'txt', one_hot=True) # reading the data

print "reading sets ok"

#sys.exit("pare aqui")

# defining weighs and initlizatinon
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# defining the convolutional and max pool layers
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# defining the model

x = tf.placeholder("float", shape=[None, (lx+1)*(lx+1)*2]) # placeholder for the spin configurations (lx+1) because of periodic boundary conditions
y_ = tf.placeholder("float", shape=[None, numberlabels]) # labels


#first layer 
# convolutional layer # 2x2 patch size, 2 channel (2 color), 64 feature maps computed
nmaps1=64
W_conv1 = weight_variable([2, 2, 2,nmaps1])
# bias for each of the feature maps
b_conv1 = bias_variable([nmaps1])

# applying a reshape of the data to get the two dimensional structure back
x_image = tf.reshape(x, [-1,(lx+1),(lx+1),2]) # with PBC 

#We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# weights and bias of the fully connected (fc) layer. In this case everything looks one dimensional because it is fully connected
nmaps2=64

W_fc1 = weight_variable([(lx) * (lx) * nmaps1,nmaps2 ]) # weights for the FC layer 
b_fc1 = bias_variable([nmaps2]) # bias vector


h_1_flat = tf.reshape(h_conv1, [-1, (lx)*(lx)*nmaps1]) # reshaping the 
# then apply the ReLU with the fully connected weights and biases.
h_fc1 = tf.nn.relu(tf.matmul(h_1_flat, W_fc1) + b_fc1)

# Dropout: To reduce overfitting, we will apply dropout before the readout layer. We create a placeholder for the probability that a neuron's output is kept during dropout. This allows us to turn dropout on during training, and turn it off during testing. TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them, so dropout just works without any additional scaling.

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer. Finally, we add a softmax layer, just like for the one layer softmax regression above.

# weights and bias
W_fc2 = weight_variable([nmaps2, numberlabels])
b_fc2 = bias_variable([numberlabels])

# apply a softmax layer
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#Train and Evaluate the Model
# cost function to minimize

#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(training):
  batch = mnist.train.next_batch(bsize)
  if i%100 == 0:
    train_accuracy = sess.run(accuracy,feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print "step %d, training accuracy %g"%(i, train_accuracy)
    print "test accuracy %g"%sess.run(accuracy, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}) 

  sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print "test accuracy %g"%sess.run(accuracy, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})



saver = tf.train.Saver([W_conv1, b_conv1, W_fc1,b_fc1,W_fc2,b_fc2])
save_path = saver.save(sess, "./model.ckpt")
print "Model saved in file: ", save_path

#producing data to get the plots we like

f = open('nnout.dat', 'w')

#output of neural net
ii=0
for i in range(Ntemp):
  av=0.0
  for j in range(samples_per_T_test):
        batch=(mnist.test.images[ii,:].reshape((1,2*(lx+1)*(lx+1))),mnist.test.labels[ii,:].reshape((1,numberlabels)))
        res=sess.run(y_conv,feed_dict={x: batch[0], y_: batch[1],keep_prob: 1.0})
        av=av+res
        #print ii, res
        ii=ii+1
  av=av/samples_per_T_test
  f.write(str(i)+' '+str(av[0,0])+' '+str(av[0,1])+"\n") 
f.close() 


f = open('acc.dat', 'w')

# accuracy vs temperature
for ii in range(Ntemp):
  batch=(mnist.test.images[ii*samples_per_T_test:ii*samples_per_T_test+samples_per_T_test,:].reshape(samples_per_T_test,2*(lx+1)*(lx+1)), mnist.test.labels[ii*samples_per_T_test:ii*samples_per_T_test+samples_per_T_test,:].reshape((samples_per_T_test,numberlabels)) )
  train_accuracy = sess.run(accuracy,feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
  f.write(str(ii)+' '+str(train_accuracy)+"\n")
f.close()
  


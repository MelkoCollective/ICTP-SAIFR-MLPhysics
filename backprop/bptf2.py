# A bit of setup
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# activation function
def sigmoid(x):
  return 1. / (1. + np.exp(-x))
# derivative of activation
def d_sigmoid(x):
    return sigmoid(x)*(1.0-sigmoid(x))  

# figure setup
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(0)

# this is a little data set of spirals with 3 branches
N = 50 # number of points per branch
D = 2 # dimensionality of the vectors to be learned
K = 3 # number of branches
X = np.zeros((N*K,D)) # matrix containing the dataset
y = np.zeros(N*K, dtype='uint8') # labels
# data generation
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

# plotting dataset
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim([-1,1])
plt.ylim([-1,1])
fig.savefig('spiral_raw.png')

# Computational graph for tensorflow

x = tf.placeholder(tf.float32, [None, D])
yl = tf.placeholder(tf.int32,[None])

h = 100 # size of hidden layer

W1 = tf.Variable(tf.random_normal([ D, h], mean=0.0, stddev=0.01, dtype=tf.float32))
b1 = tf.Variable(tf.zeros([h]))
W2 = tf.Variable(tf.random_normal([ h, K], mean=0.0, stddev=0.01, dtype=tf.float32))
b2 = tf.Variable(tf.zeros([K]))

# forward pass
z1 = tf.matmul(x, W1) + b1
a1 = tf.nn.sigmoid(z1)
z2 = tf.matmul(a1, W2) + b2
a2 = tf.nn.softmax(z2)
#cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yl,logits=z2)) # tf shortcut 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(yl,depth=K) * tf.log(a2), reduction_indices=[1])) # a little more explicit

# Shortest and least error-prone version using tensorflow layers
#a1 = tf.layers.dense(inputs=x, units=h, activation=tf.nn.sigmoid)
#a2 = tf.layers.dense(inputs=hidden_layer, units=K)
#cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yl,logits=scores))

# some hyperparameters
step_size = 1.0 #e-0

train_step = tf.train.GradientDescentOptimizer(step_size).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # gradient descent loop
    num_examples = X.shape[0]
    for i in range(20000):
        #
        sess.run(train_step, feed_dict={x: X,yl:y})
        if i % 1000 == 0:
            loss=sess.run(cross_entropy,feed_dict={x: X,yl:y})
            print "iteration %d: loss %f" % (i, loss)
  


    #hidden_layer = np.maximum(0, np.dot(X, W) + b)
    #scores = np.dot(hidden_layer, W2) + b2
    scores_=sess.run(z2,feed_dict={x:X,yl:y})
    predicted_class = np.argmax(scores_, axis=1)
    print 'training accuracy: %.2f' % (np.mean(predicted_class == y))
    
    # getting the trained weights for plotting the classifier
    W1=sess.run(W1)
    W2=sess.run(W2)
    b1=sess.run(b1) 
    b2=sess.run(b2)
    
    
## plot the resulting classifier
# plot the resulting classifier

h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.dot(sigmoid( np.dot(np.c_[xx.ravel(), yy.ravel()], W1) + b1), W2) + b2
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
fig.savefig('spiral_net_results.png')

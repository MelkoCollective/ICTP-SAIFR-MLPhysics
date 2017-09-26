
import numpy as np
import matplotlib.pyplot as plt

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


# Defining a fully connected neural network and initializing their weight matrices randomly
h = 50 # size of hidden layer
W1 = 0.01 * np.random.randn(D,h) # First layer weights
b1 = np.zeros((1,h)) # first layer bias vector
W2 = 0.01 * np.random.randn(h,K) # Second layer weight matrix
b2 = np.zeros((1,K)) # bias vector second layer

# some hyperparameters
step_size = 1.0 #e-0 training rate
# reg=0.01 # in case we want regularization

# gradient descent loop
num_examples = X.shape[0] # size of dataset
for i in range(20000):
 
  # forward pass 
  z1=np.dot(X, W1) + b1 # z1
  a1 = sigmoid(z1)
  #a1 = np.maximum(0,z1) # if we want relu
  z2 = np.dot(a1, W2) + b2 #(scores)
  # Activation in the last layer is a softmax=compute the class probabilities
  exp_scores = np.exp(z2)
  a2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K] (probs)
  
  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(a2[range(num_examples),y])
  data_loss = np.sum(corect_logprobs)/num_examples
  #reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
  loss = data_loss # + reg_loss# if we want regularization as an exercise

  # print every 1000 iteration
  if i % 1000 == 0:
    print "iteration %d: loss %f" % (i, loss)
 
  # BACKPROPAGATION 
  # compute the gradient last layer (first compute the "error d2")
  d2 = a2
  d2[range(num_examples),y] -= 1
  d2 /= num_examples
  
  # backpropate the gradient to the parameters
  # first backprop into parameters W2 and b2
  #dW2 = np.dot(hidden_layer.T, dscores)
  dW2 = np.dot(a1.T, d2)
  db2 = np.sum(d2, axis=0, keepdims=True)

  # next backprop the error d2 into hidden layer to get d1
  d1 = d_sigmoid(z1)*np.dot(d2, W2.T)
  # if we want ReLu, we backprop the ReLU non-linearity
  #d1 = np.dot(d2, W2.T)
  #d1[a1 <= 0] = 0

  # finally into W1,b1
  dW1 = np.dot(X.T, d1)
  db1 = np.sum(d1, axis=0, keepdims=True)
  
  # if we add regularization, get its constribution to the gradient
  #dW2 += reg * W2
  #dW1 += reg * W1
  
  # perform a parameter update
  W1 += -step_size * dW1
  b1 += -step_size * db1
  W2 += -step_size * dW2
  b2 += -step_size * db2



# plotting and evaluating the training
a1 = sigmoid(np.dot(X, W1) + b1)
z2 = np.dot(a1, W2) + b2
predicted_class = np.argmax(z2, axis=1)
print 'training accuracy: %.2f' % (np.mean(predicted_class == y))

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

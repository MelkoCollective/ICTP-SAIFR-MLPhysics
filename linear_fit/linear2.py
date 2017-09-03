import tensorflow as tf
import numpy as np

# creating the ``training'' data
trX = np.linspace(-1, 1, 101) #
trY = 2 * trX + np.random.randn(*trX.shape) * 0.10 + 1.0  # create a y value which is approximately linear but with some random noise


# create symbolic placeholders to hold the variables during the optimization
X = tf.placeholder("float") # holder for the x values  
Y = tf.placeholder("float") # holder for the y values


def model(X, w,b): 
    return tf.mul(X, w)+b #  it is just a straigth line X*w+b so this model line is pretty simple


w = tf.Variable(0.0, name="weights") # create a tensorflow variable for the slope of the line 
b=tf.Variable(0.0,name="bias")       # create a tensorflow variable for the intercept of the line

y_model = model(X, w,b) # define the output of the model which is just  y=X*w+b


cost =tf.reduce_mean(tf.square(Y - y_model)) # use square error for cost function

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct an optimizer to minimize cost and fit line to my data

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize variables (in this case just variable W)
    tf.initialize_all_variables().run()

    for i in range(4000):
        sess.run(train_op, feed_dict={X: trX, Y: trY})
            

    print(sess.run(w),sess.run(b))  # It should be something around 2

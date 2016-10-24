from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import numpy as np
import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_function=None,):
    # add one more layer and return the output of this layer

    Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=0.02))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global y
    y_pre = sess.run(y, feed_dict={x: v_xs, keep_prob: 1})  #feed the value xs into prediction to get the y_pre is a probobility
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1)) #the output has 10 dimension(0,0,1,0,0,0,0,0,0,0), and we get the max one, if the max one is 3(0,0,1,0,0,0,0,0,0,0) and the max value of purpose(v_xs) also is 3(0,0,1,0,0,0,0,0,0,0), and we can say the learning is working 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))   #if learning is working, we will continue to reduce the cost
    result = sess.run(accuracy, feed_dict={x: v_xs, y_: v_ys, keep_prob: 1})
    return result


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
# x isn't a specific value. It's a placeholder, a value that we'll input when we ask TensorFlow to run a computation.
# We want to be able to input any number of MNIST images, each flattened into a 784-dimensional vector. 
"""
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
"""
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([1, 10]) + 0.1,)

# add hidden layer
l1 = add_layer(x, 784, 100, activation_function=tf.nn.relu)
#l2 = add_layer(l1, 300, 50, activation_function=tf.nn.relu)
# add output layer
y = add_layer(l1, 100, 10,  activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#In this case, we ask TensorFlow to minimize cross_entropy using the gradient descent algorithm with a learning rate of 0.5. Gradient descent is a simple procedure, where TensorFlow simply shifts each variable a little bit in the direction that reduces the cost. 
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Now we have our model set up to train. One last thing before we launch it, we have to create an operation to initialize the variables we created.
init = tf.initialize_all_variables()

#We can now launch the model in a Session, and now we run the operation that initializes the variables:
sess = tf.Session()
sess.run(init)

#Let's train -- we'll run the training step 1000 times!
for i in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(200)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.6}) 
    if i%200 == 0:
        print (compute_accuracy(batch_xs, batch_ys))  #We run train_step feeding in the batches data to replace the placeholders.
"""       
#That gives us a list of booleans. To determine what fraction are correct, we cast to floating point numbers and then take the mean.
#For example, [True, False, True, True] would become [1,0,1,1] which would become 0.75.        
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
"""
print(compute_accuracy(
            mnist.test.images, mnist.test.labels))
#print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


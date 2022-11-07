import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# move warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

# read data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# create model
# input layer
x = tf.placeholder(tf.float32, [None, 784])  # input images
y_ = tf.placeholder(tf.float32, [None, 10])  # input labels

# hidden layer
hidden_layer_count = 15
w1 = tf.Variable(tf.zeros([784, hidden_layer_count]))
b1 = tf.Variable(tf.zeros([hidden_layer_count]))
y1 = tf.sigmoid(tf.matmul(x, w1) + b1)

# output layer
output_layer_count = 10
w2 = tf.Variable(tf.zeros([hidden_layer_count, output_layer_count]))
b2 = tf.Variable(tf.zeros(output_layer_count))
y2 = tf.nn.softmax(tf.matmul(y1, w2) + b2)

# loss function & optimization algorithm
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y2))
train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cross_entropy)

# new session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 1000 == 0:
        loss = sess.run(cross_entropy, {x: batch_xs, y_: batch_ys})
        print("Iter: %s loss: %s" % (i, loss))

# test
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y2, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy: ', sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

tf.logging.set_verbosity(old_v)
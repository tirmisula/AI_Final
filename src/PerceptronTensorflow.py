'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import utils
import os


def init_x_y(parameters):
    return tf.placeholder('float', [None, parameters['image_size']]), \
           tf.placeholder("float", [None, parameters['n_classes']])

# tf Graph input


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Store layers weight & bias
def train(parameters: dict, path: str = os.path.join('./model', 'model'),saver: tf.train.Saver = None) -> None:

    weights = {
        'h1': tf.Variable(tf.random_normal([parameters['image_size'], parameters['n_hidden_1']])),
        'h2': tf.Variable(tf.random_normal([parameters['n_hidden_1'], parameters['n_hidden_2']])),
        'out': tf.Variable(tf.random_normal([parameters['n_hidden_2'], parameters['n_classes']]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal(parameters['n_hidden_1'])),
        'b2': tf.Variable(tf.random_normal([parameters['n_hidden_2']])),
        'out': tf.Variable(tf.random_normal([parameters['n_classes']]))
    }
    x, y = init_x_y(parameters)
    pred = multilayer_perceptron(x, weights, biases)
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=parameters['learning_rate']).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()
    batch_size = parameters['batch_size']

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(parameters['training_epochs']):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
                # Display logs per epoch step
            if epoch % parameters['display_step'] == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        print("Optimization Finished!")
        if saver:
            saver.save(sess, save_path=path)


        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

if __name__ == '__main__':
    parameters = utils.load_parameters()

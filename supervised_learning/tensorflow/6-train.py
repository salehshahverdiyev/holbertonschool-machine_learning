#!/usr/bin/env python3
'''
    Script Documentation
'''
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    '''
        builds, trains, and saves a neural network classifier:
        X_train is a numpy.ndarray containing the training input data
        Y_train is a numpy.ndarray containing the training labels
        X_valid is a numpy.ndarray containing the validation input data
        Y_valid is a numpy.ndarray containing the validation labels
        layer_sizes is a list containing the number of nodes
            in each layer of the network
        activations is a list containing the activation functions
            for each layer of the network
        alpha is the learning rate
        iterations is the number of iterations to train over
        save_path designates where to save the model
        Add the following to the graph's collection
        placeholders x and y
        tensors y_pred, loss, and accuracy
        operation train_op
        After every 100 iterations, the 0th iteration, and
            iterations iterations, print the following:
        After {i} iterations: where i is the iteration
        \tTraining Cost: {cost} where {cost} is the training cost
        \tTraining Accuracy: {accuracy} where {accuracy}
            is the training accuracy
        \tValidation Cost: {cost} where {cost} is the validation cost
        \tValidation Accuracy: {accuracy} where {accuracy}
            is the validation accuracy
        Reminder: the 0th iteration represents the model
            before any training has occurred
        After training has completed, save the model to save_path
        You may use the following imports:
        calculate_accuracy =
            __import__('3-calculate_accuracy').calculate_accuracy
        calculate_loss = __import__('4-calculate_loss').calculate_loss
        create_placeholders =
            __import__('0-create_placeholders').create_placeholders
        create_train_op = __import__('5-create_train_op').create_train_op
        forward_prop = __import__('2-forward_prop').forward_prop
        You are not allowed to use tf.saved_model
        Returns: the path where the model was saved
    '''
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    for step in range(iterations + 1):
        t_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
        t_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
        v_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
        v_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
        if step % 100 == 0 or step == iterations:
            print("After {} iterations:".format(step))
            print("\tTraining Cost: {}".format(t_cost))
            print("\tTraining Accuracy: {}".format(t_acc))
            print("\tValidation Cost: {}".format(v_cost))
            print("\tValidation Accuracy: {}".format(v_acc))
        if step < iterations:
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})
    return saver.save(sess, save_path)

#!/usr/bin/env python3
'''
    Script Documentation
'''
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    '''
        trains a model using mini-batch gradient descent:
        network is the model to train
        data is a numpy.ndarray of shape (m, nx) containing the input data
        labels is a one-hot numpy.ndarray of
            shape (m, classes) containing the labels of data
        batch_size is the size of the batch used
            for mini-batch gradient descent
        epochs is the number of passes through data
            for mini-batch gradient descent
        verbose is a boolean that determines if output
            should be printed during training
        shuffle is a boolean that determines whether to shuffle
            the batches every epoch.
            Normally, it is a good idea to shuffle, but for reproducibility,
                we have chosen to set the default to False.
        validation_data is the data to validate the model with, if not None
        early_stopping is a boolean that indicates whether
            early stopping should be used
        early stopping should only be performed if validation_data exists
        early stopping should be based on validation loss
        patience is the patience used for early stopping
        learning_rate_decay is a boolean that indicates whether
            learning rate decay should be used
        learning rate decay should only be performed
            if validation_data exists
        the decay should be performed using inverse time decay
        the learning rate should decay in a stepwise fashion after each epoch
        each time the learning rate updates, Keras should print a message
        alpha is the initial learning rate
        decay_rate is the decay rate
        Returns: the History object generated after training the model
    '''
    def learning_rate(epochs):
        '''
            Updates the learning rate using inverse time decay.
            lr = alpha / (1 + decay_rate * epoch)
            Returns:
                float: The updated learning rate for the current epoch.
        '''
        return alpha / (1 + decay_rate * epochs)

    callbacks = []
    if (validation_data and learning_rate_decay):
        early_stopping = K.callbacks.LearningRateScheduler(learning_rate, 1)
        callbacks.append(early_stopping)

    if (validation_data and early_stopping):
        early_stopping = K.callbacks.EarlyStopping(patience=patience)
        callbacks.append(early_stopping)

    history = network.fit(data, labels, batch_size=batch_size,
                          epochs=epochs, validation_data=validation_data,
                          verbose=verbose, shuffle=shuffle,
                          callbacks=callbacks)
    return history

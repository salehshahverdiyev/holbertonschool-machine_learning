#!/usr/bin/env python3
'''
    Script Documentation
'''
import tensorflow.keras as K


def save_config(network, filename):
    '''
        Saves a model's configuration in JSON format:
        network is the model whose configuration should be saved
        filename is the path of the file that
            the configuration should be saved to
        Returns: None
    '''
    json_string = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_string)
    return None


def load_config(filename):
    '''
        loads a model with a specific configuration:
        filename is the path of the file containing
            the model's configuration in JSON format
        Returns: the loaded model
    '''
    with open(filename, "r") as f:
        network_string = f.read()
    network = K.models.model_from_json(network_string)
    return network

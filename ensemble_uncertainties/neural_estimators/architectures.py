
"""Network architectures for our feed forward neural estimators."""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def deep_architecture(output_activation):
    """Our default deep architecture for property prediction:

    in | 256 ReLU | 128 ReLU | 16 ReLU | 1 
        
    Parameters
    ----------
    output_activation : str
        Name of the activation in the final neuron (tf.keras.activations)
            
    Returns
    -------
    function
        A function that takes the number of variables
        (int) and returns the (uncompiled) model
    """
    def make_model(n_vars):
        model = Sequential()
        model.add(Dense(256, input_shape=(n_vars,), activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation=output_activation))
        return model
    return make_model


def shallow_architecture(output_activation):
    """Our default shallow architecture for property prediction:

    in | 128 ReLU | 1 
        
    Parameters
    ----------
    output_activation : str
        Name of the activation in the final neuron (tf.keras.activations)
            
    Returns
    -------
    function
        A function that takes the number of variables
        (int) and returns the (uncompiled) model
    """
    def make_model(n_vars):
        model = Sequential()
        model.add(Dense(128, input_shape=(n_vars,), activation='relu'))
        model.add(Dense(1, activation=output_activation))
        return model
    return make_model

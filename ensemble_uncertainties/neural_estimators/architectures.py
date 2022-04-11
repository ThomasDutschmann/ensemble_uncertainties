
"""Network architectures for our feed forward neural estimators."""

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model, Sequential


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


def deep_architecture_dropout(dropout_rate=0.2):
        """Our default deep architecture for regression, with dropout:

        in | 256 ReLU | Dropout | 128 ReLU | Dropout | 16 ReLU | Dropout | 1
            
        Parameters
        ----------
        dropout_rate : float \in [0, 1]
            Fraction of dropped out weights, default: 0.2
                
        Returns
        -------
        function
            A function that takes the number of variables
            (int) and returns the (uncompiled) model
        """
        def make_model(n_vars):
            # Create model. To apply dropout during inference,
            # the functional format must be used:
            inp = Input(shape=(n_vars,))
            x = Dropout(dropout_rate)(inp)
            x = Dense(256, activation='relu')(x)
            x = Dropout(dropout_rate)(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(dropout_rate)(x)
            x = Dense(16, activation='relu')(x)
            x = Dropout(dropout_rate)(x)
            out = Dense(1, activation='linear')(x)
            architecture = Model(inp, out)
            return architecture
        return make_model


def deep_architecture_mc_dropout(dropout_rate=0.2):
        """Our default deep architecture for regression, with dropout and
        the setting 'training=True' to perform MC dropout during inference.

        in | 256 ReLU | Dropout | 128 ReLU | Dropout | 16 ReLU | Dropout | 1
            
        Parameters
        ----------
        dropout_rate : float \n [0, 1]
            Fraction of dropped out weights, default: 0.2
                
        Returns
        -------
        function
            A function that takes the number of variables
            (int) and returns the (uncompiled) model
        """
        def make_model(n_vars):
            # Create model. To apply dropout during inference,
            # the functional format must be used:
            inp = Input(shape=(n_vars,))
            x = Dropout(dropout_rate)(inp)
            x = Dense(256, activation='relu')(x)
            x = Dropout(dropout_rate)(x, training=True)
            x = Dense(128, activation='relu')(x)
            x = Dropout(dropout_rate)(x, training=True)
            x = Dense(16, activation='relu')(x)
            x = Dropout(dropout_rate)(x, training=True)
            out = Dense(1, activation='linear')(x)
            architecture = Model(inp, out)
            return architecture
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

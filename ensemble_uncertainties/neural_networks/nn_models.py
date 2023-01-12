
"""Wrappers for our feed forward neural estimators."""

import logging
import os

import numpy as np

import tensorflow as tf

from ensemble_uncertainties.neural_networks.architectures import (
    deep_architecture,
    deep_architecture_dropout,
    deep_architecture_mc_dropout,
    shallow_architecture
)

from ensemble_uncertainties.constants import (
    BATCH_SIZE,
    EPOCHS,
    MEMORY_GROWTH,
    RANDOM_SEED
)


# Turn off logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
# Make deterministic
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(RANDOM_SEED)
# Set growth ability of GPU memory (if available)
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], MEMORY_GROWTH)


class NeuralNetwork:
    """ Wrapper class for TensorFlow feed forward estimators to
    dynamically exchange architectures and to overcome
    incompatibilities with scikit-learn models.
    """

    def __init__(self, loss, metric, architecture, epochs=EPOCHS,
            batch_size=BATCH_SIZE):
        """Initializer.
        
        Parameters
        ----------
        loss : str
            Name of the loss function (see tf.keras.losses)
        metric : str
            Name of the error metric (see tf.keras.metrics)
        architecture : function
            A function that takes an input shape and
            returns a tf.keras.Sequential model
        epochs : int
            Number of epochs, default: EPOCHS
        batch_size : int
            Size of batch, default: BATCH_SIZE
        """
        # Required
        self.loss = loss
        self.metric = metric
        self.architecture = architecture
        # Available as default arguments
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        """Compiles and fits the underlying TF model.

        Parameters
        ----------
        X : DataFrame
            Matrix of dependent variables. Index and header must be provided,
            name of the index column must be 'id'.
        y : DataFrame
            Vector of output variables.Index and header must be provided,
            name of the index column must be 'id'.        
        """
        self.model = self.architecture(X.shape[1])
        self.model.compile(loss=self.loss, metrics=[self.metric])
        _ = self.model.fit(
            X, y, 
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        
    def predict(self, X):
        """Applies the predict function of the underlying TF model.
        
        Parameters
        ----------
        X : DataFrame
            Test set to predict
            
        Returns
        -------
        np.array
            Predictions
        """
        predictions = self.model.predict(X, verbose=0)
        return predictions

    def save(self, path):
        """Applies the save function of the underlying TF model.
        
        Parameters
        ----------
        path : str
            File path to store the H5-model in
        """
        self.model.save(path)


class DeepNeuralClassifier(NeuralNetwork):
    """Deep classification-specific extension, based on:"""

    __doc__ += NeuralNetwork.__doc__

    def __init__(self, epochs=EPOCHS, batch_size=BATCH_SIZE):
        super().__init__(
            loss='binary_crossentropy',
            metric='accuracy',
            architecture=deep_architecture('sigmoid'),
            epochs=epochs,
            batch_size=batch_size
        )


class DeepNeuralRegressor(NeuralNetwork):
    """Deep regression-specific extension, based on:"""

    __doc__ += NeuralNetwork.__doc__

    def __init__(self, epochs=EPOCHS, batch_size=BATCH_SIZE):
        super().__init__(
            loss='mse',
            metric='mean_squared_error',
            architecture=deep_architecture('linear'),
            epochs=epochs,
            batch_size=batch_size
        )


class ShallowNeuralClassifier(NeuralNetwork):
    """Shallow classification-specific extension, based on:"""

    __doc__ += NeuralNetwork.__doc__

    def __init__(self, epochs=EPOCHS, batch_size=BATCH_SIZE):
        super().__init__(
            loss='binary_crossentropy',
            metric='accuracy',
            architecture=shallow_architecture('sigmoid'),
            epochs=epochs,
            batch_size=batch_size
        )


class ShallowNeuralRegressor(NeuralNetwork):
    """Shallow regression-specific extension, based on:"""

    __doc__ += NeuralNetwork.__doc__

    def __init__(self, epochs=EPOCHS, batch_size=BATCH_SIZE):
        super().__init__(
            loss='mse',
            metric='mean_squared_error',
            architecture=shallow_architecture('linear'),
            epochs=epochs,
            batch_size=batch_size
        )


class DeepDropoutRegressor(NeuralNetwork):
    """Regression-specific extension with dropout, based on:"""

    __doc__ += NeuralNetwork.__doc__

    def __init__(self, epochs=1000, batch_size=BATCH_SIZE, dropout_rate=0.2):
        super().__init__(
            loss='mse',
            metric='mean_squared_error',
            architecture=deep_architecture_dropout(
                dropout_rate=dropout_rate
            ),
            # Dropout requires more epochs!
            epochs=epochs,
            batch_size=batch_size
        )


class DeepMCDropoutRegressor(NeuralNetwork):
    """Regression-specific extension for MC dropout, based on:"""

    __doc__ += NeuralNetwork.__doc__

    def __init__(self, epochs=1000, batch_size=BATCH_SIZE, dropout_rate=0.2,
            n_pred=100):
        super().__init__(
            loss='mse',
            metric='mean_squared_error',
            architecture=deep_architecture_mc_dropout(
                dropout_rate=dropout_rate
            ),
            # Dropout requires more epochs!
            epochs=epochs,
            batch_size=batch_size
        )
        self.n_pred = n_pred

    def mc_predict(self, X):
        """Predicts with MC dropout, averages single per-object
        predictions and takes their standard deviation as UQ estimator.
        
        Parameters
        ----------
        X : DataFrame
            Test set to predict
            
        Returns
        -------
        np.array, np.array
            Predictions, uncertainties
        """
        p_range = range(self.n_pred)
        predictions = np.array([self.model.predict(X) for _ in p_range])
        predictions_vals = predictions[:, :, 0]
        means = predictions_vals.mean(axis=0)
        sdevs = predictions_vals.std(axis=0)
        return means, sdevs

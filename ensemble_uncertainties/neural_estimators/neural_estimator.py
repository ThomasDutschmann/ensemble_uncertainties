
"""Wrappers for our feed forward neural estimators."""

import logging
import os

import tensorflow as tf

from ensemble_uncertainties.neural_estimators.architectures import (
    deep_architecture,
    shallow_architecture
)

from ensemble_uncertainties.constants import (
    BATCH_SIZE,
    EPOCHS,
    MEMORY_GROWTH,
    RANDOM_SEED
)

# Turn off learning monitoring
logging.getLogger('tensorflow').setLevel(logging.FATAL)
# Make deterministic
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(RANDOM_SEED)
# Set growth ability of memory
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], MEMORY_GROWTH)


class NeuralEstimator:
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
        predictions = self.model.predict(X)
        return predictions

    def save(self, path):
        """Applies the save function of the underlying TF model.
        
        Parameters
        ----------
        path : str
            File path to store the H5-model in
        """
        self.model.save(path)


class DeepNeuralClassifier(NeuralEstimator):
    """Deep classification-specific extension, based on:"""

    __doc__ += NeuralEstimator.__doc__

    def __init__(self, epochs=EPOCHS, batch_size=BATCH_SIZE):
        super().__init__(
            loss='binary_crossentropy',
            metric='accuracy',
            architecture=deep_architecture('sigmoid'),
            epochs=epochs,
            batch_size=batch_size
        )


class DeepNeuralRegressor(NeuralEstimator):
    """Deep regression-specific extension, based on:"""

    __doc__ += NeuralEstimator.__doc__

    def __init__(self, epochs=EPOCHS, batch_size=BATCH_SIZE):
        super().__init__(
            loss='mse',
            metric='mean_squared_error',
            architecture=deep_architecture('linear'),
            epochs=epochs,
            batch_size=batch_size
        )


class ShallowNeuralClassifier(NeuralEstimator):
    """Shallow classification-specific extension, based on:"""

    __doc__ += NeuralEstimator.__doc__

    def __init__(self, epochs=EPOCHS, batch_size=BATCH_SIZE):
        super().__init__(
            loss='binary_crossentropy',
            metric='accuracy',
            architecture=shallow_architecture('sigmoid'),
            epochs=epochs,
            batch_size=batch_size
        )


class ShallowNeuralRegressor(NeuralEstimator):
    """Shallow regression-specific extension, based on:"""

    __doc__ += NeuralEstimator.__doc__

    def __init__(self, epochs=EPOCHS, batch_size=BATCH_SIZE):
        super().__init__(
            loss='mse',
            metric='mean_squared_error',
            architecture=shallow_architecture('linear'),
            epochs=epochs,
            batch_size=batch_size
        )

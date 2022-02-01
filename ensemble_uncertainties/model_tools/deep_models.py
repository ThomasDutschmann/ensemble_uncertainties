
"""Wrappers for our deep feed forward estimators."""

import logging
import os

import tensorflow as tf

from constants import BATCH_SIZE, EPOCHS, RANDOM_SEED

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Turn off learning monitoring
logging.getLogger('tensorflow').setLevel(logging.FATAL)
# Make deterministic
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(RANDOM_SEED)


class DeepEstimator:
    """ Wrapper class for TensorFlow feed forward estimators to
    dynamically exchange architectures and to overcome
    incompatibilities with scikit-learn models.
    """

    def __init__(self, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """Initializer.
        
        Params
        ------
        epochs : int
            Number of epochs, default: EPOCHS
        batch_size : int
            Size of batch, default: BATCH_SIZE
        """
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
        
    def architecture(self, n_vars):
        """Our default architecture for property prediction:

        in | 256 ReLU | 128 ReLU | 16 ReLU | 1 
        
        Params
        ------
        n_vars : int
            Number of independent variables
            
        Returns
        -------
        keras.engine.sequential.Sequential
            The (uncompiled) model
        """
        model = Sequential()
        model.add(Dense(256, input_shape=(n_vars,), activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation=self.output_activation))
        return model
        
    def predict(self, X):
        """Applies the predict function of the underlying TF model.
        
        Params
        ------
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
        
        Params
        ------
        path : str
            File path to store the H5-model in
        """
        self.model.save(path)
        
        
class DeepClassifier(DeepEstimator):
    """Classification-specific extension of the following base class:"""

    __doc__ += DeepEstimator.__doc__

    def __init__(self, epochs=EPOCHS, batch_size=BATCH_SIZE):
        super().__init__(epochs, batch_size)
        self.loss = 'binary_crossentropy'
        self.metric = 'accuracy'
        self.output_activation = 'sigmoid'
        
        
class DeepRegressor(DeepEstimator):
    """Regression-specific extension of the following base class:"""

    __doc__ += DeepEstimator.__doc__

    def __init__(self, epochs=EPOCHS, batch_size=BATCH_SIZE):
        super().__init__(epochs, batch_size)
        self.loss = 'mse'
        self.metric = 'mean_squared_error'
        self.output_activation = 'linear'

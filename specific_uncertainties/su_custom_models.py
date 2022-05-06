
"""Wrapper classes for regressors that can be tuned to yield uncertainties."""

import numpy as np

from ensemble_uncertainties.neural_estimators.neural_estimator import (
    DeepMCDropoutRegressor
)

from sklearn.ensemble import RandomForestRegressor


class MCDropoutRegressorWrapper(DeepMCDropoutRegressor):
    """Wrapper for MC dropout deep regressor, based on:"""

    __doc__ += DeepMCDropoutRegressor.__doc__

    def __init__(self, **mcdropout_kwargs):
        """Calls initializer of underlying core class.

        Parameters
        ----------        
        **mcdropout_kwargs
            Arguments are forwarded to DeepMCDropoutRegressor(...).
        """
        super().__init__(**mcdropout_kwargs)

    def uq_predict(self, X):
        """Calls the underlying MC dropout predict method.
        
        Parameters
        ----------
        X : DataFrame
            Test set to predict
            
        Returns
        -------
        np.array, np.array
            Predictions, uncertainties
        """
        means, sdevs = self.mc_predict(X)
        return means, sdevs


class UQRandomForestRegressor(RandomForestRegressor):
    """Wrapper for RandomForestRegressor, yields decision tree SDevs."""

    def __init__(self, **rfr_kwargs):
        """Calls initializer of underlying core class.

        Parameters
        ----------
        **rfr_kwargs
            Arguments are forwarded to RandomForestRegressor(...).
        """
        super().__init__(**rfr_kwargs)

    def uq_predict(self, X):
        """Yields the predictions of the complete RF regressor but also
        computes the standard deviation of the single
        decision-tree-predictions as uncertainty estimator.

        Parameters
        ----------
        X : DataFrame
            Test set to predict
            
        Returns
        -------
        np.array, np.array
            Predictions, uncertainties
        """
        predictions = self.predict(X)
        trees = self.estimators_
        dt_preds = np.array([tree.predict(X) for tree in trees])
        sdevs = dt_preds.std(axis=0)
        return predictions, sdevs

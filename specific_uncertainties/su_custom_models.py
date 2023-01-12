
"""Wrapper classes for regressors that can be tuned to yield uncertainties."""

from ensemble_uncertainties.constants import V_THRESHOLD
import numpy as np
import pandas as pd

from ensemble_uncertainties.automatize import run_evaluation
from ensemble_uncertainties.constants import (
    N_REPS,
    N_SPLITS,
    RANDOM_SEED,
    V_THRESHOLD
)
from ensemble_uncertainties.neural_networks.nn_models import (
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


class KFoldEnsembleRegressorWrapper():
    """Wrapper that turns every regressor into a k-fold-ensemble predictor."""

    def __init__(self, model, **evaluator_kwargs):
        """Initializer.

        Parameters
        ----------
        model : object
            A fittable regressor object, e.g. sklearn.svm.SVR()
            that implements the methods fit(X, y) and predict(X)
        **evaluator_kwargs
            Arguments that are forwarded to automatize.run_evaluation(...).
        """
        self.model = model
        self.evaluator_kwargs = evaluator_kwargs

    def custom_follow_up(self, evaluator):
        """Incorporates the evaluator-object from
        the evaluation-run as attribute.

        Parameters
        ----------
        evaluator : Evaluator
            Evaluator-object for which 'perform(...)' has been run
        """
        self.performed_evaluator = evaluator

    def fit(self, X, y):
        """Fits the underlying evaluator regressor.
        
        Parameters
        ----------
        X : DataFrame
            Train inputs
        y : DataFrame
            Train outputs
        """
        # Make y to DataFrame
        y_df = pd.DataFrame(X.index)
        y_df['y'] = y
        y_df.set_index('id', inplace=True)
        # Perform evaluator
        run_evaluation(
            model=self.model,
            task='regression',
            X=X,
            y=y_df,
            verbose=False,
            repetitions=self.evaluator_kwargs['repetitions'],
            n_splits=self.evaluator_kwargs['n_splits'],
            seed=self.evaluator_kwargs['seed'],
            scale=self.evaluator_kwargs['scale'], 
            v_threshold=self.evaluator_kwargs['v_threshold'],
            bootstrapping=False,
            path=None,
            store_all=False,
            args=None,
            follow_up=self.custom_follow_up
        )

    def uq_predict(self, X):
        """Predicts the output of X by every
        model-set in the performed evaluator.
        
        Parameters
        ----------
        X : DataFrame
            Test set to predict
            
        Returns
        -------
        np.array, np.array
            Predictions, uncertainties        
        """
        # Initialize
        evaluator = self.performed_evaluator
        predictions_df = pd.DataFrame(index=X.index)
        # Go through each iteration of the evaluator
        for rep_index in range(evaluator.repetitions):
            for split_index in range(evaluator.n_splits):
                column_name = f'{rep_index}_{split_index}'
                # Get current scaler, threshold, and model
                scaler = evaluator.scalers[rep_index][split_index]
                vt = evaluator.vt_filters[rep_index][split_index]
                model = evaluator.models[rep_index][split_index]
                # Scale/filter test inputs
                X_te_sc = pd.DataFrame(scaler.transform(X),
                    index=X.index, columns=X.columns)
                X_te_sc_filt = pd.DataFrame(X_te_sc,
                    index=X.index, columns=X.columns[vt.get_support()])
                # Predict
                predictions = model.predict(X_te_sc_filt)
                predictions_df[column_name] = predictions
        # Store latest predictions as attribute
        self.last_predictions = predictions_df
        # Return requested values
        mean = predictions_df.mean(axis=1).values
        sdevs = predictions_df.std(axis=1).values
        return mean, sdevs

    def RFRWrapper():
        """Returns wrapped sklearn RandomForestRegressor."""
        regressor = KFoldEnsembleRegressorWrapper(
            model=RandomForestRegressor(),
            **{'repetitions': N_REPS,
            'n_splits': N_SPLITS,
            'seed': RANDOM_SEED,
            'scale': False, 
            'v_threshold': V_THRESHOLD}
        )
        return regressor

    def MCDropoutWrapper():
        """Returns wrapped MC Dropout Regressor from neural_estimators."""
        regressor = KFoldEnsembleRegressorWrapper(
            model=DeepMCDropoutRegressor(),
            **{'repetitions': N_REPS,
            'n_splits': N_SPLITS,
            'seed': RANDOM_SEED,
            'scale': True, 
            'v_threshold': V_THRESHOLD}
        )
        return regressor

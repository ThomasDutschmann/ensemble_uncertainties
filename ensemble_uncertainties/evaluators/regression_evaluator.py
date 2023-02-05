
"""Evaluator for regression case."""

import pandas as pd

from ensemble_uncertainties.constants import (
    N_REPS,
    N_SPLITS,
    RANDOM_SEED,
    V_THRESHOLD
)
from ensemble_uncertainties.evaluators.evaluator import Evaluator
from ensemble_uncertainties.utils.ad_assessment import (
    auco, rmses_frac, spearman_coeff
)

from sklearn.metrics import r2_score


class RegressionEvaluator(Evaluator):
    """Regression-specific extension of the Evaluator base class:"""

    __doc__ += Evaluator.__doc__

    def __init__(self, model, verbose=True, repetitions=N_REPS,
            n_splits=N_SPLITS, seed=RANDOM_SEED, scale=True,
            v_threshold=V_THRESHOLD, bootstrapping=False):
        super().__init__(
            model=model,
            verbose=verbose,
            repetitions=repetitions,
            n_splits=n_splits,
            seed=seed,
            scale=scale,
            v_threshold=v_threshold,
            bootstrapping=bootstrapping
        )
        self.task = 'regression'
        self.metric = r2_score
        self.metric_name = 'R^2'

    # Override unimplemented method
    def handle_predictions(self, predictions):
        """Computes means and standard deviations.

        Parameters
        ----------
        predictions : DataFrame
            The predictions from a single repetition

        Returns
        -------
        DataFrame, float
            The data as DataFrame and the evaluation metric (R^2)
        """
        results = pd.DataFrame(index=self.X.index)
        # Compute final predictions by average
        results['predicted'] = predictions.mean(axis=1)
        # Compute sdev of ensemble predictions by output distribution
        results['sdep'] = predictions.std(axis=1)
        # Take subset of y (important when doing bootstrapping)
        results.dropna(subset=['predicted'], inplace=True)
        adjusted_y = self.y.loc[results.index]
        results['resid'] = adjusted_y['y'] - results['predicted']
        quality = self.metric(adjusted_y['y'], results['predicted'])
        return results, quality

    def finalize(self):
        """Run parent function, compute UQ evaluation metrics."""
        super().finalize()
        resids = self.test_ensemble_preds['resid']
        uncertainties = self.test_ensemble_preds['sdep']
        oracle_rmses, measure_rmses = rmses_frac(resids, uncertainties)
        area = auco(oracle_rmses, measure_rmses)
        rho = spearman_coeff(resids, uncertainties)
        self.auco = area
        self.rho = rho

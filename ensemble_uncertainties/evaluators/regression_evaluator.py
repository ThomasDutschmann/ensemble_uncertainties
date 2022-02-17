
"""Evaluator for regression case."""

import pandas as pd

from ensemble_uncertainties.constants import N_SPLITS, RANDOM_SEED, REPS

from ensemble_uncertainties.evaluators.evaluator import Evaluator

from sklearn.metrics import r2_score


class RegressionEvaluator(Evaluator):
    """Regression-specific extension of the Evaluator base class:"""

    __doc__ += Evaluator.__doc__

    def __init__(self, model, verbose=True, repetitions=REPS,
            n_splits=N_SPLITS, seed=RANDOM_SEED, scale=True):
        super().__init__(
            model=model,
            verbose=verbose,
            repetitions=repetitions,
            n_splits=n_splits,
            seed=seed,
            scale=scale
        )
        self.task = 'regression'
        self.metric = r2_score
        self.metric_name = 'R^2'

    # override
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
        results['resid'] = self.y['y'] - results['predicted']
        quality = self.metric(self.y['y'], results['predicted'])
        return results, quality


"""Evaluator for classification case."""

import pandas as pd

from ensemble_uncertainties.constants import (
    N_REPS,
    N_SPLITS,
    RANDOM_SEED,
    V_THRESHOLD
)
from ensemble_uncertainties.evaluators.evaluator import Evaluator

from sklearn.metrics import accuracy_score


class ClassificationEvaluator(Evaluator):
    """Classification-specific extension of the Evaluator base class:"""

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
        self.task = 'classification'
        self.metric = accuracy_score
        self.metric_name = 'ACC'

    # override
    def handle_predictions(self, predictions):
        """Computes major vote predictions and posterior probabilities.

        Parameters
        ----------
        predictions : DataFrame
            The predictions from a single repetition

        Returns
        -------
        DataFrame, float
            The data as DataFrame and the evaluation metric (ACC)
        """
        results = pd.DataFrame(index=self.X.index)
        # Compute final predictions by major vote
        results['predicted'] = predictions.mean(axis=1).round(0)
        # Compute posterior probabilities by class frequency
        results['probA'] = predictions.mean(axis=1)
        results['probB'] = 1 - results['probA']
        # Take subset of y (important when doing bootstrapping)
        results.dropna(subset=['predicted'], inplace=True)
        adjusted_y = self.y.loc[results.index]
        results['correct'] = adjusted_y['y'] == results['predicted']
        quality = self.metric(adjusted_y['y'], results['predicted'])
        return results, quality


"""Evaluator for classification case."""

import pandas as pd

from ensemble_uncertainties.constants import N_SPLITS, RANDOM_SEED, N_REPS

from ensemble_uncertainties.evaluators.evaluator import Evaluator

from sklearn.metrics import accuracy_score


class ClassificationEvaluator(Evaluator):
    """Classification-specific extension of the Evaluator base class:"""

    __doc__ += Evaluator.__doc__

    def __init__(self, model, verbose=True, repetitions=N_REPS,
            n_splits=N_SPLITS, seed=RANDOM_SEED, scale=True):
        super().__init__(
            model=model,
            verbose=verbose,
            repetitions=repetitions,
            n_splits=n_splits,
            seed=seed,
            scale=scale
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
        results['correct'] = self.y['y'] == results['predicted']
        quality = self.metric(self.y['y'], results['predicted'])
        return results, quality

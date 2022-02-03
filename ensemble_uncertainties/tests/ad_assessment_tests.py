
"""Tests concerning the computation of evaluation metrics."""

import unittest

import pandas as pd

from ensemble_uncertainties.utils.ad_assessment import (
    cumulative_accuracy,
    rmses_frac
)


class MathTests(unittest.TestCase):

    def test_rmses_fracs(self):
        """Evaluate confidence estimation on hard-coded example."""
        results = pd.DataFrame()
        results['id'] = ['A', 'B', 'C', 'D', 'E']
        results['error'] = [.1, .2, .3, .4, .5]
        results['uncertainty'] = [.5, .4, .3, .2, .1]
        results = results.set_index('id')
        # Compute expected results by hand
        #
        # RMSE of all predictions:
        # MSE = (.5^2 + .4^2 + .3^2 + .2^2 + .1^2) / 5
        #     = (.25 + .16 + .09 + .04 + 0.1) / 5
        #     = .55 / 5
        #     = .11
        # RMSE = SQRT(.11) = .33166
        #
        # Oracle RMSEs:
        # RMSE_0 = .33166 (including all predictions)
        # RMSE_1 = SQRT((.4^2 + .3^2 + .2^2 + .1^2) / 4) (without .5)
        #        = SQRT(.3 / 4) = SQRT(.075)
        #        = .27386
        # RMSE_2 = SQRT((.3^2 + .2^2 + .1^2) / 3) = .21602
        # RMSE_3 = SQRT((.2^2 + .1^2) / 2) = .15811
        # RMSE_4 = SQRT(.1^2) = .1
        #
        # Measure RMSEs (reverse order of oracle RMSEs):
        # RMSE_0 = .33166 (including all predictions)
        # RMSE_1 = SQRT((.5^2 + .4^2 + .3^2 + .2^2) / 4) (without .1)
        #        = SQRT(.54 / 4) = SQRT(.135)
        #        = .36742
        # RMSE_2 = SQRT((.5^2 + .4^2 + .3^2) / 3) = .40825
        # RMSE_3 = SQRT((.5^2 + .4^2) / 2) = .45277
        # RMSE_4 = SQRT(.5^2) = .5
        #
        resids = results['error']
        uncertainties = results['uncertainty']
        expected_oracle_rmses = [.33166, .27386, .21602, .15811, .1]
        expected_measure_rsmes = [.33166, .36742, .40825, .45277, .5]
        oracle_rmses, measure_rmses = rmses_frac(resids, uncertainties)
        for expected, actual in zip(expected_oracle_rmses, oracle_rmses):
            self.assertAlmostEqual(expected, actual, 5)
        for expected, actual in zip(expected_measure_rsmes, measure_rmses):
            self.assertAlmostEqual(expected, actual, 5)

    def test_cumulative_accuracy(self):
        """Evaluate accuracy accumulation on hard-coded example."""
        # Define true labels
        labels = pd.DataFrame()
        labels['id'] = ['A', 'B', 'C', 'D', 'E', 'F']
        labels['y'] = [1, 1, 1, 0, 0, 0]
        labels = labels.set_index('id')
        # Define output of a posterior estimation
        results = pd.DataFrame()
        results['id'] = ['A', 'B', 'C', 'D', 'E', 'F']
        # Simulate miss-classification of first and last object
        results['preds_1'] = [0, 1, 1, 0, 0, 1]
        # Assign corresponding posterior estimations
        results['probs_1'] = [.3, .8, .9, .2, .1, .7]
        results = results.set_index('id')
        # Compute cumulative accuracies
        accuracies_1 = cumulative_accuracy(
            labels['y'],
            results['preds_1'],
            results['probs_1']
        )
        # Miss-classifications come at the end, so each cumulative accuracy
        # should be equal or smaller to the cumulative accuracy before
        for i in range(1, len(accuracies_1)):
            self.assertTrue(accuracies_1[i] <= accuracies_1[i-1])
        # As the cumulative accuracy merely considers the order of the
        # posterior probabilities, these probabilites should lead to the same
        # cumulative accuracies:
        results['probs_2'] = [.49, .52, .53, .48, .47, .51]
        accuracies_2 = cumulative_accuracy(
            labels['y'],
            results['preds_1'],
            results['probs_2']
        )
        for acc1, acc2 in zip(accuracies_1, accuracies_2):
            self.assertEqual(acc1, acc2)
        # Define deliberately misleading posteriors such that the
        # miss-classified objects come first, followed by the correct
        # predictions. Compute expected results by hand:
        results['probs_3'] = [.1, .8, .8, .2, .2, .9]
        # Expected accuracies: First two wrong predictions (0, 0),
        # followed by only correct predictions (1, 1, 1, 1).
        # Expected total correct predictions:
        # [0, 0+0, 0+0+1, 0+0+1+1, 0+0+1+1+1, 0+0+1+1+1+1]
        # = [0, 0, 1, 2, 3, 4]
        # Expected accuracies (fractions of correct predictions):
        # [0/1, 0/2, 1/3, 2/4, 3/5, 4/6]
        # = [0, 0, 1/3, 1/2, 3/5, 2/3]
        expected_accuracies = [.0, .0, 1/3, .5, .6, 2/3]
        accuracies_3 = cumulative_accuracy(
            labels['y'],
            results['preds_1'],
            results['probs_3']
        )
        print(accuracies_3)
        for acc, expected_acc in zip(accuracies_3, expected_accuracies):
            self.assertAlmostEqual(acc, expected_acc, 5)        


if __name__ == '__main__':
    unittest.main()

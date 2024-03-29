
"""Library of supportive functions (unit conversions, uncertainty evaluation
metrics, etc).
"""

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.metrics import accuracy_score, r2_score


def rmses_frac(resids, uncertainties, frac=1.0):
    """Evaluates error reduction efficiency based on RMSE.

    Parameters
    ----------
    resids : Series
        Residuals
    uncertainties : Series
        Uncertainty measure values
    frac : float [0, 1]
        Fraction of predictions to remove, default: 1.0

    Returns
    -------
    list
        RMSEs for all steps
    """
    # Define up until which index predictions should be removed
    threshold = int(frac * len(resids))
    if frac == 1.0:
        threshold = len(resids)
    # Sort by residual
    oracle = list(sorted(np.absolute(resids.values)))[::-1]
    # Sort by uncertainty
    sorted_uncertainties = uncertainties.abs().sort_values(ascending=False)
    measure = list(resids.reindex(sorted_uncertainties.index).values)
    # Initialize
    oracle_rmses = list()
    measure_rmses = list()
    # Compute
    for i in range(threshold):
        # Compute oracle RMSEs
        oracle_partition = oracle[i:]
        n = len(oracle_partition)
        oracle_rmse = np.sqrt(sum([res**2 for res in oracle_partition]) / n)
        oracle_rmses.append(oracle_rmse)
        # Compute RMSEs according to uncertainty measure
        measure_partition = measure[i:]
        measure_rmse = np.sqrt(sum([res**2 for res in measure_partition]) / n)
        measure_rmses.append(measure_rmse)
    return oracle_rmses, measure_rmses


def auco(oracle_rmses, measure_rmses, normalize=False):
    """Computes the Area-Under-the-Confidence-Oracle error, see
    https://doi.org/10.1021/acs.jcim.9b00975 for more details.

    Parameters
    ----------
    oracle_rmses : list
        RMSEs in the ideal case
    measure_rmses : list
        RMSEs when using the uncertainty measure to evaluate
    normalize : bool
        Whether the 100% RMSE (including all predictions) should
        be set to 1.0, default: False

    Returns
    -------
    float
        Sum of all differences between oracle_rmses and measure_rmses
    """
    orac, meas = oracle_rmses, measure_rmses
    if normalize:
        orac = oracle_rmses / oracle_rmses[0]
        meas = measure_rmses / measure_rmses[0]
    area = sum([m - o for o, m in zip(orac, meas)])
    return area


def spearman_coeff(resids, uncertainties):
    """Computes Spearman's rank correlation coefficient.
    
    Parameters
    ----------
    resids : Series
        Residuals
    uncertainties : Series
        Uncertainty measure values

    Returns
    -------
    float
        The coefficient using scipy's spearmanr function
    """
    coeff = spearmanr(resids.abs(), uncertainties)[0]
    return coeff


def compute_uq_qualities(predictions, uncertainties, y):
    """Assigns performances and summarizes predictions/uncertainties.

    Parameters
    ----------
    predictions : 1D Numpy Array
        Predicted outputs
    uncertainties : 1D Numpy Array
        UQ values
    y : 1D Numpy Array
        Observed outputs

    Returns
    -------
    DataFrame, float, float
        Table of result, predictive quality, uncertainty quality
    """
    results = pd.DataFrame()
    # Compute final predictions by average
    results['predicted'] = predictions
    # Compute sdev of ensemble predictions by output distribution
    results['uq'] = uncertainties
    results['resid'] = y - predictions
    predictive_quality = r2_score(y, predictions)
    uncertainty_quality =  spearman_coeff(results['resid'], results['uq'])
    return results, predictive_quality, uncertainty_quality


def cumulative_accuracy(y, predicted, probabilities, start_frac=.05):
    """Evaluates accumulation efficiency of classification predictions.

    Parameters
    ----------
    y : Series
        True outputs
    predicted : Series
        Predicted class labels
    probabilities : Series
        Frame with uncertainty measure values (posterior probabilities)
    start_frac : float [0, 1]
        Fraction of predictions to start with, default: 0.05

    Returns
    -------
    list
        Accuracies for all steps
    """
    # Define starting index
    starting_index = int(start_frac*len(y))
    # Fix if smaller than 1
    if starting_index == 0:
        starting_index = 1
    # Sort by probabilities
    certainties = pd.Series(probabilities.values - .5,
        index=probabilities.index).abs()
    sorted_certainties = certainties.sort_values(ascending=False)
    sorted_pred = list(predicted.reindex(sorted_certainties.index).values)
    # Sort true outputs
    sorted_y = list(y.reindex(sorted_certainties.index).values)
    # Initialize
    accuracies = list()
    # Compute
    for i in range(starting_index, len(y)+1):
        y_partition = sorted_y[:i]
        pred_partition = sorted_pred[:i]
        current_accuracy = accuracy_score(y_partition, pred_partition)
        accuracies.append(current_accuracy)
    return accuracies

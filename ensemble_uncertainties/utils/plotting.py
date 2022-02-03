
"""Function library for plotting.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ensemble_uncertainties.utils.ad_assessment import (
    rmses_frac,
    cumulative_accuracy
)

from ensemble_uncertainties.constants import DPI, DEF_COLOR

from sklearn.metrics import roc_curve


# Settings
mpl.rcParams['figure.dpi'] = DPI


def plot_r2(evaluator, color_tr='C0', color_te='C1', path=''):
    """Plots observed vs. predicted scatter plot.

    Parameters
    ----------
    evaluator : EnsembleADEvaluator
        Applied evaluator
    color_tr : str
        Name of the color for train predictions, default: 'C0'
    color_te : str
        Name of the color for test predictions, default: 'C1'
    path : str
        Path of the file to store the plot, default: '' (no storing)
    """
    # Get data from evaluator
    y = evaluator.y['y']
    tr_preds = evaluator.train_ensemble_preds['predicted']
    te_preds = evaluator.test_ensemble_preds['predicted']
    tr_r2 = evaluator.train_ensemble_quality
    te_r2 = evaluator.test_ensemble_quality
    # Get corner values of the outputs/predictions
    smallest = min(min(y), min(te_preds.values), min(tr_preds.values))
    biggest = max(min(y), max(te_preds.values), max(tr_preds.values))
    # Plot
    plt.figure(figsize=(5, 5))
    plt.grid(zorder=1000)
    plt.plot(y, tr_preds, 'o', zorder=101, markersize=4, label=None,
        color=color_tr, mfc='none', alpha=.7)
    plt.plot(y, te_preds, 'o', zorder=100, markersize=4, label=None,
        color=color_te, mfc='none', alpha=.7)
    plt.scatter([], [], label=f'Train, $R^2$: {tr_r2:.3f}',
        color=color_tr, facecolor='none')
    plt.scatter([], [], label=f'Test,  $R^2$: {te_r2:.3f}',
        color=color_te, facecolor='none')
    plt.plot([smallest-.2, biggest+.2], [smallest-.2, biggest+.2], zorder=100,
        color='k', label='$\hat{y}$ = $y$')
    plt.xlim(smallest-.2, biggest+.2)
    plt.ylim(smallest-.2, biggest+.2)
    plt.xlabel('$y$')
    plt.ylabel('$\hat{y}$')
    plt.legend()
    if path:
        plt.savefig(f'{path}r2.png', bbox_inches='tight', pad_inches=0.01)


def plot_confidence(evaluator, frac=1.0, text=None, path=''):
    """Plots confidence curve.

    Parameters
    ----------
    evaluator : EnsembleADEvaluator
        Applied evaluator
    frac : float
        The fraction of covered outputs ([0.0, 1.0]), default: .5
    text : str
        Text to put in the plot, default: None
    path : str
        Path of the file to store the plot, default: '' (no storing)

    Returns
    -------
    float
        Area between the ideal curve and the measure curve
    """
    # Get data from evaluator
    resids = evaluator.test_ensemble_preds['resid']
    uncertainties = evaluator.test_ensemble_preds['sdep']
    oracle_rmses, measure_rmses = rmses_frac(resids, uncertainties, frac=frac)
    x_space = np.linspace(0.0, 100.0*frac, len(oracle_rmses))
    plt.figure(figsize=(5, 5))
    plt.grid(zorder=1000)
    plt.plot(x_space, oracle_rmses, label='Best (oracle)', color='k')
    plt.plot(x_space, measure_rmses, label='Uncertainty', color=DEF_COLOR,
        zorder=100)
    # Compute area
    area = sum([(m - o) for m, o in zip(measure_rmses, oracle_rmses)])
    # Put descriptions
    if text:
        plt.text(90.0*frac, .9*oracle_rmses[0], text, fontsize=20)
    plt.xlabel('Percentile')
    plt.ylabel('RMSE')
    plt.legend(loc='upper right')
    if path:
        plt.savefig(f'{path}confidence.png', bbox_inches='tight',
            pad_inches=0.01)
    return area


def plot_scatter(evaluator, path=''):
    """Plots uncertainties vs. residuals scatter plot.

    Parameters
    ----------
    evaluator : EnsembleADEvaluator
        Applied evaluator
    path : str
        Path of the file to store the plot, default: '' (no storing)
    """
    # Get data from evaluator
    resids = evaluator.test_ensemble_preds['resid'].abs()
    uncertainties = evaluator.test_ensemble_preds['sdep']
    plt.figure(figsize=(5, 5))
    plt.grid(zorder=1000)
    plt.plot(uncertainties, resids, 'o', zorder=101, markersize=4,
        color=DEF_COLOR, mfc='none', alpha=.5)
    plt.xlabel('Uncertainty')
    plt.ylabel('Absolute residuals')
    if path:
        plt.savefig(f'{path}scatter.png', bbox_inches='tight',
            pad_inches=0.0)


def plot_roc(evaluator, path=''):
    """Plots ROC curve.

    Parameters
    ----------
    evaluator : EnsembleADEvaluator
        Applied evaluator
    path : str
        Path of the file to store the plot, default: '' (no storing)
    """
    # Get data from evaluator
    y = evaluator.y['y']
    te_probs = evaluator.test_ensemble_preds['probA']
    fpr, tpr, _ = roc_curve(y, te_probs)
    # Plot
    plt.figure(figsize=(5, 5))
    plt.grid(zorder=1000)
    plt.plot(fpr, tpr, color=DEF_COLOR, zorder=100, label='ROC')
    plt.plot([0.0, 1.0], [0.0, 1.0], zorder=100,
        color='k', label='Random', linestyle='dashed')    
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.legend()
    if path:
        plt.savefig(f'{path}roc.png', bbox_inches='tight',
            pad_inches=0.0)


def plot_cumulative_accuracy(evaluator, start_frac=.05, path=''):
    """Plots cumulative accuracy plot.

    Parameters
    ----------
    evaluator : EnsembleADEvaluator
        Applied evaluator
    start_frac : float [0, 1]
        Fraction of predictions to start with, default: 0.05
    path : str
        Path of the file to store the plot, default: '' (no storing)
    """
    # Get data from evaluator
    y = evaluator.y['y']
    te_preds = evaluator.test_ensemble_preds['predicted']
    te_probs = evaluator.test_ensemble_preds['probA']
    accuracies = cumulative_accuracy(y, te_preds, te_probs,
        start_frac=start_frac)
    x_space = np.linspace(100*start_frac, 100.0, len(accuracies))
    # Plot
    plt.figure(figsize=(5, 5))
    plt.grid(zorder=1000)
    plt.plot(x_space, accuracies, color=DEF_COLOR, zorder=100)
    plt.xlabel('Percentile')
    plt.ylabel('Cumulative accuracy')
    if path:
        plt.savefig(f'{path}cumulative.png', bbox_inches='tight',
            pad_inches=0.01)

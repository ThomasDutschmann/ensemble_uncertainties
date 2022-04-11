
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

from sklearn.metrics import r2_score, roc_curve


# Settings
mpl.rcParams['figure.dpi'] = DPI


def plot_r2(y, tr_preds, te_preds, path='', show=False):
    """Plots observed vs. predicted scatter plot.

    Parameters
    ----------
    y : Series
        True values
    tr_preds : Series
        Train predictions
    te_preds : Series
        Test predictions
    path : str
        Path of the file to store the plot, default: '' (no storing)
    show : bool
        If True, plt.show() will be called, default: False
    """
    # Compute scores
    tr_r2 = r2_score(y, tr_preds)
    te_r2 = r2_score(y, te_preds)
    # Get corner values of the outputs/predictions
    smallest = min(min(y), min(te_preds.values), min(tr_preds.values))
    biggest = max(min(y), max(te_preds.values), max(tr_preds.values))
    # Plot
    plt.figure(figsize=(5, 5))
    plt.grid(zorder=1000)
    plt.plot(y, tr_preds, 'o', zorder=101, markersize=4, label=None,
        color='C0', mfc='none', alpha=.7)
    plt.plot(y, te_preds, 'o', zorder=100, markersize=4, label=None,
        color='C1', mfc='none', alpha=.7)
    plt.scatter([], [], label=f'Train, $R^2$: {tr_r2:.3f}',
        color='C0', facecolor='none')
    plt.scatter([], [], label=f'Test,  $R^2$: {te_r2:.3f}',
        color='C1', facecolor='none')
    plt.plot([smallest-.2, biggest+.2], [smallest-.2, biggest+.2], zorder=100,
        color='k', label='$\hat{y}$ = $y$')
    plt.xlim(smallest-.2, biggest+.2)
    plt.ylim(smallest-.2, biggest+.2)
    plt.xlabel('$y$')
    plt.ylabel('$\hat{y}$')
    plt.legend()
    if path:
        plt.savefig(f'{path}r2.png', bbox_inches='tight', pad_inches=0.01)
    if show:
        plt.show()


def plot_confidence(resids, uncertainties,
        frac=1.0, text=None, path='', show=False):
    """Plots confidence curve.

    Parameters
    ----------
    resids : Series
        Residuals
    uncertainties : Series
        Uncertainty measure values
    frac : float
        The fraction of covered outputs ([0.0, 1.0]), default: .5
    text : str
        Text to put in the plot, default: None
    path : str
        Path of the file to store the plot, default: '' (no storing)
    show : bool
        If True, plt.show() will be called, default: False
    """
    # Get data
    oracle_rmses, measure_rmses = rmses_frac(resids, uncertainties, frac=frac)
    x_space = np.linspace(0.0, 100.0*frac, len(oracle_rmses))
    plt.figure(figsize=(5, 5))
    plt.grid(zorder=1000)
    plt.plot(x_space, oracle_rmses, label='Best (oracle)', color='k')
    plt.plot(x_space, measure_rmses, label='Uncertainty', color=DEF_COLOR,
        zorder=100)
    # Put descriptions
    if text:
        plt.text(90.0*frac, .9*oracle_rmses[0], text, fontsize=20)
    plt.xlabel('Percentile')
    plt.ylabel('RMSE')
    plt.legend(loc='upper right')
    if path:
        plt.savefig(f'{path}confidence.png', bbox_inches='tight',
            pad_inches=0.01)
    if show:
        plt.show()


def plot_scatter(resids, uncertainties, path='', show=False):
    """Plots uncertainties vs. residuals scatter plot.

    Parameters
    ----------
    resids : Series
        Residuals
    uncertainties : Series
        Uncertainty measure values
    path : str
        Path of the file to store the plot, default: '' (no storing)
    show : bool
        If True, plt.show() will be called, default: False
    """
    plt.figure(figsize=(5, 5))
    plt.grid(zorder=1000)
    plt.plot(uncertainties, resids, 'o', zorder=101, markersize=4,
        color=DEF_COLOR, mfc='none', alpha=.5)
    plt.xlabel('Uncertainty')
    plt.ylabel('Absolute residuals')
    if path:
        plt.savefig(f'{path}scatter.png', bbox_inches='tight',
            pad_inches=0.0)
    if show:
        plt.show()


def plot_roc(y, te_probs, path='', show=False):
    """Plots ROC curve.

    Parameters
    ----------
    y : Series
        True values
    te_probs : Series
        Posterior probabilities of the test predictions
    path : str
        Path of the file to store the plot, default: '' (no storing)
    show : bool
        If True, plt.show() will be called, default: False
    """
    # Compute ROC curve
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
    if show:
        plt.show()


def plot_cumulative_accuracy(y, te_preds, te_probs,
        start_frac=.05, path='', show=False):
    """Plots cumulative accuracy plot.

    Parameters
    ----------
    y : Series
        True values
    te_preds : Series
        Test predictions
    te_probs : Series
        Posterior probabilities of the test predictions
    start_frac : float [0, 1]
        Fraction of predictions to start with, default: 0.05
    path : str
        Path of the file to store the plot, default: '' (no storing)
    show : bool
        If True, plt.show() will be called, default: False
    """
    # Get accuracies
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
    if show:
        plt.show()


"""Function library to automate uncertainty evaluation of machine learning
methods that provide uncertainties for predictions.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from copy import deepcopy

from datetime import datetime

from ensemble_uncertainties.automatize import make_folder
from ensemble_uncertainties.constants import (
    DEF_COLOR,
    DPI,
    N_SPLITS,
    RANDOM_SEED,
    V_THRESHOLD
)
from ensemble_uncertainties.evaluators.evaluator_support import (
    scale_and_filter,
    format_time_elapsed
)
from ensemble_uncertainties.utils.ad_assessment import spearman_coeff
from ensemble_uncertainties.utils.plotting import (
    plot_scatter,
    plot_confidence
)

from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from tqdm import tqdm


# Settings
mpl.rcParams['figure.dpi'] = DPI


def run_evaluation(X, y, model, n_splits=N_SPLITS, seed=RANDOM_SEED,
        scale=True, v_threshold=V_THRESHOLD, path=None, args=None):
    """Runs MC Dropout UQ evaluation for given settings.

    Parameters
    ----------
    X : DataFrame
        Matrix of dependent variables. Index and header must be provided, name
        of the index column must be 'id'.
    y : DataFrame
        Vector of output variables.Index and header must be provided, name
        of the index column must be 'id'.
    model : object
        A fittable model object that implements the methods fit(X, y) and
        uq_predict(X), which yields predictions and uncertainty values
    n_splits : int
        Number of splits in k-fold, default: constants.N_SPLITS
    seed : int
        Seed to use for splitting, default: constants.RANDOM_SEED
    scale : bool
        Whether standardize variables, default: True
    v_threshold : float
        The variance threshold to apply after normalization, variables
        with a variance below will be removed, default: V_THRESHOLD
    path : str
        Path to the directory to store the results in, default: None
    args : argparse.Namespace
        Parsed arguments from an argument parser.
        Useful for logging. Default: None.
    """
    np.random.seed(args.seed)
    start_time = datetime.now() 
    predictions, uncertainties = perform(X, y, model, n_splits=n_splits,
        seed=seed, scale=scale, v_threshold=v_threshold)
    time_elapsed = datetime.now() - start_time
    took = format_time_elapsed(time_elapsed)
    eval_results = evaluate_outcome(predictions, uncertainties, y)
    results, predictive_quality, uncertainty_quality = eval_results
    if not path.endswith('/'):
        path += '/'
    make_folder(path)
    results.to_csv(f'{path}results.csv', sep=';')
    plots_path = f'{path}plots/'
    make_folder(plots_path)
    plot_r2_test(y['y'], results['predicted'], path=plots_path)
    plot_confidence(results['resid'].abs(), results['uq'], path=plots_path)
    plot_scatter(results['resid'].abs(), uncertainties, path=plots_path)
    print()
    print(f'R^2: {predictive_quality:.3f}')
    print(f'rho: {uncertainty_quality:3f}')
    print(f'Took {took}')
    print()
    su_write_report(args, model, predictive_quality, uncertainty_quality,
        took)

    
def perform(X, y, model, n_splits=N_SPLITS, seed=RANDOM_SEED, scale=True,
        v_threshold=V_THRESHOLD):
    """Performs a complete evaluation (each split) and
    accumulates all predictions/uncertainties.

    Parameters
    ----------
    X : DataFrame
        Matrix of dependent variables. Index and header must be provided,
        name of the index column must be 'id'.
    y : DataFrame
        Vector of output variables.Index and header must be provided,
        name of the index column must be 'id'.
    model : object
        A fittable model object that implements the methods fit(X, y) and
        uq_predict(X), which yields predictions and uncertainty values
    n_splits : int
        Number of splits in k-fold, default: constants.N_SPLITS
    seed : int
        Seed to use for splitting, default: constants.RANDOM_SEED
    scale : bool
        Whether standardize variables, default: True
    v_threshold : float
        The variance threshold to apply after normalization, variables
        with a variance below will be removed, default: V_THRESHOLD

    Returns
    -------
    DataFrame, DataFrame
        Table of collected predictions, table of collected uncertainties
    """
    # Define splitting scheme
    kfold = KFold(n_splits=n_splits,
        random_state=seed, shuffle=True)
    splits = kfold.split(X)
    split_iter = tqdm(list(splits))
    # Prepare tables
    predictions = pd.DataFrame()
    uncertainties = pd.DataFrame()
    # Run all folds
    for train_id, test_id in split_iter:
        # Select train and test fraction
        X_tr, X_te = X.iloc[train_id], X.iloc[test_id]
        y_tr = y.iloc[train_id]
        # Fit and predict
        preds, uvals = run_split(X_tr, X_te, y_tr, model, scale, v_threshold)
        predictions = pd.concat([predictions, preds], axis=0)
        uncertainties = pd.concat([uncertainties, uvals], axis=0)
    # Get train and test predictive quality
    predictions = predictions.reindex(y.index)
    predictions.columns = ['pred']
    uncertainties = uncertainties.reindex(y.index)
    uncertainties.columns = ['uq']
    return predictions, uncertainties


def run_split(X_tr, X_te, y_tr, model, scale=True, v_threshold=V_THRESHOLD):
    """Performs the evaluation of a single split.

    Parameters
    ----------
    X_tr : DataFrame
        Train inputs
    X_te : DataFrame
        Test inputs
    y_tr : DataFrame
        Train outputs
    model : object
        A fittable model object that implements the methods fit(X, y) and
        uq_predict(X), which yields predictions and uncertainty values
    scale : bool
        Whether standardize variables, default: True
    v_threshold : float
        The variance threshold to apply after normalization, variables with
        a variance below will be removed, default: V_THRESHOLD

    Returns
    -------
    DataFrame, DataFrame
        Table of predictions, table of uncertainties
    """
    # Preprocess
    saf_result = scale_and_filter(X_tr, X_te,
        scale=scale, v_threshold=v_threshold)
    X_tr_sc_filt, X_te_sc_filt, _, _ = saf_result
    # Construct model (fit)
    model_fitted = deepcopy(model)
    model_fitted.fit(X_tr_sc_filt, y_tr.values.ravel())
    # Predict
    means, sdevs = model_fitted.uq_predict(X_te_sc_filt)
    preds = pd.DataFrame(means, index=X_te.index)
    uvals = pd.DataFrame(sdevs, index=X_te.index)
    return preds, uvals


def evaluate_outcome(predictions, uncertainties, y):
    """Assigns performances and summarizes predictions/uncertainties.

    Parameters
    ----------
    predictions : DataFrame
        Predicted outputs
    uncertainties : DataFrame
        UQ values
    y : DataFrame
        Observed outputs

    Returns
    -------
    DataFrame, float, float
        Table of result, predictive quality, uncertainty quality
    """
    results = pd.DataFrame(index=y.index)
    # Compute final predictions by average
    results['predicted'] = predictions['pred']
    # Compute sdev of ensemble predictions by output distribution
    results['uq'] = uncertainties['uq']
    results['resid'] = y['y'] - results['predicted']
    predictive_quality = r2_score(y['y'], results['predicted'])
    uncertainty_quality =  spearman_coeff(results['resid'], results['uq'])
    return results, predictive_quality, uncertainty_quality


def plot_r2_test(y, te_preds, path='', show=False):
    """Plots observed vs. predicted scatter plot
    for a set of test predictions.

    Parameters
    ----------
    y : Series
        True values
    te_preds : Series
        Test predictions
    path : str
        Path of the file to store the plot, default: '' (no storing)
    show : bool
        If True, plt.show() will be called, default: False
    """
    # Compute scores
    te_r2 = r2_score(y, te_preds)
    # Get corner values of the outputs/predictions
    smallest = min(min(y), min(te_preds.values))
    biggest = max(max(y), max(te_preds.values))
    # Plot
    plt.figure(figsize=(5, 5))
    plt.grid(zorder=1000)
    plt.plot(y, te_preds, 'o', zorder=100, markersize=4, label=None,
        color=DEF_COLOR, mfc='none', alpha=.7)
    plt.scatter([], [], label=f'Test,  $R^2$: {te_r2:.3f}',
        color=DEF_COLOR, facecolor='none')
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


def su_write_report(args, model, predictive_quality, uncertainty_quality,
        took):
    """Writes informative summary file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from an argument parser
    model : object
        Fittable model object used for evaluation
    predictive_quality : float
        R^2 regression performance
    uncertainty_performance : float
        Spearman's rho rank correlation coefficient
    took : str
        Formatted string describing the elapsed time
    """
    path = args.output_path
    with open(f'{path}report.txt', 'w') as f:
        f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write('Settings\n')
        f.write('--------\n')
        f.write(f'model name:   {args.model_name}\n')
        f.write(f'model object: {type(model)}\n')
        f.write(f'seed:         {args.seed}\n')
        f.write(f'n_splits:     {args.n_splits}\n')
        f.write('\n')
        f.write('Data info\n')
        f.write('---------\n')
        f.write(f'Input X file: {args.X_path}\n')
        f.write(f'Input y file: {args.y_path}\n')
        f.write(f'Output path:  {path}\n')
        f.write('\n')
        f.write('Results\n')
        f.write('-------\n')
        f.write(f'Test R^2:       {predictive_quality:.3f}\n')
        f.write(f"Spearman's rho: {uncertainty_quality:.3f}\n")
        f.write(f'Overall runtime: {took}\n')

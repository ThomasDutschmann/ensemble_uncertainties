
"""Library of functions to support the Evaluator class."""

import numpy as np
import pandas as pd

from ensemble_uncertainties.constants import (
    RANDOM_SEED,
    V_THRESHOLD
)

from numpy.random import default_rng

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm


def use_tqdm(iterable, use):
    """Returns tqdm(iterable) if use is True, else iterable."""
    if use:
        return tqdm(iterable)
    else:
        return iterable


def format_time_elapsed(time_elapsed):
    """Formats the elapsed time to a more human readable output.

    Parameters
    ----------
    time_elapsed : datetime.timedelta
        The elapsed time

    Returns
    -------
    str
        A string where hours, minutes and seconds are separated.
    """
    took = format(time_elapsed).split('.')[0]
    hours, minutes, seconds = took.split(':')
    display = ''
    if int(hours) > 0:
        display += f'{hours} Hour(s). '
    if int(minutes) > 0:
        display += f'{int(minutes)} Minute(s). '
    if int(seconds) > 0:
        display += f'{int(seconds)} Second(s). '
    if display:
        return display
    else:
        return '< 1 Second.'


def make_columns(repetitions, n_splits):
    """Provides column names for the tables of all predictions (from each
    split of each rep).

    Parameters
    ----------
    repetitions : int
        Repetitions of the n-fold validation
    n_splits : int
        number of splits in KFold

    Returns
    -------
    list
        List of the column names as strings
    """
    reps = range(repetitions)
    splits = range(n_splits)
    pre_cols = [[f'rep{i}_split{j}' for j in splits] for i in reps]
    cols = [item for sublist in pre_cols for item in sublist]
    return cols


def scale_and_filter(X_tr, X_te, scale=True, v_threshold=V_THRESHOLD):
    """Applies variable scaling and variance threshold filtering to train and
    test inputs.

    Parameters
    ----------
    X_tr : DataFrame
        Train inputs
    X_te : DataFrame
        Test inputs
    scale : bool
        Whether standardize variables, default: True
    v_threshold : float
        The variance threshold to apply after normalization, variables with a
        variance below will be removed, default: V_THRESHOLD

    Returns
    -------
    DataFrame, FataFrame, VarianceThreshold, StandardScaler
        Scaled and filtered train and test inputs,
        fitted threshold filter, fitted scaler 
    """
    # Drop variables below variance threshold (after normalization)
    X_tre_norm = pd.DataFrame(
        X_tr / X_tr.mean(),
        index=X_tr.index,
        columns=X_tr.columns
    ).fillna(1.0)
    vt = VarianceThreshold(threshold=v_threshold).fit(X_tre_norm)
    # Do not perform scaling if scale is False
    if scale:
        pre_scaler = StandardScaler()
    else:
        pre_scaler = StandardScaler(with_mean=False, with_std=False)
    # Define variance threshold filter after scaling
    scaler = pre_scaler.fit(X_tr)
    # Variance-filter and scale train inputs
    X_tr_sc = pd.DataFrame(scaler.transform(X_tr),
        index=X_tr.index, columns=X_tr.columns)
    X_tr_sc_filt = pd.DataFrame(X_tr_sc,
        index=X_tr.index, columns=X_tr.columns[vt.get_support()])
    # Variance-filter and scale test inputs
    X_te_sc = pd.DataFrame(scaler.transform(X_te),
        index=X_te.index, columns=X_te.columns)
    X_te_sc_filt = pd.DataFrame(X_te_sc,
        index=X_te.index, columns=X_te.columns[vt.get_support()])
    return X_tr_sc_filt, X_te_sc_filt, vt, scaler


def make_array(dim):
    """Makes an empty table (dim[0] x dim[1]) to store objects into."""
    return np.empty(dim, dtype=object)


def random_int32(size, seed=RANDOM_SEED):
    """Returns a <size>-long array of unique random 0+ 32-bit integers."""
    generator = default_rng(seed=seed)
    ints = generator.choice(2**32-1, size=size, replace=False)
    return ints


def print_summary(overall_run_time, metric_name, train_quality, test_quality):
    """Show train performance, test performance, and over all runtime.

    Parameters
    ----------
    overall_run_time : datetime.timedelta
        The elapsed time
    metric_name : str
        Name of the evaluation metric
    train_quality : float
        Train performance, measured in the provided metric 
    test_quality : float
        Test performance, measured in the provided metric   
    """
    took = format_time_elapsed(overall_run_time)
    print()
    print(f'Ensemble train {metric_name}:', end=' ')
    print(f'{train_quality:.3f}')
    print(f'Ensemble test {metric_name}:', end='  ')
    print(f'{test_quality:.3f}')
    print(f'Overall runtime:    {took}')
    print()

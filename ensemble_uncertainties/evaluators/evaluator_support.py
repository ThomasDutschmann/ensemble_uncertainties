
"""Library of functions to support the Evaluator class."""

import numpy as np

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
    return display


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


def make_array(dim):
    """Makes an empty table (dim[0] x dim[1]) to store objects into."""
    return np.empty(dim, dtype=object)


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

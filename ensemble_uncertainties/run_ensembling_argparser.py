
"""Specific parser for command line arguments."""

import argparse

from ensemble_uncertainties.constants import (
    N_REPS,
    N_SPLITS, 
    RANDOM_SEED,
    V_THRESHOLD
)


parser = argparse.ArgumentParser(
    prog='ensemble_uncertainties/ensemble_uncertainties/run_ensembling',
    description='Application to evaluate k-fold ensemble uncertainties.'
)

parser.add_argument(
    '-r',
    '--repetitions',
    dest='repetitions',
    type=int,
    help=f'number of repetitions, default: {N_REPS}',
    default=N_REPS
)

parser.add_argument(
    '-n',
    '--n_splits',
    dest='n_splits',
    type=int,
    help=f'number of splits, default: {N_SPLITS}',
    default=N_SPLITS
)

parser.add_argument(
    '-x',
    '--independent',
    dest='X_path',
    type=str,
    help='path to file of independent variables (X)',
    required=True,
    default=None
)

parser.add_argument(
    '-y',
    '--dependent',
    dest='y_path',
    type=str,
    help='path to file of dependent variables (y)',
    required=True,
    default=None
)

parser.add_argument(
    '-o',
    '--output',
    dest='output_path',
    type=str,
    help='path to the folder where the results are stored',
    required=True,
    default=None
)

parser.add_argument(
    '-m',
    '--model',
    dest='model_name',
    type=str,
    help='which model to use, e.g. "svm_rbf" or "dl"',
    required=True,
    default=None
)

parser.add_argument(
    '-t',
    '--task',
    dest='task',
    type=str,
    help='"classification" or "regression"',
    required=True,
    default=None
)

parser.add_argument(
    '-s',
    '--seed',
    dest='seed',
    type=int,
    help=f'random seed, default: {RANDOM_SEED}',
    default=RANDOM_SEED
)

parser.add_argument(
    '-v',
    '--verbose',
    dest='verbose',
    help='if set, much output will be produced',
    action='store_true',
)

parser.add_argument(
    '-d',
    '--deactivate_scaling',
    dest='deactivate_scaling',
    help='if set, feature scaling is deactivated (for binary/counts)',
    action='store_true',
)

parser.add_argument(
    '-e',
    '--variance_threshold',
    dest='v_threshold',
    type=float,
    help=f'variance threshold (after normalization), default: {V_THRESHOLD}',
    default=V_THRESHOLD
)

parser.add_argument(
    '-a',
    '--store_all',
    dest='store_all',
    help='if set, models and data transformers will also be stored',
    action='store_true',
)

parser.add_argument(
    '-b',
    '--bootstrapping',
    dest='bootstrapping',
    help='if set, bootstrapping will be used to generate the subsamples',
    action='store_true',
)

parser.add_argument(
    '-l',
    '--normalize',
    dest='normalize',
    help='if set, features will be normalized instead of standardized',
    action='store_true',
)

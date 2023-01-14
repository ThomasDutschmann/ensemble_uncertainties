
"""Specific parser for command line arguments."""

import argparse


parser = argparse.ArgumentParser(
    prog='ensemble_uncertainties/ensemble_uncertainties/predict',
    description='Application to predict test outputs by an ensemble.'
)

parser.add_argument(
    '-x',
    '--independent',
    dest='X_path',
    type=str,
    help='path to file of independent variables (X) of the test set',
    required=True,
    default=None
)

parser.add_argument(
    '-y',
    '--dependent',
    dest='y_path',
    type=str,
    help='path to file of dependent variables (y) of the test set',
    required=True,
    default=None
)

parser.add_argument(
    '-i',
    '--input',
    dest='input_path',
    type=str,
    help='path to the folder where the ensemble files (models, etc) are',
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
    '--model_format',
    dest='model_format',
    type=str,
    help='"network" (for TF-models), "xgb:c", "xgb:r", default: "pickle"',
    required=True,
    default=None
) 

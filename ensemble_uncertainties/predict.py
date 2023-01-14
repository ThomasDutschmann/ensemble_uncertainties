
"""
Predicts the outputs for a test set using an ensemble.
Currently only supports regression!
"""

import os
import pickle

import numpy as np
import pandas as pd

from datetime import datetime

from ensemble_uncertainties.predict_argparser import parser
from ensemble_uncertainties.automatize import (
    extend_path,
    load_data,
    make_folder
)
from ensemble_uncertainties.error_handling import writing_accessibility
from ensemble_uncertainties.evaluators.evaluator_support import (
    format_time_elapsed
)
from ensemble_uncertainties.utils.ad_assessment import compute_uq_qualities
from ensemble_uncertainties.utils.plotting import (
    plot_confidence,
    plot_r2_test,
    plot_scatter
)

from keras.models import load_model as keras_load_model

from tqdm import tqdm

from xgboost import XGBClassifier, XGBRegressor


def main():
    """usage: ensemble_uncertainties/ensemble_uncertainties/predict [-h]
        -x X_PATH -y Y_PATH -i INPUT_PATH -o OUTPUT_PATH -m MODEL_FORMAT

    Application to predict the outputs for a test set using an ensemble.

    optional arguments:
    -h, --help            show this help message and exit
    -x X_PATH, --independent X_PATH
                            path to file of independent
                            variables(X) of the test set
    -y Y_PATH, --dependent Y_PATH
                            path to file of dependent
                            variables (y) of the test set
    -i INPUT_PATH, --input INPUT_PATH
                            path to the folder where the
                            ensemble files (models, etc) are
    -o OUTPUT_PATH, --output OUTPUT_PATH
                            path to the folder where the results are stored
    -m MODEL_FORMAT, --model_format MODEL_FORMAT
                            "network" (for H5)/"xgboost:c"/"xgboost:r",
                            default: "pickle"
    """
    args = parser.parse_args()
    writing_accessibility(args.output_path)
    in_folder_path = extend_path(args.input_path)
    X_te_path = args.X_path
    y_te_path = args.y_path
    X_te, y_te = load_data(X_te_path, y_te_path)
    out_path = extend_path(args.output_path)
    model_format = args.model_format
    start_time = datetime.now()
    vts, scs, members = load_pipeline(in_folder_path, model_format)
    names = get_model_names(f'{in_folder_path}models/')
    ens_predictions = compute_predictions(vts, scs, members, X_te, names)
    p_qual, u_qual, results = evaluate_predictions(ens_predictions, y_te)
    time_elapsed = datetime.now() - start_time
    took = format_time_elapsed(time_elapsed)
    save_predictions(ens_predictions, results, out_path)
    make_plots(results, y_te, out_path)
    predict_write_report(args, p_qual, u_qual, took)


def compute_predictions(vts, scs, members, X_te, names):
    """Predicts outputs of test data with the ensemble members.

    Parameters
    ----------
    vts : list
        List of VarianceThresholds
    scs : list
        List of StandardScalers
    members : list
        List of member models
    X_te : DataFrame
        Test data, independent variables
    names : list
        List of model names (file names of model files)

    Returns
    -------
    DataFrame
        An n_objects x n_members matrix with all member predictions
    """  
    predictions = list()
    for vt, sc, member in tqdm(zip(vts, scs, members)):
        X_te_sc = pd.DataFrame(sc.transform(X_te),
            index=X_te.index, columns=X_te.columns)
        X_te_sc_filt = pd.DataFrame(X_te_sc,
            index=X_te.index, columns=X_te.columns[vt.get_support()])
        current_preds = member.predict(X_te_sc_filt)
        predictions.append(current_preds)
    predictions_df = pd.DataFrame(
        np.array(predictions).T, columns=names, index=X_te.index
    )
    return predictions_df


def get_model_names(path):
    """Gets file names of the ensemble member models.

    Parameters
    ----------
    path : str
        Path to the folder of members

    Returns
    -------
    list
        The list of names
    """  
    names = list()
    for name in sorted(os.listdir(path)):
        if 'model' in name:
            names.append(name.split('.')[0])
    return names


def evaluate_predictions(ens_predictions, y_te):
    """Evaluates quality of the test predictions.

    Parameters
    ----------
    ens_predictions : DataFrame
        An n_objects x n_members matrix with all member predictions
    y_te : DataFrame
        The observed outputs

    Returns
    -------
    float, float, DataFrame
        Predictive quality UQ quality, and results
        table with predictions, residuals and UQ values
    """
    means = ens_predictions.mean(axis=1)
    stds = ens_predictions.std(axis=1)
    results, p_quality, u_quality = compute_uq_qualities(
        means.values, stds.values, y_te['y'].values
    )
    return p_quality, u_quality, results


def save_predictions(predictions, results, out_path):
    """Stores predictions as CSV-files.

    Parameters
    ----------
    predictions : DataFrame
        An n_objects x n_members matrix with all member predictions
    results : DataFrame
        Results table with predictions, residuals and UQ values
    out_path : str
        Path to the folder to store the predictions in
    """
    make_folder(f'{out_path}predictions/')
    predictions.to_csv(f'{out_path}predictions/single_predictions.csv')
    results.to_csv(f'{out_path}predictions/ensemble_predictions.csv')


def make_plots(results, y_te, out_path):
    """Creates and stores plots outlining the quality as PNG-files.

    Parameters
    ----------
    results : DataFrame
        Results table with predictions, residuals and UQ values
    y_te : DataFrame
        The observed outputs
    out_path : str
        Path to the folder to store the plots in
    """
    make_folder(f'{out_path}plots/')
    plot_r2_test(y_te['y'], results['predicted'], path=f'{out_path}plots/')
    plot_confidence(results['resid'], results['uq'], path=f'{out_path}plots/')
    plot_scatter(results['resid'], results['uq'], path=f'{out_path}plots/')


def load_pipeline(path, model_format):
    """Loads VarianceThresholds, StandardScalers, and member models.

    Parameters
    ----------
    path : str
        Path to the mother directory of files created by run_ensembling
    model_format : str
        The format of the member models ("network" for H5, "xgb:r",
        for regression XGBoost, "xgb:c" for classification XGBoost,
        default: "pickle")

    Returns
    -------
    list, list, list
        List of VarianceThresholds, list of StandardScalers,
        and list of member models
    """
    vts_path = f'{path}variance_thresholds/'
    scs_path = f'{path}standard_scalers/'
    models_path = f'{path}models/'
    vts = load_objects(vts_path, 'pickle')
    scs = load_objects(scs_path, 'pickle')
    models = load_objects(models_path, model_format)
    return vts, scs, models


def load_object(full_path, mode):
    """Loads objects that are stored as files on disk.

    Parameters
    ----------
    full_path : str
        Path to the file
    mode : str
        The format of the object to load ("network" for H5, "xgb:r",
        for regression XGBoost, "xgb:c" for classification XGBoost,
        default: "pickle")

    Returns
    -------
    object
        The loaded object
    """
    if mode == 'pickle':
        file = open(full_path, 'rb')
        o = pickle.load(file)
    elif mode == 'network':
         o = keras_load_model(full_path)
    elif mode.startswith('xgb'):
        submode = mode.split(':')[1]
        if submode == 'c':
            o = XGBClassifier()
        elif submode == 'r':
            o = XGBRegressor()
        o.load_model(full_path)
    return o


def load_objects(path, mode):
    """Loads all objects that are stored as files on disk.

    Parameters
    ----------
    path : str
        Path to the folder of files
    mode : str
        The format of the objects to load ("network" for H5, "xgb:r",
        for regression XGBoost, "xgb:c" for classification XGBoost,
        default: "pickle")

    Returns
    -------
    list
        List of the loaded objects
    """
    objects = list()
    for file_path in sorted(os.listdir(path)):
        full_path = f'{path}{file_path}'
        o = load_object(full_path, mode)
        objects.append(o)
    return objects


def predict_write_report(args, predictive_quality, uncertainty_quality,
        took):
    """Writes informative summary file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from an argument parser
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
        f.write(f'Input folder:   {args.input_path}\n')
        f.write(f'model format:   {args.model_format}\n')
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


if __name__ == '__main__':
    main()


"""Library of functions to conveniently automatize
running an Evaluator and collecting its results.
"""

import os
import pickle
import shutil
import warnings

import pandas as pd

from datetime import datetime

from ensemble_uncertainties.constants import (
    N_REPS,
    N_SPLITS,
    RANDOM_SEED,
    V_THRESHOLD
)
from ensemble_uncertainties.error_handling import (
    file_availability,
    file_compatibility,
    y_file_compatibility
)
from ensemble_uncertainties.evaluators.classification_evaluator import (
    ClassificationEvaluator
)
from ensemble_uncertainties.evaluators.evaluator_support import (
    format_time_elapsed
)
from ensemble_uncertainties.evaluators.regression_evaluator import (
    RegressionEvaluator
)
from ensemble_uncertainties.utils.ad_assessment import (
    auco,
    rmses_frac,
    spearman_coeff
)
from ensemble_uncertainties.utils.plotting import (
    plot_r2,
    plot_confidence,
    plot_scatter,
    plot_roc,
    plot_cumulative_accuracy
)

from sklearn.preprocessing import Normalizer


def load_data(X_path, y_path):
    """Checks if file paths and file content are ok. If so, load X and y.

    Parameters
    ----------
    X_path : str
        Path to the CSV-file of the independent variables
    y_path : str
        Path to the CSV-file of the dependent variables

    Returns
    -------
    DataFrame, DataFrame
        X and y
    """
    file_availability(X_path)
    file_compatibility(X_path)
    file_availability(y_path)
    file_compatibility(y_path)
    y_file_compatibility(y_path)
    X = pd.read_csv(X_path, sep=';').set_index('id')
    y = pd.read_csv(y_path, sep=';').set_index('id')
    return X, y


def run_evaluation(model, task, X, y, verbose=True, repetitions=N_REPS,
        n_splits=N_SPLITS, seed=RANDOM_SEED, scale=True, 
        v_threshold=V_THRESHOLD, bootstrapping=False, normalize=False,
        path=None, store_all=False, args=None, follow_up=None):
    """Runs evaluation with an EnsembleADEvaluator for given settings.

    Parameters
    ----------
    model : object
        A fittable model object, e.g. sklearn.svm.SVC()
        that implements the methods fit(X, y) and predict(X)
    task : str
        'regression' for regression tasks, 'classification' otherwise
    X : DataFrame
        Matrix of dependent variables. Index and header must be provided, name
        of the index column must be 'id'.
    y : DataFrame
        Vector of output variables.Index and header must be provided, name
        of the index column must be 'id'.
    verbose : bool
        Whether progress is constantly reported, default: True
    repetitions : int
        Repetitions of the n-fold validation, default: constants.REPS
    n_splits : int
        Number of splits in k-fold, default: constants.N_SPLITS
    seed : int
        Seed to use for splitting, default: constants.RANDOM_SEED
    scale : bool
        Whether to standardize features, default: True.
        Automatically set to True when using normalization.
    v_threshold : float
        The variance threshold to apply after normalization, variables
        with a variance below will be removed, default: V_THRESHOLD
    bootstrapping : bool
        If True, bootstrapping will be used to generate the subsamples,
        default: False. n_splits will be automatically set to 1.
    normalize : bool
        Whether to normalize the features instead of to standardize them,
        default: False. 
    path : str
        Path to the directory to store the results in, default: None
    store_all : bool
        If True, models and data transformers are stored too, default: False
    args : argparse.Namespace
        Parsed arguments from an argument parser.
        Useful for logging. Default: None.
    follow_up : function
        A custom function that takes the fitted evaluator at the end of the
        evaluation run as input: follow_up(evaluator).
        Useful for testing. Default: None
    """
    if verbose:
        print()
    # Initialize evaluator
    if task == 'classification':
        evaluator_type = ClassificationEvaluator
    elif task == 'regression':
        evaluator_type = RegressionEvaluator
    evaluator = evaluator_type(
        model=model,
        verbose=verbose,
        repetitions=repetitions,
        n_splits=n_splits,
        seed=seed,
        scale=scale,
        v_threshold=v_threshold,
        bootstrapping=bootstrapping
    )
    if normalize:
        evaluator.set_scaler_class(Normalizer)
        evaluator.scale = True
    # Check for equivalent indexes
    if not (X.index == y.index).all():
        warnings.warn('Warning! X and y have different indexes/order.')
    # Run evaluation
    evaluator.perform(X, y)
    # Store input space transformers, models, and single repetition
    # predictions, if desired (i.e., if results path is given)
    if path:
        path = extend_path(path)
        plots_to_file(evaluator, task, path)
        results_tables_to_file(evaluator, path)
        write_report(args, evaluator)
        if store_all:
            transformers_to_file(evaluator, path)
            models_to_file(evaluator, path)
    if follow_up:
        follow_up(evaluator)


def extend_path(path):
    """Extends paths by '/' if necessary.

    Parameters
    ----------
    path : str
        Path to extend

    Returns
    -------
    str
        Extended path
    """
    if path.endswith('/'):
        return path 
    else:
        return path + '/'    


def make_folder(path):
    """Creates a new folder, overwrites already existing one.

    Parameters
    ----------
    path : str
        Path of the folder to create
    """
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


def plots_to_file(evaluator, task, path):
    """Creates comprehensive plots for a single evaluation.

    Parameters
    ----------
    evaluator : Evaluator
        Applied evaluator
    task : str
        'regression' for regression tasks, 'classification' otherwise    
    path : str
        Path to the directory to store the plots in
    """
    plots_path = f'{path}plots/'
    make_folder(plots_path)
    y = evaluator.y['y']
    if task == 'classification':
        # Extract
        te_preds = evaluator.test_ensemble_preds['predicted']
        te_probs = evaluator.test_ensemble_preds['probA']
        # Plot
        plot_roc(y, te_probs, path=plots_path)
        plot_cumulative_accuracy(y, te_preds, te_probs, path=plots_path)
    elif task == 'regression':
        # Extract
        tr_preds = evaluator.train_ensemble_preds['predicted']
        te_preds = evaluator.test_ensemble_preds['predicted']
        resids = evaluator.test_ensemble_preds['resid'].abs()
        uncertainties = evaluator.test_ensemble_preds['sdep']
        # Plot
        plot_r2(y, tr_preds, te_preds, path=plots_path)
        plot_confidence(resids, uncertainties, path=plots_path)
        plot_scatter(resids, uncertainties, path=plots_path)


def transformers_to_file(evaluator, path):
    """Stores X_train transformers (scalers and variable threshold filters)
    as pickle files.

    Parameters
    ----------
    evaluator : Evaluator
        Applied evaluator 
    path : str
        Path to the directory to store the objects in
    """
    # Define paths
    vpath = f'{path}variance_thresholds/'
    spath = f'{path}standard_scalers/'
    # Make directories
    make_folder(vpath)
    make_folder(spath)
    # Store
    for rep_i in range(evaluator.repetitions):
        for fold_i in range(evaluator.n_splits):
            vt = evaluator.vt_filters[rep_i][fold_i]
            pickle.dump(vt, open(f'{vpath}vt_{rep_i}_{fold_i}.p', 'wb'))
            sc = evaluator.scalers[rep_i][fold_i]
            pickle.dump(sc, open(f'{spath}sc_{rep_i}_{fold_i}.p', 'wb'))


def models_to_file(evaluator, path):
    """Stores fitted estimators default as pickle files, or
    H5-files for TensorFlow wrappers, or json-files for XGBoost models.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from an argument parser.
    evaluator : Evaluator
        Applied evaluator 
    path : str
        Path to the directory to store the objects in
    """
    # Define path
    models_path = f'{path}models/'
    # Make directorie
    make_folder(models_path)
    # Store
    for rep_i in range(evaluator.repetitions):
        for fold_i in range(evaluator.n_splits):
            mpath = f'{models_path}model_{rep_i}_{fold_i}'
            model = evaluator.models[rep_i][fold_i]
            # Store deep estimator
            if 'save' in dir(model):
                model.save(f'{mpath}.h5')
            # Store XGB model
            elif 'save_model' in dir(model):
                model.save_model(f'{mpath}.txt')
            # Store any other model
            else:
                pickle.dump(model, open(f'{mpath}.p', 'wb'))


def results_tables_to_file(evaluator, path):
    """Stores single model predictions and averaged ensemble predictions
    for train and test evaluations as CSV files.

    Parameters
    ----------
    evaluator : Evaluator
        Applied evaluator 
    path : str
        Path to the directory to store the CSV file in
    """
    # Define paths
    ppath = f'{path}single_predictions/'
    epath = f'{path}ensemble_predictions/'
    # Make directories
    make_folder(ppath)
    make_folder(epath)
    # Store
    evaluator.train_preds.to_csv(f'{ppath}train.csv', sep=';')
    evaluator.test_preds.to_csv(f'{ppath}test.csv', sep=';')
    evaluator.train_ensemble_preds.to_csv(f'{epath}train.csv', sep=';')
    evaluator.test_ensemble_preds.to_csv(f'{epath}test.csv', sep=';')


def compute_auco_and_rho(evaluator):
    """Computes raw AUCO and Spearman's coefficient for a given evaluator.
    
    Parameters
    ----------
    evaluator : Evaluator
        Performed Evaluator-object
        
    Returns
    -------
    float, float
        AUCO, Spearman's rank
    """
    resids = evaluator.test_ensemble_preds['resid']
    uncertainties = evaluator.test_ensemble_preds['sdep']
    oracle_rmses, measure_rmses = rmses_frac(resids, uncertainties)
    area = auco(oracle_rmses, measure_rmses)
    rho = spearman_coeff(resids, uncertainties)
    return area, rho


def write_report(args, evaluator):
    """Writes informative summary file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from an argument parser.
    evaluator : Evaluator
        Applied evaluator 
    """
    metric_name = evaluator.metric_name
    train_quality = evaluator.train_ensemble_quality
    test_quality = evaluator.test_ensemble_quality
    runtime = evaluator.overall_run_time
    formatted_runtime = format_time_elapsed(runtime)
    path = args.output_path
    with open(f'{path}report.txt', 'w') as f:
        f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write('Settings\n')
        f.write('--------\n')
        f.write(f'task:         {args.task}\n')
        f.write(f'model name:   {args.model_name}\n')
        f.write(f'model object: {type(evaluator.model)}\n')
        f.write(f'seed:         {args.seed}\n')
        f.write(f'repetitions:  {args.repetitions}\n')
        f.write(f'n_splits:     {evaluator.n_splits}\n')
        f.write(f'scaling:      {not args.deactivate_scaling}\n')
        f.write(f'v.threshold:  {args.v_threshold}\n')
        f.write(f'bootstrapped: {args.bootstrapping}\n')
        f.write(f'scale object: {evaluator.scaler_class}\n')
        f.write(f'verbose:      {args.verbose}\n')
        f.write('\n')
        f.write('Data info\n')
        f.write('---------\n')
        f.write(f'Input X file: {args.X_path}\n')
        f.write(f'Input y file: {args.y_path}\n')
        f.write(f'Input shape:  {evaluator.X.shape}\n')
        f.write(f'Output path:  {path}\n')
        f.write('\n')
        f.write('Results\n')
        f.write('-------\n')
        f.write(f'Train {metric_name}:       {train_quality:.3f}\n')
        f.write(f'Test {metric_name}:        {test_quality:.3f}\n')
        if args.task == 'regression':
            area, rho = compute_auco_and_rho(evaluator)
            f.write(f'AUCO:           {area:.3f}\n')
            f.write(f"Spearman's rho:  {rho:.3f}\n")
        f.write(f'Overall runtime: {formatted_runtime}\n')

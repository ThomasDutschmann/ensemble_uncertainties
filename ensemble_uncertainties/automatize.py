
"""Library of functions to conveniently automatize
running an Evaluator and collecting its results.
"""

import os
import pickle
import shutil

from ensemble_uncertainties.constants import N_SPLITS, RANDOM_SEED, REPS

from datetime import datetime

from ensemble_uncertainties.evaluators.evaluator_support import (
    format_time_elapsed
)
from ensemble_uncertainties.evaluators.classification_evaluator import (
    ClassificationEvaluator
)
from ensemble_uncertainties.evaluators.regression_evaluator import (
    RegressionEvaluator
)

from ensemble_uncertainties.utils.plotting import (
    plot_r2,
    plot_confidence,
    plot_scatter,
    plot_roc,
    plot_cumulative_accuracy
)


def run_evaluation(model, task, X, y, verbose=True, repetitions=REPS,
        n_splits=N_SPLITS, seed=RANDOM_SEED, path=None, args=None,
        follow_up=None):
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
    path : str
        Path to the directory to store the results in, default: None
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
        evaluator = ClassificationEvaluator(model=model, verbose=verbose,
            repetitions=repetitions, n_splits=n_splits, seed=seed)
    elif task == 'regression':
        evaluator = RegressionEvaluator(model=model, verbose=verbose,
            repetitions=repetitions, n_splits=n_splits, seed=seed)
    # Run evaluation
    evaluator.perform(X, y)
    # Store input space transformers, models, and single repetition
    # predictions, if desired (i.e., if results path is given)
    if path:
        if not path.endswith('/'):
            path += '/'
        plots_to_file(evaluator, task, path)
        transformers_to_file(evaluator, path)
        models_to_file(evaluator, path)
        results_tables_to_file(evaluator, path)
        write_report(args, evaluator)
    if follow_up:
        follow_up(evaluator)


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
    if task == 'classification':
        plot_roc(evaluator, path=plots_path)
        plot_cumulative_accuracy(evaluator, path=plots_path)
    elif task == 'regression':
        plot_r2(evaluator, path=plots_path)
        plot_confidence(evaluator, path=plots_path)
        plot_scatter(evaluator, path=plots_path)


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
    """Stores fitted estimators as pickle files,
    or H5-files for TensorFlow wrappers.

    Parameters
    ----------
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
            if 'model_tools.deep_models.' in str(type(model)):
                model.save(f'{mpath}.h5')
            elif 'keras.engine.sequential.Sequential' in str(type(model)):
                model.save(f'{mpath}.h5')
            # Store scikit-learn model
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
        f.write(f'n_splits:     {args.n_splits}\n')
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
        f.write(f'Overall runtime: {formatted_runtime}\n')

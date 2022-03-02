
"""Main executable script.

Example execution:

python executable.py -x X_rdkit.csv -y y.csv -r 10 -t regression -m SVM -v
                     -o results/
"""

import pandas as pd

from ensemble_uncertainties.argparser import parser

from ensemble_uncertainties.automatize import run_evaluation

from ensemble_uncertainties.error_handling import (
    file_availability,
    file_compatibility,
    model_availability,
    writing_accessibility,
    y_file_compatibility
)

from ensemble_uncertainties.model_library import models


def main():
    """Entry point."""
    args = parser.parse_args()
    writing_accessibility(args.output_path)
    model = parse_model(args.task, args.model_name)
    X, y = load_data(args.X_path, args.y_path)
    scale = not args.deactivate_scaling
    run_evaluation(
        model=model,
        task=args.task,
        X=X,
        y=y,
        verbose=args.verbose,
        repetitions=args.repetitions,
        n_splits=args.n_splits, 
        seed=args.seed,
        scale=scale,
        path=args.output_path,
        store_all=args.store_all,
        args=args
    )


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


def parse_model(task, model_name):
    """Gets corresponding model from the model library.

    Parameters
    ----------
    task : str
        'regression' for regression tasks, 'classification' otherwise
    model_name : str
        Name of the model to check for

    Returns
    -------
    function
        A model object that has fit(X, y) and predict(X)
    """    
    model_availability(task, model_name)
    model = models[task][model_name]
    return model


if __name__ == '__main__':
    main()

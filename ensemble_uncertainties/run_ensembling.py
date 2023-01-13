
"""Main executable script.

Example execution:

python executable.py -x X.csv -y y.csv -r 10 -t regression -m SVM -v
                     -o results/
"""

from ensemble_uncertainties.run_ensembling_argparser import parser

from ensemble_uncertainties.automatize import load_data, run_evaluation

from ensemble_uncertainties.error_handling import (
    model_availability,
    writing_accessibility
)

from ensemble_uncertainties.model_library import models


def main():
    """usage: ensemble_uncertainties/ensemble_uncertainties/executable [-h]
        [-r REPETITIONS] [-n N_SPLITS] -x X_PATH -y Y_PATH -o OUTPUT_PATH
        -m MODEL_NAME -t TASK [-s SEED] [-v] [-d] [-e V_THRESHOLD] [-a] [-b]
        [-l]

    Application to evaluate repetitive k-fold ensemble ADs.

    optional arguments:
    -h, --help            show this help message and exit
    -r REPETITIONS, --repetitions REPETITIONS
                            number of repetitions, default: 100
    -n N_SPLITS, --n_splits N_SPLITS
                            number of splits, default: 2
    -x X_PATH, --independent X_PATH
                            path to file of dependent variables (X)
    -y Y_PATH, --dependent Y_PATH
                            path to file of independent variables (y)
    -o OUTPUT_PATH, --output OUTPUT_PATH
                            path to the folder where the results are stored
    -m MODEL_NAME, --model MODEL_NAME
                            which model to use, e.g. "svm_rbf" or "dl"
    -t TASK, --task TASK  "classification" or "regression"
    -s SEED, --seed SEED  random seed, default: 0
    -v, --verbose         if set, much output will be produced
    -d, --deactivate_scaling
                            if set, feature scaling is deactivated
                            (for binary/counts)
    -e V_THRESHOLD, --variance_threshold V_THRESHOLD
                            variance threshold (after normalization),
                            default: 0.005
    -a, --store_all       if set, models and data
                          transformers will also be stored
    -b, --bootstrapping   if set, bootstrapping will be
                          used to generate the subsamples
    -l, --normalize       if set, features will be normalized
                          instead of standardized
    """
    args = parser.parse_args()
    writing_accessibility(args.output_path)
    model = parse_model(args.task, args.model_name)
    X, y = load_data(args.X_path, args.y_path)
    scale = not args.deactivate_scaling
    v_threshold = args.v_threshold
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
        v_threshold=v_threshold,
        bootstrapping=args.bootstrapping,
        normalize=args.normalize,
        path=args.output_path,
        store_all=args.store_all,
        args=args
    )


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

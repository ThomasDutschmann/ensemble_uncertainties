
"""Runnable script for specific (non-kfold-based) uncertainty evaluation.
"""

from su_model_library import models
from su_automatize import su_run_evaluation
from su_argparser import parser

from ensemble_uncertainties.error_handling import writing_accessibility
from ensemble_uncertainties.executable import load_data


def main():
    """usage: ensemble_uncertainties/specific_uncertainties/executable [-h]
                [-n N_SPLITS] -x X_PATH -y Y_PATH -o OUTPUT_PATH -m MODEL_NAME
                [-s SEED] [-d] [-e V_THRESHOLD]

    Application to evaluate prediction uncertainties.

    optional arguments:
    -h, --help            show this help message and exit
    -n N_SPLITS, --n_splits N_SPLITS
                            number of splits, default: 2
    -x X_PATH, --independent X_PATH
                            path to file of dependent variables (X)
    -y Y_PATH, --dependent Y_PATH
                            path to file of independent variables (y)
    -o OUTPUT_PATH, --output OUTPUT_PATH
                            path to the folder where the results are stored
    -m MODEL_NAME, --model MODEL_NAME
                            which model (with uq estimation!) to use
    -s SEED, --seed SEED  random seed, default: 0
    -d, --deactivate_scaling
                            if set, variable scaling is deactivated 
                            (for binary inputs)
    -e V_THRESHOLD, --variance_threshold V_THRESHOLD
                            variance threshold (after normalization),
                            default: 0.005
    """
    args = parser.parse_args()
    writing_accessibility(args.output_path)
    X, y = load_data(args.X_path, args.y_path)
    model = models[args.model_name]
    scale = not args.deactivate_scaling
    v_threshold = args.v_threshold
    su_run_evaluation(
        X=X,
        y=y,
        model=model,
        n_splits=args.n_splits, 
        seed=args.seed,
        scale=scale,
        v_threshold=v_threshold,
        path=args.output_path,
        args=args
    )


if __name__ == '__main__':
    main()

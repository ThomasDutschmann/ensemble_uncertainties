
"""Runnable script for specific (non-kfold-based) uncertainty evaluation.
"""

from su_model_library import models
from su_automatize import run_evaluation
from su_argparser import parser

from ensemble_uncertainties.error_handling import writing_accessibility
from ensemble_uncertainties.executable import load_data


def main():
    """Entry point."""
    args = parser.parse_args()
    writing_accessibility(args.output_path)
    X, y = load_data(args.X_path, args.y_path)
    model = models[args.model_name]
    scale = not args.deactivate_scaling
    v_threshold = args.v_threshold
    run_evaluation(
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


"""Tests concerning the exectuable."""

import os
import unittest

from ensemble_uncertainties.automatize import run_evaluation

from ensemble_uncertainties.executable import load_data, parse_model


class RunTests(unittest.TestCase):

    def error_free_evaluation(self, task, model_name):
        """Check if evaluation runs without errors."""
        # Shut up, TensorFlow!
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        # Get inputs for run_evaluation function
        model = parse_model(task, model_name)
        if task == 'regression':
            # Load Tetrahymena, RDKit descriptors
            data_folder = '../test_data/tetrahymena/'
            X_path = f'{data_folder}tetrah_X.csv'
            y_path = f'{data_folder}tetrah_y.csv'
            X, y = load_data(X_path, y_path)
        elif task == 'classification':
            # Load CYP1A2, MOE descriptors
            data_folder = '../test_data/CYP1A2_klingspohn/'
            X_path = f'{data_folder}CYP1A2_X.csv'
            y_path = f'{data_folder}CYP1A2_y.csv'            
            X, y = load_data(X_path, y_path)
            X.rename(index={X.index.name: 'id'})
            y.rename(index={X.index.name: 'id'})
        print(f'Testing {task} with {model_name}.')
        # Define follow-up function to inspect fitted evaluator.
        # The test data sets are pretty easy to model, so train
        # and test accuracy should be pretty high for all combinations
        def report_and_assert(ev):
            tr_quality = ev.train_ensemble_quality
            te_quality = ev.test_ensemble_quality
            self.assertTrue(ev.train_ensemble_quality > .8)
            self.assertTrue(ev.test_ensemble_quality > .8)
            print(f'Train: {tr_quality:.3f}  Test: {te_quality:.3f}')
            print()
        # Finally, run the evaluation
        run_evaluation(
            model=model,
            task=task,
            X=X,
            y=y,
            verbose=False,
            repetitions=5,
            n_splits=5, 
            seed=0,
            path=None,
            follow_up=report_and_assert
        )

    def test_all_evaluation_types(self):
        """Check if all classification and regression model
        evaluations run without errors. The whole test takes
        about 12 minutes on my RTX 3080 Ti.
        """
        for task in ['classification', 'regression']:
            for model_name in ['DL', 'SVM', 'XGB', 'RF']:
                self.error_free_evaluation(task, model_name)


if __name__ == '__main__':
    unittest.main()

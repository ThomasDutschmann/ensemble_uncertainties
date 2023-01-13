
"""Tests concerning complete evaluation runs.

Must be run from 
    ensemble_uncertainties/ensemble_uncertainties/
in order to find the test input files.

All in all, the tests provided here roughly
take 3 minutes on my RTX 3080 Ti.
"""

import os
import unittest

from ensemble_uncertainties.automatize import run_evaluation
from ensemble_uncertainties.evaluators.regression_evaluator import (
    RegressionEvaluator
)
from ensemble_uncertainties.run_ensembling import load_data, parse_model


class ExecutableTests(unittest.TestCase):

    # Define custom follow-up function
    def report_and_assert_overall_quality(self, evaluator):
        """For the rather easy testing data sets at hand,
        train and test accuracy should be at least 60%.
        """
        tr_quality = evaluator.train_ensemble_quality
        te_quality = evaluator.test_ensemble_quality
        print(f'Train: {tr_quality:.3f}  Test: {te_quality:.3f}')
        print()
        self.assertTrue(evaluator.train_ensemble_quality > .6)
        self.assertTrue(evaluator.test_ensemble_quality > .6)

    def error_free_evaluation(self, task, model_name, X, y):
        """Check if evaluation runs without errors.
        
        Parameters
        ----------
        task : str
            'classification' or 'regression'
        model_name : str
            Name of the machine learning method
        X : DataFrame
            Independent variables
        y : DataFrame
            Dependent variables
        """
        # Shut up, TensorFlow!
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        model = parse_model(task, model_name)
        print(f'Testing {model_name} {task}.')
        # Run the evaluation
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
            follow_up=self.report_and_assert_overall_quality
        )

    def test_most_relevant_classification_types(self):
        """Checks classification model evaluations."""
        cyp1a2_folder = '../test_data/CYP1A2_klingspohn/'
        cyp1a2_moe_path = f'{cyp1a2_folder}CYP1A2_MOE.csv'
        cyp1a2_y_path = f'{cyp1a2_folder}CYP1A2_y.csv'
        X, y = load_data(cyp1a2_moe_path, cyp1a2_y_path)
        for model_name in ['shallow', 'deep', 'svm_rbf', 'xgb', 'rf']:
            self.error_free_evaluation('classification', model_name, X, y)

    def test_most_relevant_regression_types(self):
        """Checks regression model evaluations."""
        tetrah_folder = '../test_data/tetrahymena/'
        tetrah_rdkit_path = f'{tetrah_folder}tetrah_rdkit.csv'
        tetrah_y_path = f'{tetrah_folder}tetrah_y.csv'
        X, y = load_data(tetrah_rdkit_path, tetrah_y_path)
        for model_name in ['shallow', 'dropout', 'svm_rbf', 'xgb', 'rf']:
            self.error_free_evaluation('regression', model_name, X, y)

    def test_tanimoto_performance(self):
        """Asserts superiority of Tanimoto-SVR over RBF-SVR on ECFP."""
        print('\n\nComparing RBF to Tanimoto for SVR on ECFP\n')
        # Load data
        tetrah_folder = '../test_data/tetrahymena/'
        tetrah_y_path = f'{tetrah_folder}tetrah_y.csv'
        tetrah_ecfp_path = f'{tetrah_folder}tetrah_ecfp.csv'
        X, y = load_data(tetrah_ecfp_path, tetrah_y_path)
        # Run RBF Evaluator
        rbf_evaluator = RegressionEvaluator(
            model=parse_model('regression', 'svm_rbf'),
            repetitions=5,
            n_splits=5,
            verbose=False,
            scale=False
        )
        rbf_evaluator.perform(X, y)
        # Run Tanimoto Evaluator
        tan_evaluator = RegressionEvaluator(
            model=parse_model('regression', 'svm_tanimoto'),
            repetitions=5,
            n_splits=5,
            verbose=False,
            scale=False
        )
        tan_evaluator.perform(X, y)
        # Compare
        rbf_quality = rbf_evaluator.test_ensemble_quality
        tan_quality = tan_evaluator.test_ensemble_quality
        print(f'RBF R^2:      {rbf_quality:.3f}')
        print(f'Tanimoto R^2: {tan_quality:.3f}')
        self.assertTrue(tan_quality > rbf_quality)

    def test_bootstrapping(self):
        """Checks if subsampling based on bootstrapping works."""
        # Classification
        cyp1a2_folder = '../test_data/CYP1A2_klingspohn/'
        cyp1a2_moe_path = f'{cyp1a2_folder}CYP1A2_MOE.csv'
        cyp1a2_y_path = f'{cyp1a2_folder}CYP1A2_y.csv'
        X_cl, y_cl = load_data(cyp1a2_moe_path, cyp1a2_y_path)
        model = parse_model('classification', 'svm_rbf')
        print(f'Testing bootstrapping for classification.')
        run_evaluation(
            model=model,
            task='classification',
            X=X_cl,
            y=y_cl,
            verbose=False,
            repetitions=10,
            bootstrapping=True,
            seed=0,
            path=None,
            follow_up=self.report_and_assert_overall_quality
        )
        # Regression
        tetrah_folder = '../test_data/tetrahymena/'
        tetrah_rdkit_path = f'{tetrah_folder}tetrah_rdkit.csv'
        tetrah_y_path = f'{tetrah_folder}tetrah_y.csv'
        X_rg, y_rg = load_data(tetrah_rdkit_path, tetrah_y_path)
        model = parse_model('regression', 'svm_rbf')
        print(f'Testing bootstrapping for regression.')
        run_evaluation(
            model=model,
            task='regression',
            X=X_rg,
            y=y_rg,
            verbose=False,
            repetitions=10,
            bootstrapping=True,
            seed=0,
            path=None,
            follow_up=self.report_and_assert_overall_quality
        )


if __name__ == '__main__':
    unittest.main()

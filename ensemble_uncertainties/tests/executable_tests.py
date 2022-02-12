
"""Tests concerning the exectuable.

Must be run from 
    ensemble_uncertainties/ensemble_uncertainties/
in order to find the test input files.
"""

import os
import unittest

from ensemble_uncertainties.automatize import run_evaluation

from ensemble_uncertainties.executable import load_data, parse_model


class ExecutableTests(unittest.TestCase):

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
        # Get inputs for run_evaluation function
        model = parse_model(task, model_name)
        print(f'Testing {model_name} {task}.')
        # Define follow-up function to inspect fitted evaluator.
        # Train and test accuracy should be at least 60%.
        def report_and_assert(ev):
            tr_quality = ev.train_ensemble_quality
            te_quality = ev.test_ensemble_quality
            print(f'Train: {tr_quality:.3f}  Test: {te_quality:.3f}')
            print()
            self.assertTrue(ev.train_ensemble_quality > .6)
            self.assertTrue(ev.test_ensemble_quality > .6)
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
        """Check if all classification and regression model evaluations run
        without errors and that the results are acceptable on the rather
        easy test data sets. The whole test takes about 1 minute and 20
        seconds on my RTX 3080 Ti.
        """
        # Test classification
        cla = 'classification'
        cyp1a2_folder = '../test_data/CYP1A2_klingspohn/'
        cyp1a2_moe_path = f'{cyp1a2_folder}CYP1A2_MOE_first100.csv'
        cyp1a2_y_path = f'{cyp1a2_folder}CYP1A2_y_first100.csv'
        cyp1a2_moe, cyp1a2_y = load_data(cyp1a2_moe_path, cyp1a2_y_path)
        for m_name in ['shallow', 'deep', 'svm_rbf', 'xgb', 'rf']:
            self.error_free_evaluation(cla, m_name, cyp1a2_moe, cyp1a2_y)
        # Test regression
        reg = 'regression'
        tetrah_folder = '../test_data/tetrahymena/'
        tetrah_rdkit_path = f'{tetrah_folder}tetrah_rdkit_first100.csv'
        tetrah_y_path = f'{tetrah_folder}tetrah_y_first100.csv'
        tetrah_rdkit, tetrah_y = load_data(tetrah_rdkit_path, tetrah_y_path)
        for m_name in ['shallow', 'deep', 'svm_rbf', 'xgb', 'rf']:
            self.error_free_evaluation(reg, m_name, tetrah_rdkit, tetrah_y)
        # Test Tanimoto-SVM
        tetrah_ecfp_path = f'{tetrah_folder}tetrah_ecfp_first100.csv'
        tetrah_ecfp, _ = load_data(tetrah_ecfp_path, tetrah_y_path)
        tetrah_ecfp = tetrah_ecfp.astype('int64')
        self.error_free_evaluation(reg, 'svm_tanimoto', tetrah_ecfp, tetrah_y)


if __name__ == '__main__':
    unittest.main()

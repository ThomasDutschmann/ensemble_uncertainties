
"""Base class. Contains core functionality. Ready to be extended.

The Evaluator was designed to keep track of all scalers, variance thresholds,
models, and predictions for each split in each iteration to allow for a
complete analysis afterwards, if necessary.
"""

import numpy as np
import pandas as pd

from abc import abstractmethod

from copy import deepcopy

from datetime import datetime

from ensemble_uncertainties.constants import (
    N_REPS,
    N_SPLITS,
    RANDOM_SEED,
    V_THRESHOLD
)
from ensemble_uncertainties.evaluators.evaluator_support import (
    bootstrap,
    format_time_elapsed,
    make_columns,
    make_array,
    print_summary,
    random_int32,
    scale_and_filter,
    use_tqdm
)

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class Evaluator:
    """ Object to Conveniently examine the ensemble-based applicability 
    domain measure for different models, tasks, data sets, and settings.

    For parameters, please see help(Evaluator.__init__).
    """

    def __init__(self, model, verbose=True, repetitions=N_REPS,
            n_splits=N_SPLITS, seed=RANDOM_SEED, scale=True,
            v_threshold=V_THRESHOLD, bootstrapping=False):
        """Initializer, sets constants and initializes empty tables.
        
        Parameters
        ----------
        model : object
            A fittable model object, e.g. sklearn.svm.SVC()
            that implements the methods fit(X, y) and predict(X)
        verbose : bool
            Whether progress is constantly reported, default: True
        repetitions : int
            Repetitions of the n-fold validation, default: REPS
        n_splits : int
            number of splits in KFold, default: N_SPLITS
        seed : int
            Seed to use for splitting, default: RANDOM_SEED
        scale : bool
            Whether standardize variables, default: True
        v_threshold : float
            The variance threshold to apply after normalization, variables
            with a variance below will be removed, default: V_THRESHOLD
        bootstrapping : bool
            If True, bootstrapping will be used to generate the subsamples,
            default: False. n_splits will be automatically set to 1.
        """
        # Set given parameters
        self.model = model
        self.verbose = verbose
        self.repetitions = repetitions
        self.n_splits = n_splits
        self.seed = seed
        self.scale = scale
        self.v_threshold = v_threshold
        self.bootstrapping = bootstrapping
        # Set n_splits to 1 in case of bootstrap:
        if bootstrapping:
            self.n_splits = 1
        # Initialize list- and table-like members
        # to conveniently append data to them
        self.train_preds = pd.DataFrame()
        self.test_preds = pd.DataFrame()
        self.train_interm_preds = pd.DataFrame()
        self.test_interm_preds = pd.DataFrame()
        self.train_ensemble_preds = pd.DataFrame()
        self.test_ensemble_preds = pd.DataFrame()
        self.train_qualities = list()
        self.test_qualities = list()
        # Set type of feature scaling:
        self.scaler_class = StandardScaler

    def initialize_before_run(self):
        """Sets seed, initializes empty matrices and starts timer."""
        # Set seed
        np.random.seed(self.seed)
        # Set random states for each splitting in each repetition
        self.random_states = random_int32(self.repetitions, seed=self.seed)
        # Allocate places for all model-specific transformers
        # so that we can acces them with row and column indexes
        self.vt_filters = make_array((self.repetitions, self.n_splits))
        self.scalers = make_array((self.repetitions, self.n_splits))
        self.models = make_array((self.repetitions, self.n_splits))
        self.train_predictive_quality = np.zeros(self.repetitions)
        self.test_predictive_quality = np.zeros(self.repetitions)
        self.times = list()
        # Start timer
        self.overall_start_time = datetime.now()

    def perform(self, X, y):
        """Runs an evaluation for a given data set.

        Parameters
        ----------
        X : DataFrame
            Matrix of dependent variables. Index and header must be provided,
            name of the index column must be 'id'.
        y : DataFrame
            Vector of output variables.Index and header must be provided,
            name of the index column must be 'id'.
        """
        self.X = X
        self.y = y
        self.initialize_before_run()
        rep_iter = use_tqdm(range(self.repetitions), not self.verbose)
        for rep_index in rep_iter:
            if self.verbose:
                print(f'Repetition {rep_index+1}/{self.repetitions}')
            # Initialize intermediate results collectors
            self.train_interm_preds = pd.DataFrame()
            self.test_interm_preds = pd.DataFrame()
            # Run single repetition (= complete splitted evaluation once),
            # get intermediate results (performance of single repetition)
            intermediate_results = self.single_repetition(rep_index)
            train_quality, test_quality = intermediate_results
            self.train_qualities.append(train_quality)
            self.test_qualities.append(test_quality)
            if self.verbose:
                print(f'Train {self.metric_name}: {train_quality:.3f}')
                print(f'Test {self.metric_name}:  {test_quality:.3f}\n')
        self.finalize()

    def make_splits(self, rep_index):
        """Generates train/test splits.

        Parameters
        ----------
        rep_index : int
            Index of current repetition
        
        Returns
        -------
        list
            List of train/test index pairs
        """
        state = self.random_states[rep_index]
        if self.bootstrapping:
            splits = [bootstrap(self.X, random_state=state)]
        else:
            kfold = KFold(n_splits=self.n_splits,
                random_state=state, shuffle=True)
            splits = kfold.split(self.X)
        split_iter = use_tqdm(list(enumerate(splits)), self.verbose)
        return split_iter

    def single_repetition(self, rep_index):
        """Performs a single repetition (once completely through all splits).

        Parameters
        ----------
        rep_index : int
            The index of the current repetition

        Returns
        -------
        DataFrame, float, DataFrame, float
            Train predictions & performance, test predictions & performance
        """
        # Define splitting scheme
        split_iter = self.make_splits(rep_index)
        # Run all folds
        for split_index, (train_id, test_id) in split_iter:
            # Select train and test fraction
            X_tr, X_te = self.X.iloc[train_id], self.X.iloc[test_id]
            y_tr = self.y.iloc[train_id]
            # Fit and predict (timed)
            start_time = datetime.now() 
            self.single_evaluation(rep_index, split_index, X_tr, X_te, y_tr)
            time_elapsed = datetime.now() - start_time
            took = format_time_elapsed(time_elapsed)
            self.times.append(took)
        # Get train and test predictive quality
        _, train_quality = self.handle_predictions(self.train_interm_preds)
        _, test_quality = self.handle_predictions(self.test_interm_preds)
        return train_quality, test_quality

    def single_evaluation(self, rep_index, split_index, X_tr, X_te, y_tr):
        """Performs the evaluation of a single split (in case
        of KFold: training on (k-1)/kths, testing on 1/kth).

        Parameters
        ----------
        rep_index : int
            The index of the current repetition
        split_index : int
            The index of the current split
        X_tr : DataFrame
            Train inputs
        X_te : DataFrame
            Test inputs
        y_tr : DataFrame
            Train outputs
        """
        # Preprocess
        saf_result = scale_and_filter(X_tr, X_te,
            scaler_class=self.scaler_class, scale=self.scale,
            v_threshold=self.v_threshold)
        X_tr_sc_filt, X_te_sc_filt, vt, scaler = saf_result
        # Store filter and scaler as member
        self.vt_filters[rep_index][split_index] = vt
        self.scalers[rep_index][split_index] = scaler
        # Construct model (fit)
        # "We clone the estimator to make sure that all the folds are
        # independent, and that it is pickle-able" -- sklearn
        model = deepcopy(self.model)
        model.fit(X_tr_sc_filt, y_tr.values.ravel())
        self.models[rep_index][split_index] = model
        # Predict
        self.compute_predictions(model, X_tr_sc_filt, X_te_sc_filt)

    def compute_predictions(self, model, X_train, X_test):
        """Predicts train and test outputs, stores them in DataFrames.

        Parameters
        ----------
        model : object
            Fitted model object that predicts the
            outputs with the predict() method
        X_train : DataFrame
            Train inputs
        X_test : DataFrame
            Test inputs
        """
        # Compute predictions
        # Drop duplicate train entries when bootstrapping was used
        if self.bootstrapping:
            X_train = X_train.drop_duplicates(keep='first')
        tr_preds = pd.DataFrame(model.predict(X_train), index=X_train.index)
        te_preds = pd.DataFrame(model.predict(X_test), index=X_test.index)
        # Extend tables with train predictions
        self.train_interm_preds = pd.concat(
            [self.train_interm_preds, tr_preds], axis=1)
        self.train_preds = pd.concat(
            [self.train_preds, tr_preds], axis=1)
        # Extend tables with test predictions
        self.test_interm_preds = pd.concat(
            [self.test_interm_preds, te_preds], axis=1)
        self.test_preds = pd.concat(
            [self.test_preds, te_preds], axis=1)

    def finalize(self):
        """Assigns result tables and measures overall runtime"""
        self.assign_ensemble_performances()
        columns = make_columns(self.repetitions, self.n_splits)
        self.train_preds.columns = columns
        self.test_preds.columns = columns
        self.overall_run_time = datetime.now() - self.overall_start_time
        if self.verbose:
            print_summary(self.overall_run_time, self.metric_name, 
                self.train_ensemble_quality, self.test_ensemble_quality)

    def assign_ensemble_performances(self):
        """Computes and assigns final quality metrics to this evaluator."""
        # Compute
        overall_train_r = self.handle_predictions(self.train_preds)
        overall_test_r = self.handle_predictions(self.test_preds)
        train_results, train_quality = overall_train_r
        test_results, test_quality = overall_test_r
        # Assign
        self.train_ensemble_preds = train_results
        self.train_ensemble_quality = train_quality
        self.test_ensemble_preds = test_results
        self.test_ensemble_quality = test_quality

    def set_scaler_class(self, scaler_class):
        """Changes the feature scaling method,
        default: sklearn.preprocessing.StandardScaler.

        Parameters
        ----------
        scaler_class : scaler
            Scaler initializer, must result in an object
            that implements .fit(X) and .transform(X)
        """
        self.scaler_class = scaler_class

    @abstractmethod
    def handle_predictions(self, predictions):
        """Actual implementation depends on whether
        a classification or regression is performed.

        Parameters
        ----------
        predictions : DataFrame
            The predictions from a single repetition

        Returns
        -------
        DataFrame, float
            The ensemble predictions as DataFrame and the metric performance.
        """
        raise NotImplementedError()


"""Collection of machine learning methods to apply."""

from ensemble_uncertainties.neural_networks.nn_models import (
    DeepNeuralClassifier,
    DeepNeuralRegressor,
    DeepDropoutRegressor,
    DeepMCDropoutRegressor,
    ShallowNeuralClassifier,
    ShallowNeuralRegressor
)
from ensemble_uncertainties.utils.kernel_functions import tanimoto

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

from pandas import MultiIndex, Int16Dtype
from xgboost import XGBClassifier, XGBRegressor


# Custom models must be imported:
#from my_module.my_file import my_fancy_regression_model


# Classification
# --------------
models = dict()
models['classification'] = dict()
models['classification']['rf'] = RandomForestClassifier()
models['classification']['dt'] = DecisionTreeClassifier()
models['classification']['svm_rbf'] = SVC(kernel='rbf')
models['classification']['svm_tanimoto'] = SVC(kernel=tanimoto)
models['classification']['xgb'] = XGBClassifier(silent=1, verbosity=0)
models['classification']['deep'] = DeepNeuralClassifier()
models['classification']['shallow'] = ShallowNeuralClassifier()


# Regression
# ----------
models['regression'] = dict()
models['regression']['rf'] = RandomForestRegressor()
models['regression']['dt'] = DecisionTreeRegressor()
models['regression']['svm_rbf'] = SVR(kernel='rbf')
models['regression']['svm_tanimoto'] = SVR(kernel=tanimoto)
models['regression']['xgb'] = XGBRegressor(silent=1, verbosity=0)
models['regression']['deep'] = DeepNeuralRegressor()
models['regression']['shallow'] = ShallowNeuralRegressor()
models['regression']['dropout'] = DeepDropoutRegressor()
models['regression']['mc_dropout'] = DeepMCDropoutRegressor()


# Add your custom model like so:
#models['regression']['MODEL_NAME'] = my_fancy_regression_model
# "MODEL_NAME" (or however you want to call your model)
# will be available automatically when called with the -m option.


"""Collection of machine learning methods to apply."""

from ensemble_uncertainties.neural_estimators.neural_estimator import (
    DeepNeuralClassifier,
    DeepNeuralRegressor,
    ShallowNeuralClassifier,
    ShallowNeuralRegressor
)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

from xgboost import XGBClassifier, XGBRegressor

# Custom models must be imported:
#from my_module.my_file import my_fancy_regression_model


# Classification
# --------------
models = dict()
models['classification'] = dict()
models['classification']['rf'] = RandomForestClassifier()
models['classification']['svm'] = SVC()
models['classification']['xgb'] = XGBClassifier(use_label_encoder=False)
models['classification']['deep'] = DeepNeuralClassifier()
models['classification']['shallow'] = ShallowNeuralClassifier()


# Regression
# ----------
models['regression'] = dict()
models['regression']['rf'] = RandomForestRegressor()
models['regression']['svm'] = SVR()
models['regression']['xgb'] = XGBRegressor()
models['regression']['deep'] = DeepNeuralRegressor()
models['regression']['shallow'] = ShallowNeuralRegressor()


# Add your custom model like so:
#models['regression']['MODEL_NAME'] = my_fancy_regression_model
# "MODEL_NAME" (or however you want to call your model)
# will be available automatically when called with the -m option.

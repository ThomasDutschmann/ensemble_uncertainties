
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
models['classification']['RF'] = RandomForestClassifier()
models['classification']['SVM'] = SVC()
models['classification']['XGB'] = XGBClassifier(use_label_encoder=False)
models['classification']['DL'] = DeepNeuralClassifier()
models['classification']['SL'] = ShallowNeuralClassifier()


# Regression
# ----------
models['regression'] = dict()
models['regression']['RF'] = RandomForestRegressor()
models['regression']['SVM'] = SVR()
models['regression']['XGB'] = XGBRegressor()
models['regression']['DL'] = DeepNeuralRegressor()
models['regression']['SL'] = ShallowNeuralRegressor()


# Add your custom model like so:
#models['regression']['MODEL_NAME'] = my_fancy_regression_model
# "MODEL_NAME" (or however you want to call your model)
# will be available automatically when called with the -m option.

"""Collection of machine learning methods that
provide uncertainties when predicting.
"""

from su_custom_models import (
    KFoldEnsembleRegressorWrapper,
    MCDropoutRegressorWrapper,
    UQRandomForestRegressor
)

# Regression
# ----------
models = dict()
models['rf'] = UQRandomForestRegressor()
models['mc_dropout'] = MCDropoutRegressorWrapper()
models['ensemble_rf'] = KFoldEnsembleRegressorWrapper.RFRWrapper()
models['ensemble_dropout'] = KFoldEnsembleRegressorWrapper.MCDropoutWrapper()

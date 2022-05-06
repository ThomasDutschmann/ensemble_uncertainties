
"""Collection of machine learning methods that
provide uncertainties when predicting.
"""

from su_custom_models import (
    MCDropoutRegressorWrapper,
    UQRandomForestRegressor
)

# Regression
# ----------
models = dict()
models['rf'] = UQRandomForestRegressor()
models['mc_dropout'] = MCDropoutRegressorWrapper()

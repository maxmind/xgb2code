#!/usr/bin/env python
"""Generate a single-quantile reg:quantileerror model to test"""

import random
import numpy as np
import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split
import sklearn.datasets
import xgboost as xgb

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

data = sklearn.datasets.load_diabetes(return_X_y=False)
df = pd.DataFrame(data.data)  # pylint: disable=no-member
# make some columns NaN at random.
df = df.mask(np.random.random(df.shape) < 0.2)

X_train, X_test, y_train, y_test = train_test_split(
    df,
    data.target,  # pylint: disable=no-member
    test_size=0.30,
)

XGB_MISSING = np.nan
NUM_ROUNDS = 500
DEFAULT_XGB_PARAMS = {
    # A single quantile_alpha produces a single-output model. base_score is
    # left unset so XGBoost estimates it (the target's median for alpha=0.5),
    # exercising the raw, auto-derived intercept on the identity output path.
    "objective": "reg:quantileerror",
    "quantile_alpha": 0.5,
    "eta": 0.1,
    "nthread": 1,
    "seed": RANDOM_SEED,
    "tree_method": "hist",
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bylevel": 0.9,
    "colsample_bytree": 0.9,
    "colsample_bynode": 0.9,
}

# Set feature names rather than leave them as the defaults of "0", "1", etc, to
# test that it will work with xgb2code.
feature_names = data.feature_names  # pylint: disable=no-member

dtrain = xgb.DMatrix(
    data=X_train,
    label=y_train,
    missing=XGB_MISSING,
    feature_names=feature_names,
)
dtest = xgb.DMatrix(
    data=X_test,
    label=y_test,
    missing=XGB_MISSING,
    feature_names=feature_names,
)
evallist = [(dtrain, "train"), (dtest, "eval")]

booster = xgb.train(
    DEFAULT_XGB_PARAMS,
    dtrain,
    NUM_ROUNDS,
    evals=evallist,
)

booster.save_model("model.json")
X_test.to_csv("xtest.csv", header=False, index=False)
pd.Series(booster.predict(dtest)).to_csv("preds.csv", header=False, index=False)

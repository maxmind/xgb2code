#!/usr/bin/env python
"""Generate a model with categorical features to test.

Categorical features are passed as integer category codes with per-feature
types ("c" for categorical, "q" for numeric) so the values written to xtest.csv
are exactly the codes the model splits on. Missing values are included so the
generated code's missing-value handling (default_left) is exercised for both
numeric and categorical splits.
"""

import random
import numpy as np
import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split
import xgboost as xgb

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

N = 600
# Three numeric features and three categorical features of differing
# cardinality (3, 5, 8), interleaved so split_indices mix the two kinds.
num0 = np.random.normal(size=N)
cat0 = np.random.randint(0, 3, size=N).astype(float)
num1 = np.random.normal(size=N)
cat1 = np.random.randint(0, 5, size=N).astype(float)
num2 = np.random.normal(size=N)
cat2 = np.random.randint(0, 8, size=N).astype(float)

X = np.column_stack([num0, cat0, num1, cat1, num2, cat2])
feature_types = ["q", "c", "q", "c", "q", "c"]
feature_names = ["num0", "cat0", "num1", "cat1", "num2", "cat2"]

# The label depends on the categorical features (so categorical splits are
# learned) plus a numeric contribution.
y = (
    ((cat0 == 1) | (cat1 >= 3) | (np.isin(cat2, [0, 2, 5, 7])))
    & (num0 + num1 > -0.5)
).astype(int)

# Make some values NaN at random, across both numeric and categorical columns.
mask = np.random.random(X.shape) < 0.15
X[mask] = np.nan

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
)

XGB_MISSING = np.nan
NUM_ROUNDS = 50
DEFAULT_XGB_PARAMS = {
    "objective": "binary:logistic",
    "eta": 0.2,
    "eval_metric": "auc",
    "nthread": 1,
    "seed": RANDOM_SEED,
    "tree_method": "hist",
    "max_depth": 6,
    # Set max_cat_to_onehot explicitly (rather than relying on the XGBoost
    # default, which could change) so this single fixture deterministically
    # covers both categorical split encodings. With cardinalities 3, 5, and 8,
    # the threshold of 4 means cat0 (3 categories) is split one-hot style
    # (single-category sets) while cat1 and cat2 use partition-based,
    # multi-category sets.
    "max_cat_to_onehot": 4,
}

dtrain = xgb.DMatrix(
    data=X_train,
    label=y_train,
    missing=XGB_MISSING,
    feature_names=feature_names,
    feature_types=feature_types,
    enable_categorical=True,
)
dtest = xgb.DMatrix(
    data=X_test,
    label=y_test,
    missing=XGB_MISSING,
    feature_names=feature_names,
    feature_types=feature_types,
    enable_categorical=True,
)
evallist = [(dtrain, "train"), (dtest, "eval")]

booster = xgb.train(
    DEFAULT_XGB_PARAMS,
    dtrain,
    NUM_ROUNDS,
    evals=evallist,
)

booster.save_model("model.json")
pd.DataFrame(X_test).to_csv("xtest.csv", header=False, index=False)
pd.Series(booster.predict(dtest)).to_csv("preds.csv", header=False, index=False)

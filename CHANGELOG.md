# CHANGELOG

## 1.2.0

- Added support for `-Infinity` values in a tree's `split_conditions`, which
  XGBoost's histogram tree method emits for splits that route on missingness
  (missing values follow the node's default direction and every present value
  goes to the other child). Such models previously failed to parse. Other
  non-finite split values (`+Infinity` or `NaN`) and non-finite leaf values now
  produce a clear error.

## 1.1.0 (2026-06-17)

- Added support for categorical splits (models trained with
  `enable_categorical`). Categorical features are passed as their integer
  category codes in the `data` slice, the same encoding XGBoost uses internally.

## 1.0.0 (2026-06-15)

- Refactored the codebase to allow using the code generation functionality as a
  library.
- Added support for the `reg:logistic`, `binary:logitraw`, `reg:squarederror`,
  `reg:linear`, `reg:absoluteerror`, `reg:pseudohubererror`, and
  `reg:quantileerror` objectives. The model's `base_score` is now applied as an
  intercept, and the sigmoid is only applied for the logistic objectives
  (`binary:logistic` and `reg:logistic`). Unsupported objectives, multi-output
  models (such as multi-target regression or multi-quantile
  `reg:quantileerror`), and forest models (`num_parallel_tree` greater than one)
  trained with early stopping now produce a clear error.

## 0.1.0 (2022-10-06)

- Initial version.

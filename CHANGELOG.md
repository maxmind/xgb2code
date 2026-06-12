# CHANGELOG

## 1.0.0 (2026-06-12)

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

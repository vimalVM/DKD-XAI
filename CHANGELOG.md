# Changelog — updated project package

## Updated package

- Integrated the 2013–2014 and 2015–2016 NHANES cohorts (790 and 892 participants) while preserving the original 2,342 rows and their cell values.
- Added `merge_historical_cycles.py` to reproduce the 4-cycle, 4,024-row CSV from the original base CSV and the supplied H/I XPT files.
- Added `validate_project.py` for row counts, unique participant IDs, label consistency, leakage checks, and base-row preservation.
- Updated the temporal-validation implementation to tune/train only on cycles before the newest cycle and evaluate on the newest cycle separately; fixed Youden-threshold selection to exclude an infinite ROC threshold.
- Made model searches use one worker by default to avoid resource over-subscription; added `--skip_mlp` for quick smoke tests only.
- Fixed explanation-agent handling of optional race dummy features, missing participant IDs, source-linked evidence, the exact risk percentage, and citation checks.
- Clarified that SHAP is model attribution rather than causality; replaced overly strong evidence wording and added source URLs.
- Updated the build guide and README with cross-sectional label limits, lack of NHANES survey weighting, correct cohort counts, and a warning that the historical model scores do not apply to the expanded dataset.

## Validation performed

- Python syntax compilation passed for all project scripts.
- `validate_project.py` passed: 4,024 rows, 22 columns, unique participant IDs, consistent outcome labels, and preserved base values.
- Re-running `merge_historical_cycles.py` reproduced the supplied expanded CSV's values across all rows and columns.
- Quick model smoke test passed for `dkd_any` with the MLP and temporal-validation search skipped; the deterministic agent successfully explained a generated XGBoost model and passed its surface-level faithfulness check.
- The full model-comparison run with MLP and full temporal searches was not completed for this package. Final performance claims require running the full experiment and reviewing its outputs.

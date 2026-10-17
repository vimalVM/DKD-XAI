#!/usr/bin/env python3
"""Validate the expanded cohort, label logic, leakage boundary, and base-row preservation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "seqn", "cycle", "age", "female", "race_ethnicity_code", "hba1c", "bmi",
    "waist_cm", "sbp", "dbp", "hdl", "total_chol", "dm_diagnosed",
    "insulin_use", "diabetes_pills", "dm_duration_years", "dkd_any", "early_dkd",
    "dkd_category", "uacr_mg_g", "serum_creatinine_mg_dl", "egfr_ckd_epi_2021",
]
FEATURES = [
    "age", "female", "hba1c", "bmi", "waist_cm", "sbp", "dbp", "hdl",
    "total_chol", "dm_diagnosed", "insulin_use", "diabetes_pills", "dm_duration_years",
]
LEAKAGE_COLUMNS = {"uacr_mg_g", "serum_creatinine_mg_dl", "egfr_ckd_epi_2021", "dkd_any", "early_dkd", "dkd_category"}
EXPECTED_CYCLES = {
    "2013-2014": 790,
    "2015-2016": 892,
    "2017-2020": 1455,
    "2021-2023": 887,
}


def assert_values_equal(left: pd.DataFrame, right: pd.DataFrame, description: str) -> None:
    issues = []
    if list(left.columns) != list(right.columns):
        raise AssertionError(f"{description}: column order differs")
    if len(left) != len(right):
        raise AssertionError(f"{description}: row count differs ({len(left)} vs {len(right)})")
    for col in left.columns:
        a, b = left[col].reset_index(drop=True), right[col].reset_index(drop=True)
        same = a.eq(b) | (a.isna() & b.isna())
        if not bool(same.all()):
            issues.append(f"{col}: {(~same).sum()} mismatched cells")
    if issues:
        raise AssertionError(f"{description}: " + "; ".join(issues))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="dkd_nhanes_final.csv")
    ap.add_argument("--base_csv", default="data/dkd_nhanes_2017_2023_base.csv")
    args = ap.parse_args()

    csv_path, base_path = Path(args.csv), Path(args.base_csv)
    df = pd.read_csv(csv_path)
    base = pd.read_csv(base_path)

    if list(df.columns) != EXPECTED_COLUMNS:
        raise AssertionError(f"Expected the 22-column project schema; found {len(df.columns)} columns")
    missing_features = sorted(set(FEATURES) - set(df.columns))
    if missing_features:
        raise AssertionError(f"Missing model features: {missing_features}")
    leakage = sorted(set(FEATURES) & LEAKAGE_COLUMNS)
    if leakage:
        raise AssertionError(f"Leakage columns are included as model inputs: {leakage}")
    if df["seqn"].isna().any() or df["seqn"].duplicated().any():
        raise AssertionError("SEQN must be present and unique for every row")

    actual_cycles = df["cycle"].value_counts().to_dict()
    if actual_cycles != EXPECTED_CYCLES:
        raise AssertionError(f"Unexpected cycle counts: {actual_cycles}")
    if len(df) != sum(EXPECTED_CYCLES.values()):
        raise AssertionError(f"Unexpected row count: {len(df)}")

    uacr = pd.to_numeric(df["uacr_mg_g"], errors="coerce")
    egfr = pd.to_numeric(df["egfr_ckd_epi_2021"], errors="coerce")
    expected_any = ((uacr >= 30) | (egfr < 60)).astype(int)
    if not bool(expected_any.eq(df["dkd_any"]).all()):
        raise AssertionError("dkd_any does not match the UACR/eGFR label definition")

    none = (uacr < 30) & (egfr >= 60)
    early = (uacr >= 30) & (uacr <= 300) & (egfr >= 60)
    expected_category = np.select([none, early], ["none", "early"], default="advanced")
    if not bool(pd.Series(expected_category).eq(df["dkd_category"]).all()):
        raise AssertionError("dkd_category does not match the UACR/eGFR label definition")
    expected_early = pd.Series(np.nan, index=df.index, dtype=float)
    expected_early.loc[none] = 0.0
    expected_early.loc[early] = 1.0
    same_early = expected_early.eq(df["early_dkd"]) | (expected_early.isna() & df["early_dkd"].isna())
    if not bool(same_early.all()):
        raise AssertionError("early_dkd does not match the none/early/advanced label definition")

    # The original base rows must still be the first rows, with identical values.
    base_cols = list(base.columns)
    if base_cols != EXPECTED_COLUMNS:
        raise AssertionError("The base CSV schema differs from the project schema")
    assert_values_equal(base, df.iloc[:len(base)][base_cols].reset_index(drop=True), "Base-row preservation")

    print("Validation passed")
    print(f"Rows: {len(df):,} | columns: {len(df.columns)} | unique SEQN: {df['seqn'].nunique():,}")
    print("Cycle counts:")
    print(df["cycle"].value_counts().sort_index().to_string())
    print(f"dkd_any positives: {int(df['dkd_any'].sum()):,}")
    print(f"early_dkd positives: {int(df['early_dkd'].eq(1).sum()):,}")
    print(f"advanced rows excluded from early_dkd: {int(df['early_dkd'].isna().sum()):,}")
    print("All original base CSV cells are preserved; no kidney label/measurement is a model input.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Append NHANES 2013-2014 and 2015-2016 rows to the supplied 2017-2023 CSV.

This deliberately preserves the existing 2017-2023 rows and their values. The older
cycles are constructed from real NHANES XPT participant files, using the same columns
and cohort/label definitions as the updated CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "seqn", "cycle", "age", "female", "race_ethnicity_code", "hba1c", "bmi",
    "waist_cm", "sbp", "dbp", "hdl", "total_chol", "dm_diagnosed",
    "insulin_use", "diabetes_pills", "dm_duration_years", "dkd_any", "early_dkd",
    "dkd_category", "uacr_mg_g", "serum_creatinine_mg_dl", "egfr_ckd_epi_2021",
]

FILES = ["DEMO", "DIQ", "GHB", "ALB_CR", "BIOPRO", "BMX", "BPX", "HDL", "TCHOL"]


def read_xpt(raw_dir: Path, name: str, suffix: str) -> pd.DataFrame:
    path = raw_dir / f"{name}_{suffix}.xpt"
    if not path.exists():
        raise FileNotFoundError(f"Missing NHANES input file: {path}")
    frame = pd.read_sas(path, format="xport")
    if "SEQN" not in frame:
        raise ValueError(f"{path.name} has no SEQN participant identifier")
    if frame["SEQN"].duplicated().any():
        raise ValueError(f"Duplicate SEQN values found in {path.name}")
    return frame


def make_cycle(raw_dir: Path, suffix: str, cycle_label: str) -> pd.DataFrame:
    raw = {name: read_xpt(raw_dir, name, suffix) for name in FILES}
    demo = raw["DEMO"][["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3"]].copy()
    merged = demo
    for name in FILES[1:]:
        merged = merged.merge(raw[name], on="SEQN", how="inner", validate="one_to_one")

    # Cohort eligibility: adults with doctor-diagnosed diabetes or HbA1c >= 6.5,
    # with HbA1c, urine albumin/creatinine ratio, and serum creatinine available.
    age = pd.to_numeric(merged["RIDAGEYR"], errors="coerce")
    hba1c = pd.to_numeric(merged["LBXGH"], errors="coerce")
    diagnosis_code = pd.to_numeric(merged["DIQ010"], errors="coerce")
    doctor_diagnosed = diagnosis_code.eq(1)
    uacr = pd.to_numeric(merged["URDACT"], errors="coerce")
    creatinine = pd.to_numeric(merged["LBXSCR"], errors="coerce")
    eligible = (
        age.ge(18)
        & (doctor_diagnosed | hba1c.ge(6.5))
        & hba1c.notna()
        & uacr.notna()
        & creatinine.notna()
    )
    merged = merged.loc[eligible].copy()
    age = pd.to_numeric(merged["RIDAGEYR"], errors="coerce")
    hba1c = pd.to_numeric(merged["LBXGH"], errors="coerce")
    diagnosis_code = pd.to_numeric(merged["DIQ010"], errors="coerce")
    doctor_diagnosed = diagnosis_code.eq(1)
    uacr = pd.to_numeric(merged["URDACT"], errors="coerce")
    creatinine = pd.to_numeric(merged["LBXSCR"], errors="coerce")
    female = pd.to_numeric(merged["RIAGENDR"], errors="coerce").eq(2)

    # CKD-EPI 2021 race-free eGFR based on serum creatinine, age and sex.
    sex_k = np.where(female, 0.7, 0.9)
    sex_alpha = np.where(female, -0.241, -0.302)
    scr_ratio = creatinine / sex_k
    egfr = (
        142.0
        * np.minimum(scr_ratio, 1.0) ** sex_alpha
        * np.maximum(scr_ratio, 1.0) ** (-1.200)
        * (0.9938 ** age)
        * np.where(female, 1.012, 1.0)
    ).round(1)

    systolic_columns = [c for c in merged.columns if c.startswith("BPXSY")]
    diastolic_columns = [c for c in merged.columns if c.startswith("BPXDI")]
    sbp = merged[systolic_columns].replace({0: np.nan}).mean(axis=1).round(3)
    dbp = merged[diastolic_columns].replace({0: np.nan}).mean(axis=1).round(3)

    # Keep treatment-feature encoding compatible with the original CSV convention:
    # people without a doctor-reported diagnosis are encoded 0 for these two flags;
    # among diagnosed participants, explicit yes/no becomes 1/0 and unknown stays missing.
    def treatment_flag(column: str) -> pd.Series:
        raw_flag = pd.to_numeric(merged[column], errors="coerce").map({1.0: 1.0, 2.0: 0.0})
        return raw_flag.where(doctor_diagnosed, 0.0)

    diagnosed_age = pd.to_numeric(merged["DID040"], errors="coerce").replace({777: np.nan, 999: np.nan})
    duration = (age - diagnosed_age).where(doctor_diagnosed)
    duration = duration.where(diagnosed_age.le(age) & diagnosed_age.ge(0))

    # Kidney measures define outcomes only; none enter the model feature list.
    dkd_any = ((uacr >= 30) | (egfr < 60)).astype(int)
    is_none = (uacr < 30) & (egfr >= 60)
    is_early = (uacr >= 30) & (uacr <= 300) & (egfr >= 60)
    category = np.select([is_none, is_early], ["none", "early"], default="advanced")
    early_label = pd.Series(np.nan, index=merged.index, dtype=float)
    early_label.loc[is_none] = 0.0
    early_label.loc[is_early] = 1.0

    frame = pd.DataFrame({
        "seqn": pd.to_numeric(merged["SEQN"], errors="raise").astype(int),
        "cycle": cycle_label,
        "age": age,
        "female": female.astype(int),
        "race_ethnicity_code": pd.to_numeric(merged["RIDRETH3"], errors="coerce"),
        "hba1c": hba1c,
        "bmi": pd.to_numeric(merged["BMXBMI"], errors="coerce"),
        "waist_cm": pd.to_numeric(merged["BMXWAIST"], errors="coerce"),
        "sbp": sbp,
        "dbp": dbp,
        "hdl": pd.to_numeric(merged["LBDHDD"], errors="coerce"),
        "total_chol": pd.to_numeric(merged["LBXTC"], errors="coerce"),
        "dm_diagnosed": doctor_diagnosed.astype(int),
        "insulin_use": treatment_flag("DIQ050"),
        "diabetes_pills": treatment_flag("DIQ070"),
        "dm_duration_years": duration,
        "dkd_any": dkd_any,
        "early_dkd": early_label,
        "dkd_category": category,
        "uacr_mg_g": uacr,
        "serum_creatinine_mg_dl": creatinine,
        "egfr_ckd_epi_2021": egfr,
    })
    frame = frame[COLUMNS].sort_values("seqn").reset_index(drop=True)
    if frame["seqn"].duplicated().any():
        raise ValueError(f"Duplicate participant IDs within cycle {cycle_label}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_csv", default="data/dkd_nhanes_2017_2023_base.csv",
                        help="Original CSV containing the 2017-2023 cohorts")
    parser.add_argument("--raw_dir", default="data/raw_nhanes_cycles",
                        help="Directory containing the NHANES H/I XPT files")
    parser.add_argument("--output", default="dkd_nhanes_final.csv",
                        help="Path for the merged 4-cycle CSV")
    args = parser.parse_args()

    base_path = Path(args.base_csv)
    base = pd.read_csv(base_path)
    missing = [c for c in COLUMNS if c not in base.columns]
    if missing:
        raise ValueError(f"Base CSV is missing expected columns: {missing}")
    base = base[COLUMNS].copy()
    if base["seqn"].duplicated().any():
        raise ValueError("The base CSV already contains duplicate SEQN values")
    if base["seqn"].isna().any():
        raise ValueError("The base CSV contains missing SEQN values")

    raw_dir = Path(args.raw_dir)
    historical = pd.concat([
        make_cycle(raw_dir, "H", "2013-2014"),
        make_cycle(raw_dir, "I", "2015-2016"),
    ], ignore_index=True)
    overlap = set(base["seqn"].astype(int)) & set(historical["seqn"].astype(int))
    if overlap:
        raise ValueError(f"Historical rows overlap base SEQN IDs (sample: {sorted(overlap)[:10]})")

    merged = pd.concat([base, historical], ignore_index=True)[COLUMNS]
    if merged["seqn"].duplicated().any():
        raise ValueError("Duplicate SEQN values detected after the merge")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False, float_format="%.10g")

    print(f"Base rows preserved: {len(base):,}")
    print("Rows added:")
    print(historical["cycle"].value_counts().sort_index().to_string())
    print(f"Total rows: {len(merged):,}; columns: {len(merged.columns)}")
    print("Cycle counts:")
    print(merged["cycle"].value_counts().sort_index().to_string())
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()

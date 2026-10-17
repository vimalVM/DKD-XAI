# DKD-XAI — Build Guide

**Explainable AI for Early Prediction of Diabetic Kidney Disease**
Student: Vimal Vijayan | Roll No. 17030925059 | Mentor: Dr. Mayura Nagar

This is a build spec, not a tutorial — hand it to an agent (Antigravity or otherwise) as the
task description. It describes exactly what to build, in what order, with what acceptance
criteria, based on a version of this pipeline that has already been built and tested.

> **LLM provider constraint: use the Groq API, never the Anthropic API.**
> Every "AI agent" / explanation-generation step below must call **Groq**
> (`https://console.groq.com`, OpenAI-compatible endpoint, free tier, model
> `llama-3.3-70b-versatile`) via the `groq` Python package and the `GROQ_API_KEY`
> environment variable. Do not introduce `anthropic`, `ANTHROPIC_API_KEY`, or any
> Claude model string anywhere in this project — not for the agent, not for any
> helper script, not as a fallback.

---

## 1. Project Summary

Diabetic Kidney Disease (DKD) develops silently; by the time it's caught via standard
kidney labs, damage has often already occurred. This project predicts DKD risk **early**,
using only routine, pre-diagnosis clinical parameters — never the kidney lab values used
to diagnose it — then explains every prediction with SHAP and a Groq-powered natural-language
report grounded in a fixed evidence base (so the LLM can't invent clinical claims).

**Two phases, built in order:**
- **Phase 1** — data pipeline + ML/DL model comparison + explainability + calibration
- **Phase 2** — AI agent that turns a prediction into a patient-readable, evidence-grounded explanation (Groq)

Do not start Phase 2 before Phase 1's models are trained and a best model is selected —
the agent loads whatever model Phase 1 actually saved.

---

## 2. Dataset

**Source:** NHANES (CDC), downloaded manually from
https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx — this sandbox/agent cannot fetch
CDC files itself; the files must be supplied by the user.

**Cohort:** Adults (18+) with diabetes (doctor-diagnosed OR HbA1c ≥ 6.5%) who have both a
urine albumin/creatinine ratio (UACR) and serum creatinine measured.

**Interpretation limit:** NHANES is cross-sectional here. UACR/eGFR are measured at the same
survey visit as model features, and the dataset does not establish persistence for more than
three months. Therefore, labels are *single-visit kidney-marker proxies*, not confirmed clinical
CKD diagnoses, and the model is a cross-sectional risk-stratification/classification experiment,
not a validated forecast of future-onset disease. The cycle-based temporal holdout tests
transportability across survey cycles; it is not longitudinal follow-up of the same people.

**Cycles included in the updated CSV:**

| Cycle | NHANES file tag | Status in this package |
|---|---|---|
| 2013–2014 | `_H` (for example, `DEMO_H.xpt`) | Raw XPT files included; 790 eligible adults |
| 2015–2016 | `_I` (for example, `DEMO_I.xpt`) | Raw XPT files included; 892 eligible adults |
| 2017–March 2020 | `P_` prefix | Included in the base CSV; original raw files were not supplied here |
| Aug 2021–Aug 2023 | `_L` suffix | Included in the base CSV; original raw files were not supplied here |

The updated cohort pools real NHANES participant records only. `merge_historical_cycles.py`
reconstructs the H and I cohorts from the supplied XPT files and appends them to the original
2017–2023 base CSV. Do not fabricate or synthesize patient rows to inflate sample size.

**Files needed per cycle:** `DEMO`, `DIQ`, `GHB`, `ALB_CR`, `BIOPRO`, `BMX`, the cycle-appropriate blood-pressure file (`BPX` for H/I; `BPXO` for the newer cycles as applicable), `HDL`, and `TCHOL` (optionally `TRIGLY`, `SMQ`, `BPQ` for feature expansion — see §7).

**Labels** (kidney labs are used ONLY to build these — never as model inputs):
- `eGFR` — CKD-EPI 2021 race-free equation, from serum creatinine + age + sex
- `dkd_any` = 1 if `UACR ≥ 30 mg/g` OR `eGFR < 60`, else 0
- `dkd_category` = `none` (UACR<30, eGFR≥60) / `early` (UACR 30–300, eGFR≥60) / `advanced` (else)
- `early_dkd` = 1 for `early`, 0 for `none`, excluded (NaN) for `advanced`

**Reference implementation:** `merge_historical_cycles.py` → outputs `dkd_nhanes_final.csv`.
Current cohort: **4,024 adults** (790 from 2013–2014, 892 from 2015–2016, 1,455 from
2017–2020, and 887 from 2021–2023), **1,578 (39.2%)** with any DKD, **699** early-stage,
**2,446** with no kidney damage, and **879** classified as advanced for the `early_dkd` task.
The original 2,342-row dataset remains in `data/dkd_nhanes_2017_2023_base.csv` as the input
baseline. Counts and previously reported model scores for that smaller cohort are historical
and must not be presented as results for the expanded 4,024-row cohort.

---

## 3. Model Inputs — 13 features, no leakage

```
age, female, hba1c, bmi, waist_cm, sbp, dbp, hdl, total_chol,
dm_diagnosed, insulin_use, diabetes_pills, dm_duration_years
```

**Never include** `uacr_mg_g`, `serum_creatinine_mg_dl`, or `egfr_ckd_epi_2021` as model
inputs — they exist in the CSV only to have defined the label. Including them is the exact
leakage failure mode this project is designed to avoid (see §8 for why that matters).

---

## 4. Phase 1 — Train, Tune, Validate, Explain

**Reference implementation:** `dkd_model_pipeline.py`

1. **Split:** 80/20 stratified train/test.
2. **Models:** Logistic Regression, Random Forest, XGBoost (all via scikit-learn/XGBoost),
   and a PyTorch MLP (class-weighted BCE loss, AdamW, early stopping) — compare ML vs DL,
   don't assume one wins.
3. **Tuning:** `RandomizedSearchCV`, stratified CV scored on AUROC. Full runs use 5 folds for
   Logistic Regression, Random Forest and XGBoost, and 3 folds for the MLP. `--quick` uses
   smaller searches and 3 folds for the non-MLP models; `--skip_mlp` is for smoke tests only.
4. **Threshold:** Youden's J on out-of-fold training predictions (never on the test set).
5. **Report, for every model:** CV AUROC, test AUROC + 95% bootstrap CI, AUPRC, Brier score,
   sensitivity, specificity — **never report accuracy alone**; `early_dkd` is imbalanced
   (22.2% positive among eligible early-vs-none rows in the expanded cohort), so a model
   predicting "no DKD" for everyone scores roughly 78% accuracy
   while catching zero real cases. Accuracy must always be paired with sensitivity/specificity
   at the threshold being quoted.
6. **Temporal validation:** tune and train only on all cycles before the newest cycle, then
   evaluate on every row from the newest cycle. The newest cycle must be excluded from both
   tuning and fitting for this validation result. Here, train on 2013–2014 through 2017–2020
   and test on 2021–2023. This is cycle-based transportability testing, not longitudinal follow-up.
7. **Explainability:** the reference pipeline generates TreeSHAP plots and bootstrap rank-
   stability metrics specifically for XGBoost; KernelSHAP compares the MLP on a subset unless
   skipped; permutation importance is computed for the selected best model; and logistic odds
   ratios are saved as a transparent baseline. The Phase 2 agent separately computes SHAP for
   whichever best model was actually saved. Do not describe the XGBoost stability file as the
   selected model's stability result unless XGBoost was selected.
8. **Select the best model per task** by **CV AUROC** (not test AUROC, to avoid picking based
   on test-set noise), and save it as `best_model.joblib` with its feature list and threshold.

**Historical baseline results** for the old 2,342-row cohort (reduced hyperparameter search):

| Task | Best model (by CV AUROC) | Test AUROC | Sensitivity | Specificity |
|---|---|---|---|---|
| `dkd_any` | RandomForest | 0.708 | 54.2% | 74.6% |
| `early_dkd` | LogReg | 0.693 | 64.8% | 62.0% |

These are **not validated results for the expanded 4,024-row cohort**. Re-run the pipeline and
use its new output files before reporting performance. Do not assume the updated cohort will
improve AUROC; its scores must be measured. Always state the decision threshold alongside
sensitivity/specificity, and do not choose a model using test AUROC.

**Acceptance criteria for Phase 1:** `results/<task>/best_model.joblib` exists for both tasks,
with `model_comparison_holdout.csv`, `model_comparison_temporal.csv` (unless temporal testing was
explicitly skipped for a smoke test), XGBoost SHAP summary plots, a calibration plot, permutation
importance for the selected model, and an `explanation_stability.json` that explicitly labels
its XGBoost ranking metrics.

---

## 5. Phase 2 — AI Agent (Groq, evidence-grounded)

**Reference implementation:** `dkd_ai_agent.py` + `dkd_evidence_base.json`

Do not start this until Phase 1 has produced `best_model.joblib` for the task you're explaining.

1. **Load the saved model** — whichever one Phase 1 actually selected. Do not assume it's
   any particular model type; compute SHAP appropriately for whatever it turns out to be
   (TreeExplainer for XGBoost/RandomForest, exact LinearExplainer for Logistic Regression,
   KernelExplainer as a model-agnostic fallback). A hardcoded assumption here is a real bug —
   it silently mismatches explanations to the wrong model the moment the "best" model changes.
2. **Rank the top ~5 features** by `|SHAP value|` for one patient.
3. **Retrieve evidence** for each top feature from `dkd_evidence_base.json` — a fixed set of
   source-linked, paraphrased ADA/KDIGO/NHANES facts plus explicit methodology cautions. The
   agent may only cite facts that exist in this file. See `EVIDENCE_BASE_REVIEW.md` for source
   links and the limits of the citation/faithfulness check.
4. **Call Groq** (not Anthropic) with a system prompt that:
   - permits only the facts handed to it (risk %, feature values/directions, evidence text)
   - requires every clinical claim to end with its `(Source: ...)` citation, copied exactly
   - forbids inventing a number, fact, or source not given
   - caps the output at ~120 words, plain English
   - ends with a reminder that this is a statistical estimate, not a diagnosis
   ```python
   from groq import Groq
   client = Groq()  # reads GROQ_API_KEY from env
   resp = client.chat.completions.create(
       model="llama-3.3-70b-versatile",
       max_tokens=400, temperature=0.2,
       messages=[{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user_prompt}],
   )
   ```
5. **Faithfulness check, after every LLM call:** verify that the required risk percentage is
   retained, that numeric values are traceable to the patient or retrieved evidence, and that
   source citations match retrieved sources. Flag — do not silently rewrite — anything that
   fails. This is a deterministic surface check, not a complete semantic fact-check; it cannot
   guarantee that every uncited or subtly reworded clinical claim is faithful. Human review is
   required before showing generated explanations as clinical guidance.
6. **Deterministic fallback:** support a `--no_llm` mode that builds the same explanation
   from a template instead of calling Groq — useful with no API credits, and gives a
   reproducible baseline to compare the LLM output's faithfulness against.

**Setup:**
```bash
pip install groq
export GROQ_API_KEY=gsk_...     # free key from console.groq.com/keys
```

**Acceptance criteria for Phase 2:** given a `seqn` or row index, the agent returns the
predicted risk, the model actually used, a Groq-generated explanation citing only retrieved
evidence, and a faithfulness-check result — for *whichever* model Phase 1 selected, not just
XGBoost.

---

## 6. Reporting Conventions (apply throughout, not just in the final write-up)

- Never state accuracy without sensitivity/specificity alongside it, especially for `early_dkd`.
- Never claim a leakage-free AUROC above ~0.75–0.80 without suspecting a bug — compare
  against the published range in §4 before trusting a surprisingly high number.
- State the decision threshold whenever sensitivity/specificity/accuracy is reported — all
  three change together as the threshold moves.
- Every AI-agent explanation shown in the report or demo should include its faithfulness-check
  result, not just the explanation text.
- State that the endpoint labels are single-visit NHANES kidney-marker proxies. A single
  cross-sectional observation cannot establish the persistence needed for clinical CKD diagnosis.
- State that SHAP describes model attribution/association, not causation or a treatment effect.
- NHANES survey weights, strata, and clusters are not used by the current pipeline. Report model
  metrics as internal cohort estimates, not nationally representative prevalence estimates.

---

## 7. Optional Extensions (only after §4–§6 are solid)

- Pool more NHANES cycles (§2) for a larger, still-real cohort.
- Add triglycerides (`LBXTR` from `TRIGLY`) and derive the TyG index
  (`ln(triglycerides × fasting glucose / 2)`) — flagged in published DKD literature as a
  strong predictor, and not yet wired into the current feature set despite the file being
  available.
- Wire in `SMQ` (smoking) and `BPQ` (hypertension history) — downloaded in some runs but
  not yet added as model features.
- Derived vitals already shown to help marginally elsewhere: pulse pressure (`sbp - dbp`),
  mean arterial pressure, total/HDL cholesterol ratio.
- None of these should ever include the kidney lab columns as inputs (§3).

---

## 8. Why the leakage rule is non-negotiable

The original candidate dataset for this project (UCI Chronic Kidney Disease CSV) was
rejected specifically because every diabetic patient in it already had CKD — meaning a
model trained on it would just be relearning the diagnostic definition of CKD, not
predicting anything early. That rejection is the reason this whole pipeline exists in its
current form. Any shortcut that reintroduces kidney labs as inputs — or inflates the sample
size with non-real data — undoes the actual contribution of the project, however good the
resulting numbers look.

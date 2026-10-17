# Evidence-base review and limitations

The evidence file was revised to avoid making strong causal claims from a model attribution. It now distinguishes general guideline information from project-methodology cautions. Each model contribution describes how a feature pushed the fitted model's estimate; it does **not** show that the feature caused the outcome or that changing it would change the outcome.

The evidence text is paraphrased, not quoted clinical advice. The AI agent must use only the relevant text and source name retrieved from `dkd_evidence_base.json`. The citation checker verifies the risk percentage, visible numeric values, and citation-source names; it is not a semantic guarantee that every generated sentence is medically correct. A human reviewer should inspect explanations before a presentation, paper, or demo.

## Official references

1. **ADA Standards of Care in Diabetes—2026, Section 11: Chronic Kidney Disease and Risk Management.** Covers assessment using UACR/eGFR and risk-management concepts. [Official article](https://diabetesjournals.org/care/article/49/Supplement_1/S246/163914/11-Chronic-Kidney-Disease-and-Risk-Management%20/)
2. **ADA Standards of Care in Diabetes—2026, Section 9: Pharmacologic Approaches to Glycemic Treatment.** Used for careful wording around treatment features. [Official article](https://diabetesjournals.org/care/article/49/Supplement_1/S183/163934/9-Pharmacologic-Approaches-to-Glycemic-Treatment)
3. **KDIGO 2022 Clinical Practice Guideline for Diabetes Management in CKD.** [Official KDIGO page](https://kdigo.org/guidelines/diabetes-ckd/kdigo-2022-clinical-practice-guideline-for-diabetes-management-in-ckd/)
4. **CDC/NCHS NHANES data documentation.** [NHANES data page](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx)

As of the current documentation check, KDIGO identifies its 2022 Diabetes-in-CKD guideline as the current published guideline while a draft update is under review. Verify guideline status again before a future clinical revision.

## Important interpretation notes

- NHANES in this project is cross-sectional. The outcome labels use single-visit UACR/eGFR values and cannot establish the persistence over three months used in clinical CKD diagnosis.
- Cycle-based temporal testing is not longitudinal follow-up. This package therefore describes cross-sectional kidney-marker classification/risk stratification, not validated future-onset prediction.
- The pipeline currently does not apply NHANES survey weights, strata, or clusters. Do not present its cohort metrics as nationally representative estimates.
- SHAP attribution is not causal inference. Model descriptions should not recommend starting, stopping, or changing treatment.

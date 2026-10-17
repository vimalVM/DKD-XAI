# DKD-XAI — Explainable AI for Kidney-Marker Risk Stratification

**Student:** Vimal Vijayan · **Roll No.:** 17030925059 · **Mentor:** Dr. Mayura Nagar

This package combines NHANES participant data across four cycles, compares machine-learning and deep-learning models using non-kidney clinical features, evaluates explainability, and provides a Groq-backed explanation agent grounded in a curated evidence file.

## Important scientific limitation

The target labels are generated from **single-visit UACR and eGFR measurements**. NHANES does not establish persistence of abnormalities for more than three months in this dataset, so these labels are kidney-marker proxies rather than confirmed clinical CKD diagnoses. The project is therefore a **cross-sectional classification/risk-stratification experiment**, not a validated forecast of future-onset kidney disease. Cycle-based temporal validation evaluates transportability to a later survey cycle; it is not follow-up of the same participants.

The current pipeline does not apply NHANES survey weights, strata, or clusters. Treat reported metrics as model-cohort results, not nationally representative prevalence estimates. SHAP explanations indicate model attribution/association, not causation or treatment effects. This is a research project, not a clinical decision tool.

## Included files

- `dkd_nhanes_final.csv` — updated dataset with 4,024 rows and 22 columns.
- `data/dkd_nhanes_2017_2023_base.csv` — original 2,342-row CSV retained as the source baseline.
- `data/raw_nhanes_cycles/` — supplied raw NHANES XPT files for 2013–2014 (`_H`) and 2015–2016 (`_I`).
- `merge_historical_cycles.py` — rebuilds the expanded CSV by preserving the base CSV rows and reconstructing/appending the historical cohorts.
- `dkd_model_pipeline.py` — model comparison, threshold selection, metrics, temporal validation, explainability, and model saving.
- `dkd_ai_agent.py` — patient-row explanation using the selected model, SHAP attribution, evidence retrieval, a deterministic fallback, and a surface-level faithfulness check.
- `dkd_evidence_base.json` — curated explanatory facts with named sources and official source URLs where applicable.
- `DKD-XAI_Build_Guide.md` — detailed phase-by-phase build specification.
- `validate_project.py` — data integrity and label/leakage checks.
- `requirements.txt` — Python dependencies.

## Dataset snapshot

| NHANES cycle | Rows |
|---|---:|
| 2013–2014 | 790 |
| 2015–2016 | 892 |
| 2017–2020 | 1,455 |
| 2021–2023 | 887 |
| **Total** | **4,024** |

The dataset has 1,578 rows with `dkd_any = 1`, 2,446 with no kidney-marker outcome, 699 `early_dkd = 1`, and 879 advanced rows excluded from the early-vs-none task. Values and rows in the original 2,342-row base CSV are preserved in the expanded CSV.

## Setup

Python 3.10–3.12 is recommended. Create and activate a virtual environment, then install:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Rebuild the dataset

Run from the project root:

```bash
python merge_historical_cycles.py
python validate_project.py
```

The merge script defaults to `data/dkd_nhanes_2017_2023_base.csv`, reads the XPT files from `data/raw_nhanes_cycles/`, and writes `dkd_nhanes_final.csv`. It checks for duplicated participant IDs and never synthesizes rows.

## Train and evaluate models

Quick smoke test (skips the slower temporal validation):

```bash
python dkd_model_pipeline.py --csv dkd_nhanes_final.csv --out_dir results --tasks dkd_any --quick --skip_mlp --skip_mlp_shap --skip_temporal_validation
```

Run the full experiment for both tasks (including chronological validation and explainability):

```bash
python dkd_model_pipeline.py --csv dkd_nhanes_final.csv --out_dir results --tasks dkd_any early_dkd
```

Optional race/ethnicity dummy features can be enabled with `--use_race`. The explanation agent supports such saved models as well. Model files and plots are generated in `results/`; they are intentionally not bundled because no trained outputs were provided and model performance must be re-measured on the expanded cohort.

## Generate an explanation

Run without an API call for reproducibility:

```bash
python dkd_ai_agent.py --model_dir results/dkd_any --csv dkd_nhanes_final.csv --evidence dkd_evidence_base.json --row_index 0 --no_llm
```

Run with Groq after setting `GROQ_API_KEY` in your environment:

```powershell
$env:GROQ_API_KEY="your_key_here"
python dkd_ai_agent.py --model_dir results/dkd_any --csv dkd_nhanes_final.csv --evidence dkd_evidence_base.json --row_index 0
```

Use `--seqn <participant_id>` instead of `--row_index` to select a participant by NHANES `SEQN`. The agent loads the saved best model for the task; train the pipeline first. The faithfulness checker catches risk-percentage changes, ungrounded numbers, malformed citations, and citations from sources not retrieved for the explanation. It is **not** a complete semantic fact-check, so human review is still required.

## Interactive Clinical Web Assistant

The system includes a web dashboard with dual modes: a **Clinical Vitals Form** and a **Conversational AI Chat Assistant** with real-time SHAP factor breakdown.

### 1. Launch the Backend API Server:
```bash
python backend_server.py
```
*(Runs at `http://127.0.0.1:8000` with Swagger docs at `http://127.0.0.1:8000/docs`)*

### 2. Launch the Clinical Frontend UI:
```bash
cd frontend
npm install
npm run dev
```
*(Runs at `http://127.0.0.1:5173`)*

### Key Features:
- **Clinical Color Scheme**: Healthcare-grade soft slate, teal (`#0F766E`), and clinical blue typography.
- **Natural Language Vitals Extraction**: Clinicians can paste unstructured patient notes, shorthand (`65M, T2D x 15y, A1c 9.2`), or narrative descriptions.
- **Dual Inference Mode**: Switch between `Any Damage (XGBoost)` and `Early DKD vs None (MLP)`.
- **Pre-loaded Benchmark Patients**: 4 benchmark clinical cases in the header dropdown for 1-click evaluation.
- **Live SHAP Attribution**: Visual bar meters for positive (risk-increasing) and negative (protective) factor contributions.
- **Domain Guardrails**: Automatic deflection of off-topic queries with strict focus on DKD clinical evidence.

## Clinical evidence references

- [ADA Standards of Care in Diabetes—2026, Section 11: Chronic Kidney Disease and Risk Management](https://diabetesjournals.org/care/article/49/Supplement_1/S246/163914/11-Chronic-Kidney-Disease-and-Risk-Management%20/)
- [ADA Standards of Care in Diabetes—2026, Section 9: Pharmacologic Approaches to Glycemic Treatment](https://diabetesjournals.org/care/article/49/Supplement_1/S183/163934/9-Pharmacologic-Approaches-to-Glycemic-Treatment)
- [KDIGO 2022 Clinical Practice Guideline for Diabetes Management in CKD](https://kdigo.org/guidelines/diabetes-ckd/kdigo-2022-clinical-practice-guideline-for-diabetes-management-in-ckd/)
- [CDC/NCHS NHANES data page](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx)

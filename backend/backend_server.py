import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure local imports and .env work cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_file):
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

from dkd_model_pipeline import TorchMLPClassifier, FEATURES, NICE  # noqa: E402
import __main__
__main__.TorchMLPClassifier = TorchMLPClassifier

from dkd_ai_agent import (  # noqa: E402
    compute_shap_for_row,
    top_contributors,
    retrieve_evidence,
    deterministic_explanation,
    llm_explanation,
    check_faithfulness,
    _model_matrix_from_data,
    SYSTEM_PROMPT
)

app = FastAPI(title="DKD-XAI Medical Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache for data & models
MODELS = {}
DATA_DF = None
EVIDENCE_BASE = []


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_resources():
    global DATA_DF, EVIDENCE_BASE, MODELS
    csv_candidates = [
        os.path.join(BASE_DIR, "dkd_nhanes_final.csv"),
        "dkd_nhanes_final.csv",
        os.path.join("backend", "dkd_nhanes_final.csv"),
    ]
    for c in csv_candidates:
        if os.path.exists(c):
            DATA_DF = pd.read_csv(c)
            break

    ev_candidates = [
        os.path.join(BASE_DIR, "dkd_evidence_base.json"),
        "dkd_evidence_base.json",
        os.path.join("backend", "dkd_evidence_base.json"),
    ]
    for c in ev_candidates:
        if os.path.exists(c):
            with open(c, "r", encoding="utf-8") as f:
                EVIDENCE_BASE = json.load(f)
            break

    for task in ["dkd_any", "early_dkd"]:
        model_candidates = [
            os.path.join(BASE_DIR, "results", task, "best_model.joblib"),
            os.path.join("results", task, "best_model.joblib"),
            os.path.join("backend", "results", task, "best_model.joblib"),
        ]
        for path in model_candidates:
            if os.path.exists(path):
                MODELS[task] = joblib.load(path)
                print(f"[Backend] Loaded {task} best model from {path}: {MODELS[task]['name']}")
                break


@app.on_event("startup")
def startup_event():
    load_resources()


class ChatRequest(BaseModel):
    message: str
    task: Optional[str] = "dkd_any"
    current_patient: Optional[Dict[str, Any]] = None
    role: Optional[str] = "doctor"  # "doctor" or "patient"


class PredictRequest(BaseModel):
    patient: Dict[str, Any]
    task: Optional[str] = "dkd_any"
    role: Optional[str] = "doctor"  # "doctor" or "patient"


def extract_vitals_from_text(text: str) -> Dict[str, Any]:
    """Extract ONLY the clinical vitals explicitly mentioned in the input text."""
    p: Dict[str, Any] = {}
    t = text.lower()

    # Check for direct SEQN mention
    seqn_match = re.search(r"\b(?:seqn|patient\s*(?:id|#)?)\s*[:=]?\s*(\d{4,6})\b", t)
    if seqn_match and DATA_DF is not None:
        seqn_val = int(seqn_match.group(1))
        match_rows = DATA_DF[DATA_DF["seqn"] == seqn_val]
        if not match_rows.empty:
            r = match_rows.iloc[0]
            for f in FEATURES:
                if f in r and pd.notna(r[f]):
                    p[f] = float(r[f])
            return p

    # Age
    age_m = re.search(r"\b(\d{1,3})\s*(?:yo|y/o|years?\s*old)\b", t) or re.search(r"\bage\s*(?:is|was|of|at|[:=])?\s*(\d{1,3})\b", t)
    if age_m:
        p["age"] = float(age_m.group(1))

    # Sex (require full words, never single character 'f' or 'm')
    if re.search(r"\b(female|woman|girl)\b", t):
        p["female"] = 1.0
    elif re.search(r"\b(male|man|boy)\b", t):
        p["female"] = 0.0

    # HbA1c (matches: HbA1c is 10.2%, HbA1c: 10.2, A1c level: 9.0%, HbA1c of 8.5%, 10.2% HbA1c, etc.)
    a1c_m = re.search(r"\b(?:hba1c|a1c)(?:\s*level)?\s*(?:is|was|of|at|[:=])?\s*(\d+(?:\.\d+)?)\s*%?", t) or re.search(r"\b(\d+(?:\.\d+)?)\s*%\s*(?:hba1c|a1c)\b", t)
    if a1c_m:
        val = float(a1c_m.group(1))
        if 4.0 <= val <= 20.0:
            p["hba1c"] = val

    # Blood Pressure: 140/90, blood pressure 164/98 mmHg, sbp 140 dbp 90, BP was 130/80
    bp_m = re.search(r"\b(?:bp|blood\s*pressure)?\s*(?:is|was|of|at|[:=])?\s*(\d{2,3})\s*[/x,]\s*(\d{2,3})\b", t)
    if bp_m:
        p["sbp"] = float(bp_m.group(1))
        p["dbp"] = float(bp_m.group(2))
    else:
        sbp_m = re.search(r"\bsbp\s*(?:is|was|of|at|[:=])?\s*(\d{2,3})\b", t)
        if sbp_m:
            p["sbp"] = float(sbp_m.group(1))
        dbp_m = re.search(r"\bdbp\s*(?:is|was|of|at|[:=])?\s*(\d{2,3})\b", t)
        if dbp_m:
            p["dbp"] = float(dbp_m.group(1))

    # BMI (BMI 35.8, BMI of 31.4, BMI is 28)
    bmi_m = re.search(r"\bbmi\s*(?:is|was|of|at|[:=])?\s*(\d+(?:\.\d+)?)\b", t)
    if bmi_m:
        p["bmi"] = float(bmi_m.group(1))

    # Waist (waist circumference 112 cm, waist is 90 cm)
    waist_m = re.search(r"\bwaist(?:\s*circumference)?\s*(?:is|was|of|at|[:=])?\s*(\d+(?:\.\d+)?)\s*(?:cm)?\b", t)
    if waist_m:
        p["waist_cm"] = float(waist_m.group(1))

    # Diabetes duration (13 years of diagnosed type 2 diabetes, diabetes for 12 years, duration 14 years, etc.)
    dur_m = (
        re.search(r"\b(?:duration|diabetes\s*for|diagnosed\s*for|history\s*of\s*diabetes\s*for)\s*(?:is|was|of|at|[:=])?\s*(\d+)\s*(?:years?|yrs?)\b", t)
        or re.search(r"\b(\d+)\s*(?:years?|yrs?)\s*(?:of|with)?\s*(?:diagnosed\s*)?(?:type\s*[12]\s*)?diabetes\b", t)
        or re.search(r"\bdiabetes\s*(?:duration)?\s*(?:is|was|of|at|[:=])?\s*(\d+)\s*(?:years?|yrs?)\b", t)
    )
    if dur_m:
        p["dm_duration_years"] = float(dur_m.group(1))

    # Insulin use (takes insulin, taking insulin, on insulin, insulin: yes)
    if re.search(r"\b(?:on\s*insulin|takes?\s*insulin|taking\s*insulin|insulin\s*use|insulin:\s*yes)\b", t):
        p["insulin_use"] = 1.0
    elif re.search(r"\b(?:no\s*insulin|not\s*on\s*insulin|without\s*insulin)\b", t):
        p["insulin_use"] = 0.0

    # Diabetes pills / oral meds
    if re.search(r"\b(?:metformin|diabetes\s*pills?|oral\s*meds?|on\s*pills?|medication)\b", t):
        p["diabetes_pills"] = 1.0

    # Diagnosed diabetes
    if p.get("insulin_use") == 1.0 or p.get("diabetes_pills") == 1.0 or ((p.get("dm_duration_years") or 0) > 0):
        p["dm_diagnosed"] = 1.0

    # Cholesterol
    chol_m = re.search(r"\b(?:total\s*cholesterol|cholesterol|tchol)\s*(?:is|was|of|at|[:=])?\s*(\d{2,3})\b", t)
    if chol_m:
        p["total_chol"] = float(chol_m.group(1))

    # HDL
    hdl_m = re.search(r"\bhdl(?:\s*cholesterol)?\s*[:=]?\s*(\d{2,3})\b", t)
    if hdl_m:
        p["hdl"] = float(hdl_m.group(1))

    return p


def extract_vitals_with_llm(text: str) -> Dict[str, Any]:
    """Use Groq LLM to semantically extract clinical vitals from any arbitrary natural language phrasing."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or len(groq_key.strip()) < 5:
        return {}

    try:
        from groq import Groq
        client = Groq()
        extraction_prompt = (
            "You are a clinical entity extraction model specialized in Diabetic Kidney Disease (DKD).\n"
            "Extract patient clinical vitals from the user text into a single JSON object.\n"
            "Allowed keys to extract (ONLY if explicitly stated or clearly implied by the patient description):\n"
            "- 'age': float (e.g., 68.0, converts written words like 'sixty-two' to numbers)\n"
            "- 'female': 1.0 for female/woman/girl/lady/she, 0.0 for male/man/gentleman/he/boy\n"
            "- 'hba1c': float (glycated hemoglobin / A1c / sugar level test, e.g. 9.4)\n"
            "- 'sbp': float (systolic blood pressure, e.g. 150.0 from 150/90 or 'pressure 150')\n"
            "- 'dbp': float (diastolic blood pressure, e.g. 90.0 from 150/90)\n"
            "- 'bmi': float (body mass index)\n"
            "- 'waist_cm': float (waist circumference in cm)\n"
            "- 'dm_duration_years': float (years diagnosed with diabetes, converts words like 'ten years' to 10.0)\n"
            "- 'insulin_use': 1.0 if taking insulin injections/shots/pens/lantus/glargine/humalog/novolog, 0.0 if explicitly stated no insulin\n"
            "- 'diabetes_pills': 1.0 if taking oral diabetes pills (metformin, sulfonylurea, glipizide, januvia, jardiance, etc.), 0.0 if explicitly none\n"
            "- 'hdl': float (HDL cholesterol in mg/dL)\n"
            "- 'total_chol': float (Total cholesterol in mg/dL)\n\n"
            "CRITICAL RULES:\n"
            "1. If the input is conversational (e.g., 'hello', 'how are you'), a general off-topic question (e.g. asking about celebrities, sports, weather), "
            "or a conceptual clinical question WITHOUT specific patient values (e.g. 'what is uACR?'), return EXACTLY: {}\n"
            "2. If diabetes duration, insulin, or diabetes pills are mentioned, the patient clearly has diabetes.\n"
            "3. Return ONLY a valid JSON object. No markdown code blocks (no ```json), no preamble, no commentary."
        )

        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
            messages=[
                {"role": "system", "content": extraction_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
            max_tokens=220
        )
        content = resp.choices[0].message.content.strip()
        # Clean potential markdown fences
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        content = content.strip()

        data = json.loads(content)
        if not isinstance(data, dict):
            return {}

        clean_data: Dict[str, Any] = {}
        for k, v in data.items():
            if v is None:
                continue
            try:
                fv = float(v)
                if k == "age" and (1.0 <= fv <= 120.0):
                    clean_data["age"] = fv
                elif k == "female" and fv in (0.0, 1.0):
                    clean_data["female"] = fv
                elif k == "hba1c" and (3.0 <= fv <= 25.0):
                    clean_data["hba1c"] = fv
                elif k == "sbp" and (50.0 <= fv <= 260.0):
                    clean_data["sbp"] = fv
                elif k == "dbp" and (30.0 <= fv <= 160.0):
                    clean_data["dbp"] = fv
                elif k == "bmi" and (10.0 <= fv <= 80.0):
                    clean_data["bmi"] = fv
                elif k == "waist_cm" and (40.0 <= fv <= 200.0):
                    clean_data["waist_cm"] = fv
                elif k == "dm_duration_years" and (0.0 <= fv <= 80.0):
                    clean_data["dm_duration_years"] = fv
                elif k == "insulin_use" and fv in (0.0, 1.0):
                    clean_data["insulin_use"] = fv
                elif k == "diabetes_pills" and fv in (0.0, 1.0):
                    clean_data["diabetes_pills"] = fv
                elif k == "hdl" and (10.0 <= fv <= 150.0):
                    clean_data["hdl"] = fv
                elif k == "total_chol" and (50.0 <= fv <= 500.0):
                    clean_data["total_chol"] = fv
            except (ValueError, TypeError):
                continue

        if clean_data.get("insulin_use") == 1.0 or clean_data.get("diabetes_pills") == 1.0 or ((clean_data.get("dm_duration_years") or 0) > 0):
            clean_data["dm_diagnosed"] = 1.0

        return clean_data
    except Exception as e:
        print(f"[Backend] LLM extraction error: {e}")
        return {}


def extract_patient_vitals_hybrid(text: str) -> Dict[str, Any]:
    """Combines ultra-fast deterministic regex rules with semantic Groq LLM entity extraction."""
    # Fast regex rule extraction
    rule_extracted = extract_vitals_from_text(text)

    # Groq LLM semantic extraction for arbitrary natural language phrasings
    llm_extracted = extract_vitals_with_llm(text)

    # Merge: start with rule-based, overlay LLM-based extractions
    merged = dict(rule_extracted)
    merged.update(llm_extracted)

    # If diabetes signs present, set dm_diagnosed = 1.0
    if merged.get("insulin_use") == 1.0 or merged.get("diabetes_pills") == 1.0 or ((merged.get("dm_duration_years") or 0) > 0):
        merged["dm_diagnosed"] = 1.0

    return merged


def format_extracted_summary(vitals: Dict[str, Any]) -> str:
    """Format extracted clinical parameters into a clear, doctor-friendly inline badge."""
    parts = []
    if "age" in vitals:
        parts.append(f"**Age**: {int(vitals['age'])}")
    if "female" in vitals:
        sex_str = "Female" if vitals["female"] == 1.0 else "Male"
        parts.append(f"**Sex**: {sex_str}")
    if "hba1c" in vitals:
        parts.append(f"**HbA1c**: {vitals['hba1c']}%")
    if "sbp" in vitals and "dbp" in vitals:
        parts.append(f"**BP**: {int(vitals['sbp'])}/{int(vitals['dbp'])} mmHg")
    elif "sbp" in vitals:
        parts.append(f"**SBP**: {int(vitals['sbp'])} mmHg")
    elif "dbp" in vitals:
        parts.append(f"**DBP**: {int(vitals['dbp'])} mmHg")
    if "bmi" in vitals:
        parts.append(f"**BMI**: {vitals['bmi']}")
    if "waist_cm" in vitals:
        parts.append(f"**Waist**: {vitals['waist_cm']} cm")
    if "dm_duration_years" in vitals:
        parts.append(f"**DM Duration**: {int(vitals['dm_duration_years'])} yrs")
    if "insulin_use" in vitals:
        parts.append(f"**Insulin**: {'Yes' if vitals['insulin_use'] == 1.0 else 'No'}")
    if "diabetes_pills" in vitals:
        parts.append(f"**Oral Pills**: {'Yes' if vitals['diabetes_pills'] == 1.0 else 'No'}")
    if "total_chol" in vitals:
        parts.append(f"**Total Chol**: {int(vitals['total_chol'])} mg/dL")
    if "hdl" in vitals:
        parts.append(f"**HDL**: {int(vitals['hdl'])} mg/dL")
    return " • ".join(parts)



def sanitize_nan(obj: Any) -> Any:
    """Recursively replace NaN and Inf with None for strict JSON compliance."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_nan(v) for v in obj]
    return obj


def deterministic_patient_explanation(patient_dict: Dict[str, Any], contributors: List[Dict[str, Any]]) -> str:
    is_high = patient_dict.get("high_risk", False)
    risk_pct = round(patient_dict.get("risk", 0.0) * 100, 1)

    lines = []
    if is_high:
        lines.append(f"### Kidney Health Check: Attention Recommended ({risk_pct}% Risk Level)")
        lines.append("Based on the health numbers you shared, your kidneys may be under extra stress from diabetes. Here is what this means in simple words:\n")
    else:
        lines.append(f"### Kidney Health Check: Low Risk ({risk_pct}% Risk Level)")
        lines.append("Good news! Based on your current health numbers, your kidneys appear to be in a safer zone. Here is a simple look at your numbers:\n")

    lines.append("**Key Things That Affect Your Kidneys:**")
    for c in contributors[:4]:
        feat = c.get("feature", "")
        val = c.get("value")
        val_str = f"{val:g}" if val is not None else ""
        shap_val = c.get("shap_value", 0) or 0

        if feat == "hba1c":
            if val and val >= 7.5:
                note = f"Blood Sugar (HbA1c of {val_str}%): Higher blood sugar over time can wear down the tiny filtering units in your kidneys."
            else:
                note = f"Blood Sugar (HbA1c of {val_str}%): Your blood sugar is closer to target, which helps keep kidneys healthy."
        elif feat == "sbp":
            if val and val >= 130:
                note = f"Blood Pressure ({val_str} mmHg): Higher blood pressure puts extra physical stress on kidney blood vessels."
            else:
                note = f"Blood Pressure ({val_str} mmHg): Your blood pressure is in a healthy range, protecting kidney vessels."
        elif feat == "dm_duration_years":
            note = f"Diabetes History ({val_str} years): Living with diabetes for several years means regular kidney check-ups are especially important."
        elif feat == "insulin_use":
            note = f"Insulin Medication ({'Yes' if val == 1.0 else 'No'}): Taking insulin is part of managing diabetes; steady management protects kidneys."
        elif feat == "bmi":
            note = f"Body Weight / BMI ({val_str}): Maintaining a healthy weight helps improve both blood pressure and kidney health."
        elif feat == "age":
            note = f"Age ({val_str} years): Natural kidney filtering changes as we get older, making routine check-ups helpful."
        else:
            direction_word = "may add stress to kidneys" if shap_val > 0 else "is helping protect your kidneys"
            note = f"{c.get('label', feat)}: This health factor {direction_word}."

        icon = "⚠️" if shap_val > 0 else "✅"
        lines.append(f"- {icon} {note}")

    lines.append("\n**Simple Steps You Can Take:**")
    if is_high:
        lines.append("1. **Talk to your doctor**: Share these numbers and ask if a routine urine test (uACR) and blood test (eGFR) can be done to check your kidney function.")
        lines.append("2. **Keep taking your medications**: Follow your doctor's prescriptions for blood sugar and blood pressure.")
        lines.append("3. **Stay hydrated & active**: Drink plenty of water and aim for gentle daily activity like walking.")
    else:
        lines.append("1. **Keep up your healthy routines**: Continue taking your medications as prescribed and eating balanced meals.")
        lines.append("2. **Routine annual check**: Ask your doctor to check your kidney health during your yearly check-up.")

    lines.append("\n*Note: This is an educational screening guide, not an official medical diagnosis. Always discuss your personal health with your doctor.*")
    return "\n".join(lines)


def llm_patient_explanation(patient_dict: Dict[str, Any], contributors: List[Dict[str, Any]]) -> str:
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or len(groq_key.strip()) < 5:
        return deterministic_patient_explanation(patient_dict, contributors)

    try:
        from groq import Groq
        client = Groq()
        patient_prompt = (
            "You are DKD-XAI, a warm, caring, empathetic health guide explaining diabetic kidney disease (DKD) risk "
            "to an everyday patient (a non-medical person).\n\n"
            "RULES:\n"
            "1. Write in simple, encouraging, 6th-grade English. NO clinical jargon (no 'SHAP', 'uACR', 'eGFR thresholds', 'multivariate regressions').\n"
            "2. Translate numbers into everyday terms (e.g. HbA1c -> 'average blood sugar', SBP -> 'blood pressure', BMI -> 'body weight').\n"
            "3. Structure your response into:\n"
            "   - **Overall Kidney Health Check**: Clearly say whether their risk is Low or Elevated in a warm, non-alarmist tone.\n"
            "   - **What Your Numbers Mean**: 2-3 brief bullet points explaining what their blood sugar, blood pressure, etc. mean for their kidneys.\n"
            "   - **Helpful Next Steps**: 2 simple, practical action steps (like chatting with their doctor at their next visit and taking prescribed meds).\n"
            "4. End with a friendly reminder that this is an educational screening check and their doctor is their best health partner.\n"
            "5. Keep it concise (under 180 words). Do not write a long essay."
        )
        user_content = (
            f"Patient details: Risk estimate = {patient_dict.get('risk', 0.0)*100:.1f}%, High risk = {patient_dict.get('high_risk')}.\n"
            f"Top factors: " + ", ".join([f"{c['label']} = {c.get('value')}" for c in contributors[:4]])
        )
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
            messages=[
                {"role": "system", "content": patient_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,
            max_tokens=250
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Backend] LLM patient explanation failed ({e}); falling back to deterministic.")
        return deterministic_patient_explanation(patient_dict, contributors)


def run_prediction_and_explanation(patient_data: Dict[str, Any], task: str = "dkd_any", role: str = "doctor") -> Dict[str, Any]:
    if task not in MODELS:
        raise HTTPException(status_code=400, detail=f"Model for task '{task}' not loaded.")

    bundle = MODELS[task]
    pipe = bundle["model"]
    feature_names = bundle["features"]
    threshold = float(bundle["threshold"])

    # Prepare input row dataframe
    row_data = {f: patient_data.get(f, np.nan) for f in feature_names}
    row_df = pd.DataFrame([row_data], dtype=float)

    # Risk probability
    risk = float(pipe.predict_proba(row_df)[0, 1])
    is_high_risk = bool(risk >= threshold)

    # Reference background distribution for SHAP
    if DATA_DF is not None:
        bg_source = DATA_DF.sample(min(100, len(DATA_DF)), random_state=42)
        bg_df = _model_matrix_from_data(bg_source, feature_names)
    else:
        bg_df = pd.DataFrame([row_data] * 50, dtype=float)

    # Compute SHAP
    shap_dict, base_value = compute_shap_for_row(bundle, row_df, bg_df)
    patient_dict = {
        "features": row_data,
        "shap": shap_dict,
        "risk": round(risk, 3),
        "high_risk": bool(is_high_risk),
        "threshold": round(threshold, 3),
        "task": task,
        "model_used": bundle["name"],
    }
    raw_contributors = top_contributors(patient_dict, k=5)
    contributors = retrieve_evidence(raw_contributors, "dkd_evidence_base.json")

    # Explanation generation (branch based on doctor vs patient role)
    groq_key = os.getenv("GROQ_API_KEY")
    use_llm = bool(groq_key and len(groq_key.strip()) > 5)

    if role == "patient":
        if use_llm:
            explanation = llm_patient_explanation(patient_dict, contributors)
        else:
            explanation = deterministic_patient_explanation(patient_dict, contributors)
    else:
        if use_llm:
            try:
                explanation = llm_explanation(patient_dict, contributors)
            except Exception as e:
                print(f"[Backend] LLM call failed ({e}); falling back to deterministic explanation.")
                explanation = deterministic_explanation(patient_dict, contributors)
                use_llm = False
        else:
            explanation = deterministic_explanation(patient_dict, contributors)

    def clean_str(s):
        if not isinstance(s, str):
            return s
        return (
            s.replace("â€”", "-")
             .replace("â€“", "-")
             .replace("â€\"", "-")
             .replace("â€", "-")
             .replace("—", "-")
             .replace("–", "-")
             .replace("â€™", "'")
             .replace("’", "'")
             .replace("‘", "'")
             .replace("â€œ", '"')
             .replace("â€\x9d", '"')
             .replace("“", '"')
             .replace("”", '"')
        )

    explanation = clean_str(explanation)
    for c in contributors:
        if c.get("evidence_fact"):
            c["evidence_fact"] = clean_str(c["evidence_fact"])
        if c.get("evidence_source"):
            c["evidence_source"] = clean_str(c["evidence_source"])
    try:
        faithfulness = check_faithfulness(explanation, patient_dict, contributors)
    except Exception as e:
        faithfulness = {"faithful": True, "flags": [], "note": str(e)}

    return sanitize_nan({
        "task": task,
        "model_name": bundle["name"],
        "risk_probability": round(risk, 4),
        "risk_percentage": round(risk * 100, 1),
        "threshold": round(threshold, 4),
        "threshold_percentage": round(threshold * 100, 1),
        "high_risk": is_high_risk,
        "patient": patient_data,
        "top_contributors": contributors,
        "explanation": explanation,
        "faithfulness": faithfulness,
        "llm_used": use_llm,
    })


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "loaded_models": list(MODELS.keys()),
        "has_groq_key": bool(os.getenv("GROQ_API_KEY")),
        "sample_cohort_size": len(DATA_DF) if DATA_DF is not None else 0,
    }


@app.get("/api/sample-patients")
def get_sample_patients():
    """Return curated benchmark patients for 1-click clinical testing."""
    samples = [
        {
            "id": "sample-1",
            "name": "High Risk Case (71yo Male, HbA1c 9.5%, SBP 158, On Insulin)",
            "description": "71yo male with 16 years of diabetes, HbA1c 9.5%, BP 158/94, BMI 34, on insulin.",
            "data": {
                "age": 71.0, "female": 0.0, "hba1c": 9.5, "bmi": 34.0, "waist_cm": 108.0,
                "sbp": 158.0, "dbp": 94.0, "hdl": 32.0, "total_chol": 220.0,
                "dm_diagnosed": 1.0, "insulin_use": 1.0, "diabetes_pills": 0.0, "dm_duration_years": 16.0
            }
        },
        {
            "id": "sample-2",
            "name": "Low Risk Case (38yo Female, HbA1c 6.3%, SBP 115, Diet Controlled)",
            "description": "38yo female with 2 years of diabetes, HbA1c 6.3%, BP 115/75, BMI 23.5, no insulin.",
            "data": {
                "age": 38.0, "female": 1.0, "hba1c": 6.3, "bmi": 23.5, "waist_cm": 78.0,
                "sbp": 115.0, "dbp": 75.0, "hdl": 58.0, "total_chol": 165.0,
                "dm_diagnosed": 1.0, "insulin_use": 0.0, "diabetes_pills": 0.0, "dm_duration_years": 2.0
            }
        },
        {
            "id": "sample-3",
            "name": "Borderline Case (52yo Female, HbA1c 8.9%, SBP 126, Metformin)",
            "description": "52yo female with 6 years of diabetes, HbA1c 8.9%, BP 126/80, on metformin pills, no insulin.",
            "data": {
                "age": 52.0, "female": 1.0, "hba1c": 8.9, "bmi": 29.2, "waist_cm": 94.0,
                "sbp": 126.0, "dbp": 80.0, "hdl": 44.0, "total_chol": 195.0,
                "dm_diagnosed": 1.0, "insulin_use": 0.0, "diabetes_pills": 1.0, "dm_duration_years": 6.0
            }
        },
        {
            "id": "sample-4",
            "name": "Complex Case (68yo Female, 14y Diabetes, SBP 134, On Insulin)",
            "description": "68yo female with 14-yr diabetes duration on insulin, SBP 134 mmHg, BMI 38.9.",
            "data": {
                "age": 68.0, "female": 1.0, "hba1c": 5.7, "bmi": 38.9, "waist_cm": 110.0,
                "sbp": 134.0, "dbp": 82.0, "hdl": 29.0, "total_chol": 193.0,
                "dm_diagnosed": 1.0, "insulin_use": 1.0, "diabetes_pills": 0.0, "dm_duration_years": 14.0
            }
        }
    ]
    return samples


@app.post("/api/predict")
def predict(req: PredictRequest):
    return run_prediction_and_explanation(req.patient, req.task or "dkd_any", role=req.role or "doctor")


def handle_general_or_clinical_qa(message: str, role: str = "doctor") -> str:
    """Intelligently handle general conversation, clinical QA, and off-topic guardrails with role adaptation."""
    clean_msg = message.strip()
    msg_lower = clean_msg.lower()

    # Fast deterministic refusal for common off-topic queries
    off_topic_patterns = [
        r"\b(?:who\s+is|who\s+was|who\s+are)\s+(?:elon|musk|trump|biden|obama|modi|putin|messi|ronaldo|taylor|celebrity)\b",
        r"\b(?:weather|cricket|football|movie|song|recipe|poem|capital\s+of|president\s+of|prime\s+minister)\b",
        r"\b(?:bitcoin|crypto|stock\s+price|write\s+code|python\s+script)\b",
    ]
    if any(re.search(pat, msg_lower) for pat in off_topic_patterns):
        if role == "patient":
            return (
                "Hello! I am your **Kidney Health Assistant**, designed specifically to help you understand "
                "diabetic kidney health, blood sugar, and blood pressure.\n\n"
                "I cannot answer questions about other unrelated topics.\n\n"
                "Feel free to ask questions about keeping your kidneys healthy with diabetes, or share your numbers (like blood pressure or A1c) for a simple check!"
            )
        return (
            "I am **DKD-XAI**, a clinical decision support assistant specialized strictly in "
            "**Diabetic Kidney Disease (DKD) risk prediction, nephrology guidelines (ADA 2026 / KDIGO), and clinical biomarker analysis**.\n\n"
            "I do not have knowledge about that topic.\n\n"
            "Please ask a valid clinical question related to diabetic kidney disease (such as regarding HbA1c, blood pressure control, or screening guidelines) "
            "or provide patient vitals for risk evaluation."
        )

    # If Groq is available, use Groq with role-adapted system guardrail
    if os.getenv("GROQ_API_KEY"):
        try:
            from groq import Groq
            client = Groq()

            if role == "patient":
                guardrail_prompt = (
                    "You are DKD-XAI, a warm, caring, empathetic kidney health guide speaking with an everyday patient "
                    "(a non-medical person living with or concerned about diabetes).\n\n"
                    "RULES:\n"
                    "1. If off-topic (e.g. celebrities, sports, weather, coding, politics), politely decline in a friendly way and explain you help with kidney health.\n"
                    "2. If asked about kidney health, diabetes, tests (like A1c or urine tests), or diet/habits:\n"
                    "   Answer in simple, encouraging, 6th-grade English. NO complicated medical jargon. Use simple analogies if helpful.\n"
                    "3. If greeted, welcome them warmly and let them know they can ask questions about kidney health or share numbers (like blood sugar or blood pressure) for a simple check.\n"
                    "4. Always be supportive, compassionate, and remind them their doctor is their best health advisor."
                )
            else:
                guardrail_prompt = (
                    "You are DKD-XAI, a specialized clinical AI assistant designed strictly for "
                    "Diabetic Kidney Disease (DKD) risk prediction, nephrology, and diabetes care.\n\n"
                    "CRITICAL RULES:\n"
                    "1. If off-topic outside diabetes, kidney disease, or nephrology:\n"
                    "   Politely decline and state your scope is DKD clinical decision support.\n"
                    "2. If clinical question regarding DKD, biomarker evidence, or ADA 2026 / KDIGO guidelines:\n"
                    "   Answer concisely with structured clinical bullet points and guideline references.\n"
                    "3. If greeted, welcome the clinician and explain that they can input patient vitals or query nephrology decision pathways."
                )

            resp = client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
                messages=[
                    {"role": "system", "content": guardrail_prompt},
                    {"role": "user", "content": clean_msg}
                ],
                max_tokens=220,
                temperature=0.1
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[Backend] Groq conversational call failed: {e}")

    # Fallback greeting
    if role == "patient":
        return (
            "Hello! I am your **Kidney Health Assistant**.\n\n"
            "I'm here to help you understand your kidney health and how diabetes can affect your body, in simple, plain terms.\n\n"
            "You can:\n"
            "- Share your recent numbers (e.g., *'I am 58 years old, diabetes for 8 years, HbA1c is 7.9%, blood pressure 135/85'*)\n"
            "- Ask any question about how to protect your kidneys."
        )

    return (
        "Hello! I am **DKD-XAI**, your clinical AI decision support assistant for **Diabetic Kidney Disease (DKD)**.\n\n"
        "You can:\n"
        "- Describe a patient's vitals (e.g., *'68yo female with 14 years diabetes, HbA1c 8.5%, BP 138/84, on insulin'*)\n"
        "- Ask clinical questions about DKD risk factors, screening guidelines (ADA 2026 / KDIGO), or biomarker evidence."
    )


@app.post("/api/chat")
def chat(req: ChatRequest):
    extracted = extract_patient_vitals_hybrid(req.message)
    user_msg_lower = req.message.lower().strip()
    role = (req.role or "doctor").lower().strip()

    # Check if user explicitly asked to evaluate/analyze a patient
    explicit_patient_eval = bool(re.search(
        r"\b(analyze\s+benchmark|analyze\s+patient|evaluate\s+this\s+patient|assess\s+this\s+patient|calculate\s+risk|what\s+is\s+(?:the|her|his|my)?\s*risk|what\s+is\s+(?:the|her|his)?\s*kidney\s*risk|predict\s+risk|assess\s+risk)\b",
        user_msg_lower
    ))

    # Check if newly provided vitals were detected in this message
    has_extracted_vitals = len(extracted) >= 1 and any(
        k in extracted for k in ["hba1c", "sbp", "bmi", "insulin_use", "dm_duration_years", "age", "dbp", "waist_cm", "diabetes_pills"]
    )

    if not has_extracted_vitals and not explicit_patient_eval:
        # Conversational query, clinical Q&A, or off-topic question!
        reply = handle_general_or_clinical_qa(req.message, role=role)
        return {
            "reply": reply,
            "patient": req.current_patient,
            "prediction": None,
        }

    # Otherwise, patient vitals are present or evaluation was requested:
    updated_patient = dict(req.current_patient or {})
    updated_patient.update(extracted)

    # Check if we have minimum viable data to run inference
    has_minimum_data = any(updated_patient.get(k) is not None for k in ["age", "hba1c", "sbp", "bmi", "insulin_use"])
    if not has_minimum_data:
        reply = handle_general_or_clinical_qa(req.message, role=role)
        return {
            "reply": reply,
            "patient": updated_patient,
            "prediction": None,
        }

    # Run prediction and SHAP explanation
    pred = run_prediction_and_explanation(updated_patient, req.task or "dkd_any", role=role)

    risk_pct = pred["risk_percentage"]
    thresh_pct = pred["threshold_percentage"]
    is_high = pred["high_risk"]

    if role == "patient":
        # Warm, layman-friendly format
        status_title = "⚠️ Attention Recommended" if is_high else "✅ Low Risk"
        reply = f"### Kidney Health Check: {status_title} ({risk_pct}%)\n\n"
        if is_high:
            reply += "Based on the numbers you provided, your kidneys may be experiencing extra strain from diabetes.\n\n"
        else:
            reply += "Good news! Based on the numbers you provided, your kidneys appear to be in a safer, healthy range.\n\n"

        reply += "#### Key Factors In Your Health:\n"
        for c in pred["top_contributors"][:4]:
            shap_val = c.get("shap_value", 0) or 0
            feat = c.get("feature", "")
            val = c.get("value")
            val_str = f" ({val:g})" if val is not None else ""
            icon = "⚠️" if shap_val > 0 else "✅"
            impact_desc = "may put extra strain on kidneys" if shap_val > 0 else "is helping protect your kidneys"

            friendly_label = c.get("label", feat)
            if feat == "hba1c":
                friendly_label = "Blood Sugar (HbA1c)"
            elif feat == "sbp":
                friendly_label = "Blood Pressure (Systolic)"
            elif feat == "dm_duration_years":
                friendly_label = "Years with Diabetes"
            elif feat == "insulin_use":
                friendly_label = "Insulin Medication"
            elif feat == "bmi":
                friendly_label = "Body Weight / BMI"

            reply += f"- {icon} **{friendly_label}{val_str}**: {impact_desc}\n"

        reply += "\n#### Simple Next Steps:\n"
        if is_high:
            reply += "- **Talk to your doctor**: Share these numbers and ask if a routine urine test (uACR) and blood test (eGFR) can be scheduled to check your kidney function.\n"
            reply += "- **Keep taking your medications**: Follow your doctor's prescriptions for blood sugar and blood pressure.\n"
            reply += "- **Healthy habits**: Drink water regularly and aim for gentle daily walking.\n"
        else:
            reply += "- **Keep up the good habits**: Continue eating well, staying active, and taking any medications as prescribed.\n"
            reply += "- **Annual screening**: Ask your doctor to check your kidney labs once a year during regular checkups.\n"

        if pred.get("explanation"):
            reply += f"\n#### Summary for You:\n"
            reply += f"{pred['explanation']}\n"

        reply += f"\n*(Friendly Health Guide | Model: {pred['model_name']} | Always consult your doctor for medical advice)*"
    else:
        # Clinician / Doctor format
        badge = "🔴 HIGH RISK" if is_high else "🟢 LOW RISK"
        diff = "above" if is_high else "below"

        reply = f"### Risk Assessment: {badge} ({risk_pct}%)\n\n"
        reply += f"The patient's predicted risk is **{diff}** the clinical screening threshold of **{thresh_pct}%**.\n\n"

        reply += "#### Key Contributing Factors:\n"
        for c in pred["top_contributors"][:4]:
            shap_val = c.get("shap_value", 0) or 0
            shap_pct = shap_val * 100
            sign = "+" if shap_pct > 0 else ""
            impact_icon = "🔺" if shap_pct > 0 else "🔻"
            val_str = f" ({c['value']:g})" if c.get("value") is not None else ""
            reply += f"- {impact_icon} **{c['label']}{val_str}**: {sign}{shap_pct:.1f}% risk impact\n"

        reply += "\n#### Recommended Clinical Action:\n"
        if is_high:
            reply += "- Order formal diagnostic labs: **uACR** (urine albumin) and **eGFR** (serum creatinine).\n"
            reply += "- Optimize blood pressure (< 130/80 mmHg) and evaluate kidney-protective pharmacotherapy (SGLT2i / ACEi / ARB) per ADA 2026.\n"
        else:
            reply += "- Routine annual surveillance: Continue periodic uACR and eGFR monitoring.\n"
            reply += "- Maintain current glycemic control and cardiovascular risk reduction.\n"

        if pred.get("llm_used") and pred.get("explanation"):
            reply += f"\n#### Clinical Synthesis (Groq AI):\n"
            reply += f"{pred['explanation']}\n"

        ai_engine = f"Groq AI ({os.getenv('GROQ_MODEL', 'Qwen 27B')})" if pred.get("llm_used") else "Deterministic XAI Engine"
        reply += f"\n*(Model: {pred['model_name']} | Engine: {ai_engine} | Faithfulness: 100% ADA 2026 Grounded)*"

    return sanitize_nan({
        "reply": reply,
        "patient": updated_patient,
        "prediction": pred,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

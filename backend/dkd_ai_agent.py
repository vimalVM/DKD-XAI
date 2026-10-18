"""
dkd_ai_agent.py
================
The "AI Agent -> Evidence-based explanation" step of the pipeline.

INPUT   : a patient's row + whichever model dkd_model_pipeline.py saved as "best"
          for that task (best_model.joblib) - SHAP is computed live against THAT
          model (TreeSHAP for XGBoost/RandomForest, exact LinearSHAP for Logistic
          Regression, KernelSHAP as a fallback for anything else), never against
          a stale, possibly-mismatched CSV from a different model.
PROCESS : 1) rank features by |SHAP| for this patient
          2) look up a short guideline-based fact for each top feature
             (dkd_evidence_base.json - a small curated knowledge base, so
             the agent can only cite facts that actually exist)
          3) ask an LLM to turn (feature, direction, evidence) into one
             plain-language paragraph, quoting the source for every claim
          4) a faithfulness checker verifies every number and every cited
             source in the output actually came from step 1-2, and flags
             (does not silently fix) anything the LLM added that wasn't given

This is retrieval-grounded, not free generation: the LLM is only allowed to
phrase facts it was handed, never invent a new one. --no_llm runs the same
pipeline with a deterministic template instead of an LLM call, so you always
have a reproducible baseline (useful for your paper's methodology section).

RUN
    pip install groq
    export GROQ_API_KEY=gsk_...          # free key from console.groq.com/keys
    python dkd_ai_agent.py --model_dir results/dkd_any --evidence dkd_evidence_base.json --seqn 12345
    python dkd_ai_agent.py --model_dir results/dkd_any --evidence dkd_evidence_base.json --seqn 12345 --no_llm

Uses Groq's free-tier API (OpenAI-compatible), default model llama-3.3-70b-versatile.
Groq's free-tier model lineup changes over time - check console.groq.com/docs/models
and pass --model to override if llama-3.3-70b-versatile is ever retired.
"""
import argparse
import json
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# needed so joblib can unpickle the saved model bundle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import __main__
from dkd_model_pipeline import TorchMLPClassifier, NICE  # noqa: E402
__main__.TorchMLPClassifier = TorchMLPClassifier

TOP_K = 5
BACKGROUND_SIZE = 100          # rows used as the SHAP reference distribution


def _model_matrix_from_data(data, feature_names):
    """Build exactly the columns saved with the model, including optional race dummies."""
    out = pd.DataFrame(index=data.index)
    for feature in feature_names:
        if feature in data.columns:
            out[feature] = pd.to_numeric(data[feature], errors="coerce")
        elif feature.startswith("race_") and "race_ethnicity_code" in data.columns:
            try:
                code = float(feature[len("race_"):])
                out[feature] = (pd.to_numeric(data["race_ethnicity_code"], errors="coerce") == code).astype(float)
            except ValueError:
                out[feature] = np.nan
        else:
            out[feature] = np.nan
    return out[feature_names]


def _model_row_from_csv_row(row, feature_names):
    """Create one model row from a source CSV row without failing on race dummies."""
    source = row.to_frame().T
    values = _model_matrix_from_data(source, feature_names).iloc[0]
    return {feature: (None if pd.isna(values[feature]) else float(values[feature]))
            for feature in feature_names}


# =============================================================================
# 1. LIVE SHAP - dispatches by whichever model type was actually saved as best
# =============================================================================
def _transform_up_to_clf(pipe, X):
    """Run every pipeline step except the final classifier (e.g. imputer, scaler)."""
    Xt = X
    for _, step in pipe.steps[:-1]:
        Xt = step.transform(Xt)
    return Xt


def _expanded_feature_names(pipe, base_features):
    """Names after the imputer's add_indicator - e.g. adds 'missingindicator_bmi'."""
    for _, step in pipe.steps[:-1]:
        if hasattr(step, "get_feature_names_out"):
            base_features = list(step.get_feature_names_out(base_features))
    return base_features


def _fold_indicator_columns(values, expanded_names, base_features):
    """Missingness-indicator SHAP contributions get folded back into their base
    feature (e.g. 'missingindicator_bmi' adds into 'bmi'), so downstream code
    only ever has to reason about the original, evidence-base-keyed features."""
    out = {f: 0.0 for f in base_features}
    for name, v in zip(expanded_names, values):
        base = name.replace("missingindicator_", "")
        if base in out:
            out[base] += float(v)
    return out


def _positive_class(sv, expected_value):
    """Normalize the several shapes shap.TreeExplainer/LinearExplainer can return
    across sklearn model types/versions into a flat (n_features,) array for the
    positive class, e.g. RandomForest gives shape (1, n_features, 2), XGBoost
    binary gives (1, n_features), older shap gives a [class0, class1] list."""
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:                      # (n_samples, n_features, n_classes)
        sv = sv[:, :, -1]
    ev = np.ravel(expected_value)
    ev = float(ev[-1]) if ev.size > 1 else float(ev[0])
    return sv[0], ev


def compute_shap_for_row(bundle, row_df, background_df):
    """Returns (dict: feature -> shap value, base_value) for ONE patient row,
    using the SHAP method appropriate to whatever model was actually saved."""
    pipe = bundle["model"]
    clf = pipe.named_steps["clf"]
    base_features = bundle["features"]

    if isinstance(clf, (xgb.XGBClassifier, RandomForestClassifier)):
        Xt = _transform_up_to_clf(pipe, row_df)
        names = _expanded_feature_names(pipe, base_features)
        expl = shap.TreeExplainer(clf)
        sv, base_value = _positive_class(expl.shap_values(Xt), expl.expected_value)
        return _fold_indicator_columns(sv, names, base_features), base_value

    if isinstance(clf, LogisticRegression):
        Xt = _transform_up_to_clf(pipe, row_df)
        Xbg = _transform_up_to_clf(pipe, background_df)
        names = _expanded_feature_names(pipe, base_features)
        expl = shap.LinearExplainer(clf, Xbg, feature_perturbation="interventional")
        sv, base_value = _positive_class(expl.shap_values(Xt), expl.expected_value)
        return _fold_indicator_columns(sv, names, base_features), base_value

    # fallback for anything else (e.g. the MLP) - model-agnostic KernelSHAP
    f = lambda A: pipe.predict_proba(pd.DataFrame(A, columns=base_features))[:, 1]
    bg = background_df[base_features].sample(min(50, len(background_df)), random_state=0)
    expl = shap.KernelExplainer(f, bg)
    sv, base_value = _positive_class(
        expl.shap_values(row_df[base_features], nsamples=100, silent=True), expl.expected_value)
    return dict(zip(base_features, sv)), base_value


# =============================================================================
# 2. LOAD MODEL + PATIENT + LIVE SHAP
# =============================================================================
def load_patient(model_dir, csv_path, seqn=None, row_index=0, background_size=BACKGROUND_SIZE):
    bundle = joblib.load(os.path.join(model_dir, "best_model.joblib"))
    data = pd.read_csv(csv_path)

    if data.empty:
        raise ValueError(f"The patient CSV contains no rows: {csv_path}")
    if seqn is None:
        if row_index < 0 or row_index >= len(data):
            raise IndexError(f"row_index {row_index} is outside the CSV (0 to {len(data) - 1}).")
        idx = row_index
    else:
        matches = data.index[data["seqn"] == seqn].tolist()
        if not matches:
            raise ValueError(f"Participant SEQN {seqn} was not found in {csv_path}.")
        idx = matches[0]
    row = data.iloc[idx]

    feats = _model_row_from_csv_row(row, bundle["features"])
    X = pd.DataFrame([feats], columns=bundle["features"], dtype=float)
    risk = float(bundle["model"].predict_proba(X)[0, 1])
    high_risk = risk >= bundle["threshold"]

    background = _model_matrix_from_data(data, bundle["features"]).sample(
        min(background_size, len(data)), random_state=0).reset_index(drop=True)
    shap_dict, _base_value = compute_shap_for_row(bundle, X, background)

    return {
        "seqn": int(row.get("seqn", idx)),
        "features": feats,
        "shap": shap_dict,
        "risk": round(risk, 3),
        "high_risk": bool(high_risk),
        "threshold": round(bundle["threshold"], 3),
        "task": bundle["task"],
        "model_used": bundle["name"],
    }


# =============================================================================
# 3. RETRIEVE EVIDENCE FOR THE TOP FEATURES  (grounding step)
# =============================================================================
def top_contributors(patient, k=TOP_K):
    shap_map = patient["shap"]
    ranked = sorted(shap_map.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
    out = []
    for feat, sv in ranked:
        out.append({
            "feature": feat,
            "label": NICE.get(feat, feat),
            "value": patient["features"].get(feat),
            "shap_value": round(float(sv), 4),
            "direction": "pushes the model estimate upward" if sv > 0 else "pushes the model estimate downward",
        })
    return out


def retrieve_evidence(contributors, evidence_path):
    kb = json.load(open(evidence_path))
    grounded = []
    for c in contributors:
        e = kb.get(c["feature"])
        grounded.append({**c, "evidence_fact": e["fact"] if e else None,
                         "evidence_source": e["source"] if e else None,
                         "evidence_url": e.get("source_url") if e else None})
    return grounded


# =============================================================================
# 4a. DETERMINISTIC EXPLANATION (no LLM - reproducible baseline)
# =============================================================================
def deterministic_explanation(patient, grounded):
    risk_pct = round(patient["risk"] * 100, 1)
    band = ("at or above the model decision threshold" if patient["high_risk"]
            else "below the model decision threshold")
    lines = [f"Estimated risk: {risk_pct}% ({band}, decision threshold {patient['threshold']*100:.1f}%)."]
    for c in grounded:
        if c["value"] is None or c["evidence_fact"] is None:
            continue
        lines.append(f"- {c['label']} = {c['value']:g}; this contribution {c['direction']}. "
                     f"{c['evidence_fact']} (Source: {c['evidence_source']})")
    lines.append("Feature contributions are model associations, not causes. This is a statistical "
                 "estimate, not a diagnosis. Discuss these results with a doctor.")
    return "\n".join(lines)


# =============================================================================
# 4b. LLM EXPLANATION (retrieval-grounded prompt)
# =============================================================================
SYSTEM_PROMPT = """You write clean, structured, bulleted clinical explanations of a diabetic kidney
disease (DKD) risk estimate for a patient.

Rules:
1. Use ONLY the facts given to you in the CONTRIBUTORS list below. Never invent a
   clinical fact, a number, or a source that isn't in that list.
2. Every clinical claim you make must end with its (Source: ...) citation, copied
   exactly as given.
3. SHAP directions describe how this model arrived at its estimate. They are not causes,
   treatment effects, or proof that changing a feature would change a patient's outcome.
4. Report the risk percentage exactly as given (e.g. 39.8%) - do not round differently or restate
   it as a diagnosis.
5. Format with structured bullet points for readability:
   - Provide a clear 1-line risk overview stating the estimated risk percentage and threshold.
   - List each key contributor as a separate bullet point starting with its direction (pushing estimate upward or downward), the exact feature name and value, the plain-language clinical note, and the (Source: ...).
   - Do NOT write one giant unbroken wall of text. Use clear bullet points.
6. End by stating feature contributions are model associations, this is a statistical
   estimate rather than a diagnosis, and the result should be discussed with a doctor.
"""


def build_user_prompt(patient, grounded):
    risk_pct = round(patient["risk"] * 100, 1)
    lines = [f"RISK ESTIMATE: {risk_pct}% (threshold used: {patient['threshold']*100:.1f}%)",
             "CONTRIBUTORS (only use these facts):"]
    for c in grounded:
        if c["value"] is None or c["evidence_fact"] is None:
            continue
        lines.append(f"- {c['label']} = {c['value']:g}; contribution: {c['direction']}. "
                     f"Fact: {c['evidence_fact']} (Source: {c['evidence_source']})")
    return "\n".join(lines)


def llm_explanation(patient, grounded, model=None):
    if model is None:
        model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    from groq import Groq
    client = Groq()                             # reads GROQ_API_KEY from env
    resp = client.chat.completions.create(
        model=model, max_tokens=220, temperature=0.2,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": build_user_prompt(patient, grounded)}],
    )
    return resp.choices[0].message.content


# =============================================================================
# 5. FAITHFULNESS CHECK - catch anything the LLM added on its own
# =============================================================================
def check_faithfulness(explanation_text, patient, grounded):
    issues = []
    allowed_sources = {c["evidence_source"] for c in grounded if c["evidence_source"]}
    cited = set(re.findall(r"Source:\s*([^)]+)\)", explanation_text))
    for s in cited:
        if s.strip() not in allowed_sources:
            issues.append(f"Cited a source not in the retrieved evidence: '{s.strip()}'")

    # word-boundary number pattern: skips digits glued to letters (e.g. the
    # "1" inside the label "HbA1c"), which a plain \d+ regex would wrongly flag.
    NUM_RE = r"(?<![A-Za-z])\d+\.?\d*(?![A-Za-z])"
    risk_pct = round(patient["risk"] * 100, 1)
    threshold_pct = round(patient["threshold"] * 100, 1)
    numbers_in_text = {float(n) for n in re.findall(NUM_RE, explanation_text)}
    allowed_numbers = {risk_pct, threshold_pct}

    # The generated text must retain the risk percentage supplied by the model.
    expected_risk_text = f"{risk_pct:.1f}%"
    if expected_risk_text not in explanation_text:
        issues.append(f"Expected risk percentage {expected_risk_text} is missing or changed.")

    source_mentions = explanation_text.count("(Source:")
    citation_matches = list(re.finditer(r"\(Source:\s*([^)]+)\)", explanation_text))
    if source_mentions != len(citation_matches):
        issues.append("One or more source citations are malformed.")
    if any(c.get("evidence_fact") for c in grounded) and not citation_matches:
        issues.append("No evidence-source citations were included in the explanation.")
    for c in grounded:
        if c["value"] is not None:
            allowed_numbers.add(round(c["value"], 1))
            allowed_numbers.add(round(c["value"]))
        # numbers that are legitimately part of the retrieved evidence text
        # itself (e.g. "KDIGO 2022") are not model-invented - exempt them
        for txt in (c.get("evidence_fact"), c.get("evidence_source")):
            if txt:
                allowed_numbers.update(float(n) for n in re.findall(NUM_RE, txt))
    unexplained = {n for n in numbers_in_text
                  if not any(abs(n - a) < 0.15 for a in allowed_numbers)}
    if unexplained:
        issues.append(f"Numbers in text not traceable to the data given: {sorted(unexplained)}")
    return {"faithful": len(issues) == 0, "issues": issues}


# =============================================================================
# 6. RUN
# =============================================================================
def explain_patient(model_dir, csv_path, evidence_path, seqn=None, row_index=0,
                    use_llm=True, model="llama-3.3-70b-versatile"):
    patient = load_patient(model_dir, csv_path, seqn, row_index)
    grounded = retrieve_evidence(top_contributors(patient), evidence_path)

    if use_llm:
        try:
            text = llm_explanation(patient, grounded, model)
        except Exception as e:
            print(f"[LLM call failed ({e}) - falling back to deterministic template]",
                 file=sys.stderr)
            text = deterministic_explanation(patient, grounded)
    else:
        text = deterministic_explanation(patient, grounded)

    check = check_faithfulness(text, patient, grounded)
    return {"seqn": patient["seqn"], "task": patient["task"], "model_used": patient["model_used"],
            "risk": patient["risk"], "high_risk": patient["high_risk"],
            "top_contributors": grounded, "explanation": text, "faithfulness_check": check}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="results/dkd_any")
    ap.add_argument("--csv", default="dkd_nhanes_final.csv")
    ap.add_argument("--evidence", default="dkd_evidence_base.json")
    ap.add_argument("--seqn", type=int, default=None)
    ap.add_argument("--row_index", type=int, default=0)
    ap.add_argument("--no_llm", action="store_true")
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    args = ap.parse_args()

    out = explain_patient(args.model_dir, args.csv, args.evidence, args.seqn,
                          args.row_index, use_llm=not args.no_llm, model=args.model)
    print(json.dumps(out, indent=2))

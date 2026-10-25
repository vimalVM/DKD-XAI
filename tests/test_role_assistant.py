import unittest
import json
import urllib.request
import urllib.parse
from backend_server import (
    deterministic_patient_explanation,
    deterministic_explanation,
    extract_vitals_from_text,
    handle_general_or_clinical_qa
)

BASE_URL = "http://127.0.0.1:8000"


class TestRoleBasedAssistantUnit(unittest.TestCase):
    """Unit tests for role-based explanations, vitals extraction, and guardrails."""

    def setUp(self):
        self.mock_high_risk_patient = {
            "risk": 0.742,
            "high_risk": True,
            "threshold": 0.513,
            "features": {"hba1c": 9.5, "sbp": 158.0, "insulin_use": 1.0, "dm_duration_years": 16.0}
        }
        self.mock_low_risk_patient = {
            "risk": 0.185,
            "high_risk": False,
            "threshold": 0.513,
            "features": {"hba1c": 6.2, "sbp": 118.0, "insulin_use": 0.0, "dm_duration_years": 2.0}
        }
        self.mock_contributors = [
            {"feature": "hba1c", "label": "Glycated Hemoglobin (HbA1c)", "value": 9.5, "shap_value": 0.28, "direction": "pushes the model estimate upward"},
            {"feature": "sbp", "label": "Systolic BP", "value": 158.0, "shap_value": 0.22, "direction": "pushes the model estimate upward"},
            {"feature": "insulin_use", "label": "Insulin use", "value": 1.0, "shap_value": 0.15, "direction": "pushes the model estimate upward"},
            {"feature": "dm_duration_years", "label": "Diabetes duration (years)", "value": 16.0, "shap_value": 0.10, "direction": "pushes the model estimate upward"}
        ]

    def test_patient_explanation_high_risk_plain_language(self):
        """Verify patient high-risk explanation uses plain language and avoids clinical jargon."""
        explanation = deterministic_patient_explanation(self.mock_high_risk_patient, self.mock_contributors)
        self.assertIn("Kidney Health Check", explanation)
        self.assertIn("Attention Recommended", explanation)
        self.assertIn("Blood Sugar", explanation)
        self.assertIn("Blood Pressure", explanation)
        # Verify no raw statistical jargon is dumped on patient
        self.assertNotIn("decision boundary", explanation.lower())
        self.assertNotIn("linear regression", explanation.lower())
        self.assertIn("Talk to your doctor", explanation)

    def test_patient_explanation_low_risk_reassurance(self):
        """Verify patient low-risk explanation is reassuring and informative."""
        low_contributors = [
            {"feature": "hba1c", "label": "Glycated Hemoglobin (HbA1c)", "value": 6.2, "shap_value": -0.20, "direction": "pushes the model estimate downward"},
            {"feature": "sbp", "label": "Systolic BP", "value": 118.0, "shap_value": -0.15, "direction": "pushes the model estimate downward"},
        ]
        explanation = deterministic_patient_explanation(self.mock_low_risk_patient, low_contributors)
        self.assertIn("Low Risk", explanation)
        self.assertIn("Good news", explanation)
        self.assertIn("Routine annual check", explanation)

    def test_doctor_explanation_contains_clinical_metrics(self):
        """Verify doctor explanation retains technical SHAP attribution and threshold."""
        grounded_contributors = [
            {**c, "evidence_fact": "High A1c is associated with albuminuria progression.", "evidence_source": "ADA 2026 Standards"}
            for c in self.mock_contributors
        ]
        explanation = deterministic_explanation(self.mock_high_risk_patient, grounded_contributors)
        self.assertIn("decision threshold", explanation.lower())
        self.assertIn("ADA 2026", explanation)
        self.assertIn("pushes the model estimate upward", explanation)

    def test_extract_vitals_from_natural_text(self):
        """Verify extraction parses clinical vitals correctly."""
        text = "64yo male with 12 years diabetes, HbA1c 8.8%, BP 142/88, on insulin"
        vitals = extract_vitals_from_text(text)
        self.assertEqual(vitals.get("age"), 64.0)
        self.assertEqual(vitals.get("female"), 0.0)
        self.assertEqual(vitals.get("hba1c"), 8.8)
        self.assertEqual(vitals.get("sbp"), 142.0)
        self.assertEqual(vitals.get("dbp"), 88.0)
        self.assertEqual(vitals.get("insulin_use"), 1.0)
        self.assertEqual(vitals.get("dm_duration_years"), 12.0)

    def test_guardrails_patient_vs_doctor(self):
        """Verify off-topic handling adjusts between patient and doctor tone."""
        off_topic_query = "Who is Elon Musk?"
        doc_resp = handle_general_or_clinical_qa(off_topic_query, role="doctor")
        pat_resp = handle_general_or_clinical_qa(off_topic_query, role="patient")
        self.assertIn("clinical decision support", doc_resp)
        self.assertIn("Kidney Health Assistant", pat_resp)


class TestRoleBasedAssistantIntegration(unittest.TestCase):
    """Integration tests hitting the active FastAPI backend."""

    def _post(self, path, payload):
        req = urllib.request.Request(
            f"{BASE_URL}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_predict_endpoint_doctor_role(self):
        payload = {
            "patient": {
                "age": 71.0, "female": 0.0, "hba1c": 9.5, "bmi": 34.0, "waist_cm": 108.0,
                "sbp": 158.0, "dbp": 94.0, "hdl": 32.0, "total_chol": 220.0,
                "dm_diagnosed": 1.0, "insulin_use": 1.0, "diabetes_pills": 0.0, "dm_duration_years": 16.0
            },
            "task": "dkd_any",
            "role": "doctor"
        }
        res = self._post("/api/predict", payload)
        self.assertIn("risk_percentage", res)
        self.assertIn("explanation", res)
        self.assertTrue(res["high_risk"])

    def test_predict_endpoint_patient_role(self):
        payload = {
            "patient": {
                "age": 71.0, "female": 0.0, "hba1c": 9.5, "bmi": 34.0, "waist_cm": 108.0,
                "sbp": 158.0, "dbp": 94.0, "hdl": 32.0, "total_chol": 220.0,
                "dm_diagnosed": 1.0, "insulin_use": 1.0, "diabetes_pills": 0.0, "dm_duration_years": 16.0
            },
            "task": "dkd_any",
            "role": "patient"
        }
        res = self._post("/api/predict", payload)
        self.assertIn("risk_percentage", res)
        self.assertIn("Kidney Health", res["explanation"])

    def test_chat_endpoint_doctor_format(self):
        payload = {
            "message": "71yo male with 16 years of diabetes, HbA1c 9.5%, BP 158/94, on insulin. What is the risk?",
            "task": "dkd_any",
            "role": "doctor"
        }
        res = self._post("/api/chat", payload)
        reply = res["reply"]
        self.assertIn("Risk Assessment: 🔴 HIGH RISK", reply)
        self.assertIn("screening threshold", reply)
        self.assertIn("uACR", reply)
        self.assertIn("eGFR", reply)

    def test_chat_endpoint_patient_format(self):
        payload = {
            "message": "I am 71 years old, have had diabetes for 16 years, A1c is 9.5, BP 158/94, taking insulin. What does this mean?",
            "task": "dkd_any",
            "role": "patient"
        }
        res = self._post("/api/chat", payload)
        reply = res["reply"]
        self.assertIn("Kidney Health Check", reply)
        self.assertIn("Blood Sugar (HbA1c)", reply)
        self.assertIn("Talk to your doctor", reply)
        # Should not have dense raw nephrology jargon in headings
        self.assertNotIn("decision threshold of", reply)


if __name__ == "__main__":
    unittest.main()

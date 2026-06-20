"""
report_generator.py
-------------------
Uses Google Gemini API to generate structured claim review reports
from CNN prediction outputs and OpenCV damage analysis.

Set your API key:
    export GOOGLE_API_KEY="your-key-here"
or pass api_key= to ClaimsReportGenerator().
"""

import os
import json
from datetime import datetime
import google.generativeai as genai


# ── Severity helpers ──────────────────────────────────────────────────────────
SEVERITY_LABELS = {
    0: "No Damage",
    1: "Minor Damage",
    2: "Moderate Damage",
    3: "Severe Damage",
}

SEVERITY_COLOR = {
    0: "🟢",
    1: "🟡",
    2: "🟠",
    3: "🔴",
}


# ── Report generator ─────────────────────────────────────────────────────────
class ClaimsReportGenerator:
    """
    Wraps the Gemini API to produce insurance claim review reports.
    """

    def __init__(self, api_key: str = None,
                 model_name: str = "gemini-1.5-flash"):
        key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise ValueError(
                "No Gemini API key found. "
                "Set GOOGLE_API_KEY env var or pass api_key= argument."
            )
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model_name)
        print(f"[LLM] Gemini model ready: {model_name}")

    # ── Main entry point ──────────────────────────────────────────────────────
    def generate_report(self,
                        cnn_prediction: dict,
                        opencv_analysis: dict,
                        claim_metadata: dict = None) -> dict:
        """
        Generate a structured claim assessment report.

        Args:
            cnn_prediction: output of model.predict_single()
                {predicted_class, confidence, severity_score, all_probabilities}
            opencv_analysis: output of preprocess.full_pipeline()
                {n_regions, damage_regions, damage_score, ...}
            claim_metadata: optional dict with policyholder info
                {vehicle_make, vehicle_model, year, claim_number, ...}

        Returns:
            dict with keys: report_text, fraud_risk, recommendation,
                            severity_label, claim_ref, timestamp
        """
        claim_metadata = claim_metadata or {}
        prompt         = self._build_prompt(cnn_prediction,
                                            opencv_analysis,
                                            claim_metadata)
        print("[LLM] Generating claim report ...")

        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text
        except Exception as e:
            print(f"[LLM] API error: {e}")
            raw_text = self._fallback_report(cnn_prediction, opencv_analysis)

        parsed = self._parse_response(raw_text, cnn_prediction)
        parsed["timestamp"]   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        parsed["claim_ref"]   = claim_metadata.get(
            "claim_number",
            f"CLM-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        parsed["raw_prompt"]  = prompt
        return parsed

    # ── Prompt builder ────────────────────────────────────────────────────────
    def _build_prompt(self, cnn: dict, cv: dict, meta: dict) -> str:
        severity_label = SEVERITY_LABELS.get(cnn.get("severity_score", -1),
                                             cnn.get("predicted_class", "Unknown"))
        damage_score_pct = round(cv.get("damage_score", 0) * 100, 1)

        probs_str = "\n".join(
            f"  - {cls}: {pct}%"
            for cls, pct in cnn.get("all_probabilities", {}).items()
        )
        regions_str = json.dumps(cv.get("damage_regions", [])[:5], indent=2)

        vehicle_info = ""
        if meta:
            vehicle_info = f"""
Vehicle information:
  Make/Model : {meta.get('vehicle_make', 'N/A')} {meta.get('vehicle_model', '')}
  Year       : {meta.get('year', 'N/A')}
  Policy No  : {meta.get('policy_number', 'N/A')}
  Claimant   : {meta.get('claimant_name', 'N/A')}
"""

        return f"""You are a senior insurance claims analyst AI assistant. 
Analyze the following vehicle damage assessment data and produce a 
professional insurance claim review report.

=== CNN Damage Assessment ===
Predicted damage class : {cnn.get('predicted_class')}
Severity label         : {severity_label}
Confidence score       : {cnn.get('confidence')}%
Severity score (0-3)   : {cnn.get('severity_score')}

Class probabilities:
{probs_str}

=== OpenCV Image Analysis ===
Damage regions detected : {cv.get('n_regions', 0)}
Total damage area ratio : {damage_score_pct}%
Region bounding boxes   : {regions_str}
{vehicle_info}
=== Instructions ===
Generate a structured claim report with the following EXACT sections 
(use these exact headings):

**CLAIM ASSESSMENT REPORT**

**1. Executive Summary**
(2-3 sentences summarising the damage and recommendation)

**2. Vehicle Damage Analysis**
(Detail the damage location, type, and extent based on the data)

**3. Damage Severity Assessment**
(Explain the severity classification and what it means for repair costs)

**4. Supporting Evidence**
(Describe what the computer vision analysis found)

**5. Fraud Risk Assessment**
(Rate risk as LOW / MEDIUM / HIGH and explain reasoning based on:
 damage consistency, confidence score, damage area ratio)

**6. Cost Estimate Range**
(Provide a rough repair cost bracket in INR based on severity)

**7. Recommendation**
(One of: APPROVE / APPROVE WITH INSPECTION / REFER TO ADJUSTER / REJECT)
Explain the recommendation in 2-3 sentences.

**8. Notes for Insurance Officer**
(Any flags, caveats, or additional steps needed)

Keep the tone professional. Be specific about the damage data provided.
Do not invent information not present in the data.
"""

    # ── Response parser ───────────────────────────────────────────────────────
    def _parse_response(self, raw_text: str, cnn: dict) -> dict:
        """Extract key fields from the LLM response."""
        fraud_risk    = "UNKNOWN"
        recommendation = "REFER TO ADJUSTER"

        text_upper = raw_text.upper()

        # Fraud risk extraction
        for level in ["LOW", "MEDIUM", "HIGH"]:
            if f"FRAUD RISK" in text_upper and level in text_upper:
                # Prefer the one nearest "FRAUD RISK"
                idx_fr  = text_upper.find("FRAUD RISK")
                idx_lvl = text_upper.find(level, idx_fr)
                if idx_lvl != -1 and idx_lvl - idx_fr < 200:
                    fraud_risk = level
                    break

        # Recommendation extraction
        for rec in ["APPROVE WITH INSPECTION", "REFER TO ADJUSTER",
                    "REJECT", "APPROVE"]:
            if rec in text_upper:
                recommendation = rec
                break

        severity_label = SEVERITY_LABELS.get(
            cnn.get("severity_score", -1),
            cnn.get("predicted_class", "Unknown")
        )

        return {
            "report_text":     raw_text,
            "fraud_risk":      fraud_risk,
            "recommendation":  recommendation,
            "severity_label":  severity_label,
            "confidence":      cnn.get("confidence", 0),
            "severity_score":  cnn.get("severity_score", -1),
        }

    # ── Fallback (no API key / error) ─────────────────────────────────────────
    def _fallback_report(self, cnn: dict, cv: dict) -> str:
        severity_label = SEVERITY_LABELS.get(
            cnn.get("severity_score", -1), "Unknown"
        )
        return f"""**CLAIM ASSESSMENT REPORT** (Offline Mode)

**1. Executive Summary**
The submitted vehicle image has been analyzed using computer vision and 
deep learning. The system detected {severity_label.lower()} with 
{cnn.get('confidence', 0)}% confidence.

**2. Vehicle Damage Analysis**
Damage classification: {cnn.get('predicted_class', 'N/A')}
Detected damage regions: {cv.get('n_regions', 0)}

**3. Damage Severity Assessment**
Severity: {severity_label} (Score: {cnn.get('severity_score', 'N/A')}/3)

**4. Supporting Evidence**
OpenCV detected {cv.get('n_regions', 0)} distinct damage regions.
Damage area ratio: {round(cv.get('damage_score', 0) * 100, 1)}%

**5. Fraud Risk Assessment**
LOW — Automated assessment only. Manual review recommended.

**6. Cost Estimate Range**
Requires human adjuster review.

**7. Recommendation**
REFER TO ADJUSTER — Please review this claim manually.

**8. Notes for Insurance Officer**
This report was generated in offline mode. Gemini API was unavailable.
"""

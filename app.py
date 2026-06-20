"""
app.py  —  Intelligent Car Accident Claims Reviewer
====================================================
Streamlit UI for:
  1. Uploading accident images
  2. OpenCV preprocessing + visualisation
  3. CNN damage classification
  4. Gemini-powered report generation
  5. Downloadable claim report

Run:
    streamlit run app.py
"""

import os
import io
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# ── Local imports ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils.preprocess        import full_pipeline, compute_damage_score
from utils.report_generator  import ClaimsReportGenerator, SEVERITY_LABELS, SEVERITY_COLOR

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Car Claims Reviewer",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border: 1px solid #e0e0e0;
    margin-bottom: 1rem;
}
.severity-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
}
.risk-low    { background:#d4edda; color:#155724; }
.risk-medium { background:#fff3cd; color:#856404; }
.risk-high   { background:#f8d7da; color:#721c24; }
.report-box {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 10px;
    padding: 1.5rem;
    font-family: 'Segoe UI', sans-serif;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/car-crash.png", width=80)
    st.title("Configuration")

    st.subheader("🔑 API Settings")
    gemini_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get your key at https://makersuite.google.com/app/apikey"
    )

    st.subheader("🧠 Model Settings")
    model_path = st.text_input(
        "CNN Model Path",
        value="models/best_model.keras",
        help="Path to your trained .keras model file"
    )

    st.subheader("📋 Claim Metadata (optional)")
    col1, col2 = st.columns(2)
    with col1:
        vehicle_make  = st.text_input("Make",  placeholder="Toyota")
        vehicle_year  = st.text_input("Year",  placeholder="2021")
        policy_no     = st.text_input("Policy No.", placeholder="POL-001")
    with col2:
        vehicle_model = st.text_input("Model", placeholder="Camry")
        claimant_name = st.text_input("Claimant", placeholder="John Doe")
        claim_no      = st.text_input("Claim No.", placeholder="CLM-001")

    use_demo_mode = st.checkbox(
        "🎭 Demo mode (no real model needed)",
        value=False,
        help="Simulates predictions when model file is not available"
    )

    st.divider()
    st.caption("Intelligent Car Accident Claims Reviewer v1.0")


# ── Helper: load model ────────────────────────────────────────────────────────
@st.cache_resource
def get_model(model_path):
    try:
        import json
        from tensorflow.keras.models import load_model

        model = load_model(model_path, compile=False)

        with open("models/class_names.json", "r") as f:
            class_names = json.load(f)

        print("MODEL LOADED SUCCESSFULLY")
        print("CLASSES:", class_names)

        return model, class_names

    except Exception as e:
        import traceback

        print("\nMODEL LOAD ERROR:")
        print(str(e))
        traceback.print_exc()

        st.error(f"REAL ERROR: {e}")

        return None, None


def demo_predict(img_array: np.ndarray) -> dict:
    """Deterministic demo prediction based on image statistics."""
    brightness = img_array.mean()
    # Map brightness to a demo severity (purely illustrative)
    if brightness > 0.75:
        cls, sev, conf = "no_damage",        0, 91.2
    elif brightness > 0.55:
        cls, sev, conf = "minor_damage",     1, 87.4
    elif brightness > 0.35:
        cls, sev, conf = "moderate_damage",  2, 92.6
    else:
        cls, sev, conf = "severe_damage",    3, 88.9

    classes = ["no_damage", "minor_damage", "moderate_damage", "severe_damage"]
    base    = [3.0, 4.5, 5.0, 3.0]
    base[sev] = conf
    probs   = {c: round(v, 1) for c, v in zip(classes, base)}
    return {
        "predicted_class":   cls,
        "confidence":        conf,
        "severity_score":    sev,
        "all_probabilities": probs,
    }


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🚗 Intelligent Car Accident Claims Reviewer")
st.markdown(
    "Upload a vehicle damage image. The system will preprocess it with "
    "**OpenCV**, classify damage with a **CNN**, and generate a full "
    "**claim review report** using **Google Gemini AI**."
)
st.divider()

uploaded_file = st.file_uploader(
    "Upload accident image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Front, rear, or side-view damage photos work best"
)

if uploaded_file:
    # Save to temp file so OpenCV can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # ── Step 1: Display original ──────────────────────────────────────────────
    st.subheader("📷 Uploaded Image")
    pil_img = Image.open(tmp_path)
    st.image(pil_img, caption="Original", use_column_width=False, width=420)

    st.divider()

    # ── Step 2: OpenCV preprocessing ─────────────────────────────────────────
    with st.spinner("Running OpenCV preprocessing …"):
        try:
            cv_result = full_pipeline(tmp_path, save_steps=False)
            damage_score = compute_damage_score(
                cv_result["damage_regions"],
                img_area=224 * 224,
            )
            cv_result["damage_score"] = damage_score
            cv_ok = True
        except Exception as e:
            st.warning(f"OpenCV error: {e}. Using original image.")
            cv_result = {"n_regions": 0, "damage_regions": [],
                         "damage_score": 0.0}
            cv_ok = False

    st.subheader("🔬 OpenCV Preprocessing Pipeline")
    if cv_ok:
        cols = st.columns(4)
        steps = [
            ("Original",   "original"),
            ("Denoised",   "denoised"),
            ("Enhanced",   "enhanced"),
            ("Edges",      "edges"),
        ]
        for col, (title, key) in zip(cols, steps):
            frame = cv_result.get(key)
            if frame is not None:
                col.image(bgr_to_rgb(frame) if len(frame.shape) == 3
                          else frame,
                          caption=title, use_column_width=True)

        cols2 = st.columns(2)
        cols2[0].image(bgr_to_rgb(cv_result["segmented"]),
                       caption="Segmented (GrabCut)", use_column_width=True)
        cols2[1].image(bgr_to_rgb(cv_result["annotated"]),
                       caption=f"Damage regions ({cv_result['n_regions']} detected)",
                       use_column_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Damage regions", cv_result["n_regions"])
        m2.metric("Damage area ratio", f"{damage_score*100:.1f}%")
        m3.metric("Image size", "224 × 224 px")

    st.divider()

    # ── Step 3: CNN Prediction ────────────────────────────────────────────────
    st.subheader("🧠 CNN Damage Classification")

    # FIX: demo mode check and prediction now correctly scoped inside `if uploaded_file`
    with st.spinner("Running damage classification …"):
        if use_demo_mode:
            # Demo mode: use image array directly, no model needed
            from utils.data_loader import load_single_image
            img_arr    = load_single_image(tmp_path)
            cnn_result = demo_predict(img_arr)
        else:
            model, class_names = get_model(model_path)

            # FIX: model check inside the uploaded_file block
            if model is None:
                st.error(
                    f"Could not load model from '{model_path}'. "
                    "Enable Demo Mode in the sidebar or provide a valid model."
                )
                os.unlink(tmp_path)
                st.stop()

            from utils.data_loader import load_single_image
            from models.model import predict_single

            img_arr    = load_single_image(tmp_path)
            cnn_result = predict_single(model, img_arr, class_names)

    sev   = cnn_result["severity_score"]
    label = SEVERITY_LABELS.get(sev, cnn_result["predicted_class"])
    emoji = SEVERITY_COLOR.get(sev, "⚪")

    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.85rem;color:#666;">Damage class</div>
            <div style="font-size:1.6rem;font-weight:700;">{emoji} {label}</div>
            <div style="font-size:0.85rem;color:#666;margin-top:8px;">Confidence</div>
            <div style="font-size:1.4rem;font-weight:600;">{cnn_result['confidence']}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        probs = cnn_result["all_probabilities"]
        fig = go.Figure(go.Bar(
            x=list(probs.values()),
            y=list(probs.keys()),
            orientation="h",
            marker_color=["#198754","#ffc107","#fd7e14","#dc3545"],
            text=[f"{v}%" for v in probs.values()],
            textposition="outside",
        ))
        fig.update_layout(
            title="Class probabilities (%)",
            xaxis=dict(range=[0, 110], showgrid=False),
            height=220, margin=dict(l=0, r=0, t=35, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Severity gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cnn_result["confidence"],
        title={"text": "Confidence Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#0d6efd"},
            "steps": [
                {"range": [0,  60], "color": "#f8d7da"},
                {"range": [60, 80], "color": "#fff3cd"},
                {"range": [80, 100],"color": "#d4edda"},
            ],
        },
        number={"suffix": "%"},
    ))
    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=0))
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    # ── Step 4: Gemini Report ─────────────────────────────────────────────────
    st.subheader("📋 AI-Generated Claim Review Report")

    claim_metadata = {
        "vehicle_make":   vehicle_make,
        "vehicle_model":  vehicle_model,
        "year":           vehicle_year,
        "policy_number":  policy_no,
        "claimant_name":  claimant_name,
        "claim_number":   claim_no,
    }

    generate_btn = st.button("🤖 Generate Report with Gemini AI",
                             type="primary", use_container_width=True)

    if generate_btn:
        if not gemini_key:
            st.warning("Please enter your Gemini API key in the sidebar, "
                       "or the system will use an offline template.")

        with st.spinner("Gemini AI is writing your claim report …"):
            try:
                reporter = ClaimsReportGenerator(
                    api_key=gemini_key if gemini_key else "DUMMY_KEY"
                )
                report   = reporter.generate_report(
                    cnn_prediction=cnn_result,
                    opencv_analysis=cv_result,
                    claim_metadata=claim_metadata,
                )
            except Exception as e:
                # Fallback report if API fails
                reporter = ClaimsReportGenerator.__new__(ClaimsReportGenerator)
                report   = {
                    "report_text":    reporter._fallback_report(cnn_result, cv_result)
                                        if hasattr(reporter, "_fallback_report")
                                        else f"Report generation failed: {e}",
                    "fraud_risk":     "UNKNOWN",
                    "recommendation": "REFER TO ADJUSTER",
                    "severity_label": label,
                    "confidence":     cnn_result["confidence"],
                    "severity_score": sev,
                    "claim_ref":      claim_no or "N/A",
                    "timestamp":      "",
                }

            # Summary cards
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Claim Ref",  report.get("claim_ref", "N/A"))
            r2.metric("Severity",   report.get("severity_label", label))
            r3.metric("Confidence", f"{report.get('confidence', 0)}%")

            risk  = report.get("fraud_risk", "UNKNOWN")
            r_cls = {"LOW":"risk-low","MEDIUM":"risk-medium","HIGH":"risk-high"}.get(risk, "")
            r4.markdown(
                f"**Fraud Risk**<br>"
                f"<span class='severity-badge {r_cls}'>{risk}</span>",
                unsafe_allow_html=True,
            )

            st.markdown(f"""
            <div class="report-box">
            {report['report_text'].replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

            # Recommendation banner
            rec = report.get("recommendation", "REFER TO ADJUSTER")
            color = {
                "APPROVE":                  "#d4edda",
                "APPROVE WITH INSPECTION":  "#fff3cd",
                "REFER TO ADJUSTER":        "#cce5ff",
                "REJECT":                   "#f8d7da",
            }.get(rec, "#e2e3e5")
            text_color = {
                "APPROVE":                  "#155724",
                "APPROVE WITH INSPECTION":  "#856404",
                "REFER TO ADJUSTER":        "#004085",
                "REJECT":                   "#721c24",
            }.get(rec, "#383d41")

            st.markdown(f"""
            <div style="background:{color};color:{text_color};
                 border-radius:10px;padding:1rem 1.5rem;
                 font-size:1.1rem;font-weight:600;margin-top:1rem;">
              📌 Recommendation: {rec}
            </div>
            """, unsafe_allow_html=True)

            # Download button
            report_txt = report["report_text"]
            meta_block = f"""
CLAIM REFERENCE : {report.get('claim_ref', 'N/A')}
GENERATED       : {report.get('timestamp', 'N/A')}
SEVERITY        : {report.get('severity_label', label)}
CONFIDENCE      : {report.get('confidence', 0)}%
FRAUD RISK      : {risk}
RECOMMENDATION  : {rec}

{'='*60}

"""
            full_report = meta_block + report_txt

            st.download_button(
                label="⬇️ Download Report (.txt)",
                data=full_report,
                file_name=f"claim_report_{report.get('claim_ref', 'N/A')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    # FIX: cleanup moved outside the generate_btn block so it always runs
    os.unlink(tmp_path)

else:
    # Landing placeholder
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;
         background:#f8f9fa;border-radius:16px;border:2px dashed #dee2e6;">
        <div style="font-size:3rem;">🚗</div>
        <h3 style="color:#495057;margin:1rem 0 0.5rem;">Upload an accident image to get started</h3>
        <p style="color:#6c757d;">Supports JPG, PNG, WEBP — front, rear, and side-view images</p>
    </div>
    """, unsafe_allow_html=True)
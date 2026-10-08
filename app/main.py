"""
Explainable AI Document Forgery Detection & Forensic Decision Assistant
Streamlit Research Application featuring:
- Authenticity Verdict & Calibrated Confidence (MTDocForensics / Random Forest / MobileNetV2)
- Ground-Truth Grounded Pixel-Level Forgery Localization
- Multi-Method Visual Attribution (Grad-CAM, Grad-CAM++, Integrated Gradients, Occlusion)
- Handcrafted Feature Attribution (TreeSHAP on LBP, ELA, Edge, Color Descriptors)
- Actionable Spatial & Feature Counterfactual Explanations
- Case-Based Latent Exemplar Retrieval
- Live Physical Robustness Stress Testing
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# Setup Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.explainability.forensic_pipeline import ForensicInspectionSystem
from src.explainability.forensic_explainer import map_feature_to_forensic_semantics, generate_forensic_explanation_dossier

# ── Streamlit Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocForensics XAI – Explainable Document Forgery Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Forensic Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-base:      #0b0d13;
    --bg-surface:   #12151f;
    --bg-raised:    #1a1f2e;
    --accent:       #00d2aa;
    --accent-glow:  rgba(0,210,170,0.25);
    --danger:       #ff4d6d;
    --warning:      #f59e0b;
    --text-primary: #f0f2f8;
    --text-muted:   #8b91a7;
    --border:       rgba(255,255,255,0.08);
}

html, body, .stApp {
    background-color: var(--bg-base) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}

.forensic-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}

.badge-authentic {
    background: rgba(0, 210, 170, 0.15);
    color: #00d2aa;
    border: 1px solid #00d2aa;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 1.1rem;
    display: inline-block;
}

.badge-forged {
    background: rgba(255, 77, 109, 0.15);
    color: #ff4d6d;
    border: 1px solid #ff4d6d;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 1.1rem;
    display: inline-block;
}

.badge-uncertain {
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    border: 1px solid #f59e0b;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 1.1rem;
    display: inline-block;
}

.metric-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ── Cached Model Loader ────────────────────────────────────────────────────────
@st.cache_resource
def get_forensic_inspector():
    """Initializes and caches the forensic inspection system."""
    return ForensicInspectionSystem()

inspector = get_forensic_inspector()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/verified-account.png", width=64)
st.sidebar.title("DocForensics XAI")
st.sidebar.caption("Explainable AI Document Forensics System")

# Model Selection
model_choice = st.sidebar.selectbox(
    "Active Architecture",
    [
        "MT-DocForensics (Proposed)",
        "Random Forest Baseline",
        "MobileNetV2 Baseline"
    ]
)

# Document Input Mode
input_mode = st.sidebar.radio("Document Input Source", ["Upload Document", "Benchmark Sample"])

SAMPLE_MAP = {
    "Tampered Student Certificate (student_doc_304.png)": "data/processed/images/student_doc_304.png",
    "Authentic Student Certificate (student_doc_002.png)": "data/processed/images/student_doc_002.png",
    "Authentic Result Sheet (student_doc_001.png)": "data/processed/images/student_doc_001.png",
    "Attendance Altered Document (student_doc_010.png)": "data/processed/images/student_doc_010.png",
    "Candidate Details Tampered (student_doc_025.png)": "data/processed/images/student_doc_025.png",
}

img_to_analyze = None
doc_name = None

if input_mode == "Upload Document":
    uploaded_file = st.sidebar.file_uploader("Upload Document Image (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        img_to_analyze = Image.open(uploaded_file).convert("RGB")
        doc_name = uploaded_file.name
else:
    selected_sample = st.sidebar.selectbox("Choose Benchmark Sample", ["Select a document..."] + list(SAMPLE_MAP.keys()))
    if selected_sample != "Select a document...":
        sample_path = SAMPLE_MAP[selected_sample]
        if os.path.exists(sample_path):
            img_to_analyze = Image.open(sample_path).convert("RGB")
            doc_name = os.path.basename(sample_path)
        else:
            st.sidebar.error(f"Sample not found on disk at {sample_path}")

st.sidebar.markdown("---")
confidence_threshold = st.sidebar.slider("Decision Threshold", 0.30, 0.70, 0.50, 0.05)
run_slow_explainers = st.sidebar.checkbox("Compute Slow Visual Explainers (Integrated Gradients & Occlusion)", value=False)
st.sidebar.info("Decision routing: Documents within +/- 8% of threshold are flagged for manual expert verification.")

# ── Main Header ────────────────────────────────────────────────────────────────
st.title("Explainable Document Forgery Detection & Localization")
st.markdown("Multi-task deep learning architecture coupled with game-theoretic, axiomatic, and counterfactual decision interpretation.")

if img_to_analyze is None:
    # ── Landing Dashboard Overview ─────────────────────────────────────────────
    st.info("👋 Upload a document from the sidebar or choose a benchmark sample to initiate the full forensic XAI inspection pipeline.")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="forensic-card">
            <h4>Dual Detection & Localization</h4>
            <p style='color: #8b91a7;'>Simultaneously predicts document authenticity and pixel-level tampering segmentation masks with Dice supervision.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="forensic-card">
            <h4>Complementary Visual XAI</h4>
            <p style='color: #8b91a7;'>Grad-CAM, Grad-CAM++, Integrated Gradients, and Occlusion Sensitivity evaluated with Deletion AUC and Insertion AUC.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="forensic-card">
            <h4>Grounded Interpretability</h4>
            <p style='color: #8b91a7;'>TreeSHAP on forensic micro-textures, action-oriented counterfactuals, and latent space exemplar matching.</p>
        </div>
        """, unsafe_allow_html=True)
        
    # Published Benchmark Metrics Table (from outputs/metrics/model_benchmark_comparison.json)
    st.markdown("### Published Model Benchmark Results (Hold-Out Test Set: N=90)")
    bench_file = "outputs/metrics/model_benchmark_comparison.json"
    if os.path.exists(bench_file):
        with open(bench_file, "r") as f:
            bench_data = json.load(f)
        df_bench = pd.DataFrame(bench_data)
        df_bench["accuracy"] = (df_bench["accuracy"] * 100).round(2).astype(str) + "%"
        df_bench["precision"] = (df_bench["precision"] * 100).round(2).astype(str) + "%"
        df_bench["recall"] = (df_bench["recall"] * 100).round(2).astype(str) + "%"
        df_bench["f1"] = df_bench["f1"].round(4)
        df_bench["roc_auc"] = df_bench["roc_auc"].round(4)
        df_bench["pr_auc"] = df_bench["pr_auc"].round(4)
        df_bench["ece"] = df_bench["ece"].round(4)
        df_bench["mean_iou"] = df_bench["mean_iou"].apply(lambda v: f"{v:.4f}" if v > 0 else "N/A (Classifier only)")
        df_bench["mean_dice"] = df_bench["mean_dice"].apply(lambda v: f"{v:.4f}" if v > 0 else "N/A (Classifier only)")
        df_bench = df_bench.rename(columns={
            "model": "Model Architecture",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1 Score",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "ece": "Expected Calibration Error (ECE)",
            "mean_iou": "Localization mIoU",
            "mean_dice": "Dice Coefficient"
        })
        st.dataframe(df_bench, use_container_width=True)
    else:
        st.warning("Benchmark metrics file not found at outputs/metrics/model_benchmark_comparison.json.")

    # Published Visual XAI Faithfulness Table (from outputs/metrics/xai_comparison_table.json)
    st.markdown("### Visual Explainability Faithfulness & Robustness Benchmark")
    xai_file = "outputs/metrics/xai_comparison_table.json"
    if os.path.exists(xai_file):
        with open(xai_file, "r") as f:
            xai_data = json.load(f)
        df_xai = pd.DataFrame(xai_data).rename(columns={
            "method": "Explainability Method",
            "deletion_auc": "Deletion AUC (Lower = Better)",
            "insertion_auc": "Insertion AUC (Higher = Better)",
            "localization_iou": "Localization IoU",
            "stability_pearson": "Stability (Pearson r)",
            "gini_sparsity": "Gini Sparsity",
            "avg_latency_ms": "Latency (ms)"
        })
        st.dataframe(df_xai, use_container_width=True)

else:
    # ── Real Forensic Inspection Pipeline Execution ───────────────────────────
    st.markdown(f"### Forensic Analysis for `{doc_name}`")
    
    with st.spinner(f"Executing real forensic inference with {model_choice}..."):
        t0 = time.time()
        results = inspector.inspect_document(
            img_to_analyze,
            model_name=model_choice,
            run_slow_explainers=run_slow_explainers
        )
        elapsed = time.time() - t0
        
    rep = results["report"]
    prob_forged = rep["forgery_probability"]
    confidence = rep["confidence"]
    decision = rep["decision"]
    boxes = rep["bounding_boxes"]
    
    # Verdict Badges
    if "FORGED" in decision:
        status_badge = '<span class="badge-forged">🚨 FORGED / TAMPERED</span>'
    elif "GENUINE" in decision:
        status_badge = '<span class="badge-authentic">✅ GENUINE / AUTHENTIC</span>'
    else:
        status_badge = '<span class="badge-uncertain">⚠️ UNCERTAIN (MANUAL REVIEW)</span>'
        
    # Calibration ECE Lookup
    if "Random Forest" in model_choice:
        ece_val = "0.0623 (Low)"
    elif "MobileNet" in model_choice:
        ece_val = "0.3381 (Uncalibrated)"
    else:
        ece_val = "0.1287 (Calibrated)"
        
    # Top Status Bar
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.markdown(f"**Verification Verdict:**<br>{status_badge}", unsafe_allow_html=True)
    with col2:
        st.markdown(f"**Confidence:**<br><span class='metric-number'>{confidence:.1f}%</span>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"**Forged Probability:**<br><span class='metric-number'>{prob_forged*100:.1f}%</span>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"**Calibration Index:**<br><span class='metric-number' style='color:#00d2aa;'>{ece_val}</span>", unsafe_allow_html=True)
        
    st.caption(f"Inference latency: {elapsed*1000:.1f} ms | Evaluated on device: `{inspector.device}` | Architecture: `{model_choice}`")
    st.markdown("---")
    
    # ── Forensic Evidence Panel: "WHY WAS THIS DOCUMENT FLAGGED?" ─────────────
    forensic_expl = rep.get("forensic_explanation")
    if not forensic_expl:
        forensic_expl = generate_forensic_explanation_dossier(
            rep, results["gradcam"], results["pred_mask"], orig_shape=results["original_img"].shape[:2]
        )

    if "FORGED" in decision:
        panel_title = "🔎 WHY WAS THIS DOCUMENT FLAGGED?"
    elif "GENUINE" in decision:
        panel_title = "🔎 WHY WAS THIS DOCUMENT NOT FLAGGED?"
    else:
        panel_title = "🔎 FORENSIC VERDICT EXPLANATION"

    st.markdown(f"## {panel_title}")

    # Forensic Evidence Summary Card
    st.markdown(f"""
    <div class="forensic-card" style="border-left: 5px solid {'#ff4d6d' if 'FORGED' in decision else '#00d2aa'};">
        <h4 style="margin-top:0;">📋 Forensic Evidence Summary</h4>
        <p style="font-size: 1.05rem; line-height: 1.6; margin-bottom: 0;">{forensic_expl['forensic_summary']}</p>
    </div>
    """, unsafe_allow_html=True)

    # A. Overall Forensic Evidence
    st.markdown("### A. Overall Forensic Evidence")
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 14px 18px; border-radius: 8px; margin-bottom: 20px;">
        <span style="font-size: 1.02rem; color: #f0f2f8;">{forensic_expl['overall_evidence']}</span>
    </div>
    """, unsafe_allow_html=True)

    # B. Top Contributing Forensic Signals (Ranked Table)
    st.markdown("### B. Top Contributing Forensic Signals")
    signals_data = forensic_expl.get("signals_table", [])
    if signals_data:
        df_signals = pd.DataFrame(signals_data)
        df_signals_display = pd.DataFrame({
            "Feature": df_signals["feature"],
            "SHAP Contribution": df_signals["shap_formatted"],
            "Observed Value": df_signals["observed_formatted"],
            "Human-Readable Interpretation": df_signals["human_readable"],
            "Evidence Direction": df_signals["direction"]
        })
        st.dataframe(df_signals_display, use_container_width=True)
    else:
        st.info("No tabular forensic features available for attribution.")

    # C. Strongest Evidence Region (Connecting SHAP + Visual XAI + Localization)
    st.markdown("### C. Strongest Evidence Region")
    st.caption("Cross-modal verification: Correlating feature-level evidence with spatial localization and gradient saliency.")
    
    col_reg1, col_reg2, col_reg3, col_reg4 = st.columns(4)
    with col_reg1:
        st.markdown("**Original Document**")
        st.image(results["original_img"], use_container_width=True)
        st.caption("Input document canvas")
    with col_reg2:
        st.markdown(f"**Localization Box ({len(boxes)} Found)**")
        st.image(results["annotated_img"], use_container_width=True)
        st.caption("Pixel tampering bounding box")
    with col_reg3:
        st.markdown("**Grad-CAM Saliency**")
        st.image(results["gradcam_overlay"], use_container_width=True)
        st.caption("First-order gradient overlay")
    with col_reg4:
        st.markdown("**Grad-CAM++ Attribution**")
        st.image(results["gradcam_pp_overlay"], use_container_width=True)
        st.caption("Pixel-weighted feature overlay")

    overlap_info = forensic_expl.get("overlap_info", {})
    st.markdown(f"""
    <div style="background: rgba(0, 210, 170, 0.08); border-left: 4px solid #00d2aa; padding: 12px 16px; border-radius: 6px; margin-top: 10px; margin-bottom: 20px;">
        <strong>Cross-Modal Spatial Alignment:</strong> {overlap_info.get('statement', '')}<br>
        <span style="color: #8b91a7; font-size: 0.9rem;">{overlap_info.get('details', '')}</span>
    </div>
    """, unsafe_allow_html=True)

    # D. WHAT WOULD CHANGE THE DECISION? (Counterfactual Explanation)
    st.markdown("### D. WHAT WOULD CHANGE THE DECISION?")
    cf_details = forensic_expl.get("counterfactual_details", {})
    
    cf_metric1, cf_metric2, cf_metric3, cf_metric4 = st.columns(4)
    with cf_metric1:
        st.markdown(f"**Active Model Forgery Prob:**<br><span class='metric-number'>{cf_details.get('active_model_probability', prob_forged*100):.1f}%</span>", unsafe_allow_html=True)
    with cf_metric2:
        st.markdown(f"**Feature Model Prob Before:**<br><span class='metric-number'>{cf_details.get('counterfactual_probability_before', prob_forged*100):.1f}%</span>", unsafe_allow_html=True)
    with cf_metric3:
        st.markdown(f"**Target Prob After Shift:**<br><span class='metric-number' style='color:#00d2aa;'>{cf_details.get('counterfactual_probability_after', 45.0):.1f}%</span>", unsafe_allow_html=True)
    with cf_metric4:
        st.markdown(f"**Anomalous Features Perturbed:**<br><span class='metric-number'>{cf_details.get('features_involved_count', 0)}</span>", unsafe_allow_html=True)

    st.markdown(f"**Counterfactual Mechanism:** {cf_details.get('narrative', '')}")

    if cf_details.get("perturbations"):
        df_cf = pd.DataFrame(cf_details["perturbations"]).rename(columns={
            "feature": "Feature",
            "human_readable": "Forensic Characteristic",
            "original_value": "Original Value",
            "counterfactual_value": "Target Value",
            "prob_reduction": "Probability Delta"
        })
        st.dataframe(df_cf, use_container_width=True)

    # E. Distinguish Model Evidence from Document Semantics
    st.markdown("### E. Model Forensic Evidence vs. Document Semantics")
    st.markdown("""
    <div style="background: rgba(245, 158, 11, 0.08); border-left: 4px solid #f59e0b; padding: 14px 18px; border-radius: 8px; margin-bottom: 20px;">
        <strong style="color: #f59e0b; font-size: 1.05rem;">⚖️ MODEL FORENSIC EVIDENCE vs. SEMANTIC DOCUMENT CONTENT</strong>
        <p style="color: #cbd5e1; margin-top: 8px; margin-bottom: 6px; font-size: 0.95rem; line-height: 1.5;">
            The model evaluates <em>low-level physical and signal forensics</em>: micro-texture statistics (LBP), error-level compression residuals (ELA), high-frequency edge gradients, and ink/color channel distributions.
        </p>
        <p style="color: #cbd5e1; margin-bottom: 0; font-size: 0.95rem; line-height: 1.5;">
            <strong>Important distinction:</strong> The model DOES NOT automatically infer semantic alterations (e.g. <em>"student GPA was changed"</em>, <em>"student name was altered"</em>, or <em>"date was modified"</em>) without verified OCR and field-level database comparisons. Localized bounding boxes indicate <strong>WHERE</strong> anomalous signals concentrate, not <strong>WHAT</strong> semantic statement was forged.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # F. Forensic Conclusion
    st.markdown("### F. FORENSIC CONCLUSION")
    st.markdown(f"""
    <div class="forensic-card" style="border: 1px solid rgba(255,255,255,0.15);">
        <h4 style="margin-top:0; color: {'#ff4d6d' if 'FORGED' in decision else '#00d2aa'};">
            Classification: {decision} &nbsp;|&nbsp; Confidence: {confidence:.2f}%
        </h4>
        <p style="font-weight: 600; margin-bottom: 6px; color: #f0f2f8;">Primary Evidence:</p>
        <ul style="color: #cbd5e1; margin-top: 0; padding-left: 20px;">
            {"".join(f'<li style="margin-bottom: 4px;">{item}</li>' for item in forensic_expl.get('primary_evidence_items', []))}
        </ul>
        <div style="background: rgba(0,0,0,0.25); border-left: 3px solid #8b91a7; padding: 10px 14px; border-radius: 4px; margin-top: 14px;">
            <strong style="color: #8b91a7;">Important limitation:</strong><br>
            <span style="color: #a0aec0; font-size: 0.9rem;">
                This system detects forensic inconsistencies associated with document tampering. It does not independently prove which exact textual field was altered unless supported by additional OCR/content-comparison evidence.
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔬 Detailed Technical Diagnostic Tabs")
    
    # ── Tabbed Forensic Inspection Interface ──────────────────────────────────
    tab_loc, tab_vis, tab_feat, tab_cf, tab_sim, tab_stress = st.tabs([
        "📍 WHERE? (Localization)",
        "🔥 WHY? (Visual XAI)",
        "📊 WHICH FEATURES? (SHAP Attributions)",
        "🔄 WHAT WOULD CHANGE IT? (Counterfactuals)",
        "🔎 SIMILAR EXAMPLES",
        "⚡ ROBUSTNESS TEST"
    ])
    
    # Tab 1: Pixel Localization
    with tab_loc:
        st.subheader("Pixel-Level Forgery Localization")
        if "MT-DocForensics" not in model_choice:
            st.info(f"Note: {model_choice} is a whole-image classifier. Localization decoder shown below is powered by the MT-DocForensics FPN segmentation head.")
            
        c1, c2 = st.columns(2)
        with c1:
            st.image(results["original_img"], caption="Original Document Canvas", use_container_width=True)
        with c2:
            st.image(results["annotated_img"], caption=f"Localization Output ({len(boxes)} Tampering Regions Found)", use_container_width=True)
            
        if len(boxes) > 0:
            st.error(f"Detected {len(boxes)} suspicious manipulated region(s):")
            box_data = []
            for i, (bx, by, bw, bh) in enumerate(boxes):
                box_data.append({
                    "Region #": i + 1,
                    "X (Left)": bx,
                    "Y (Top)": by,
                    "Width": bw,
                    "Height": bh,
                    "Area (px)": bw * bh
                })
            st.dataframe(pd.DataFrame(box_data), use_container_width=True)
        else:
            st.success("No suspicious tampering clusters identified across the document canvas.")
            
    # Tab 2: Visual XAI Explanations
    with tab_vis:
        st.subheader("Multi-Method Visual Decision Attributions")
        st.caption("Side-by-side attribution comparison on model penultimate feature maps:")
        
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            st.markdown("**1. Grad-CAM**")
            st.image(results["gradcam_overlay"], use_container_width=True)
            st.caption("Shows image regions contributing to the classification decision.")
        with v2:
            st.markdown("**2. Grad-CAM++**")
            st.image(results["gradcam_pp_overlay"], use_container_width=True)
            st.caption("Highlights more localized/multiple discriminative regions.")
        with v3:
            st.markdown("**3. Integrated Gradients**")
            if run_slow_explainers:
                st.image(results["integrated_gradients"], clamp=True, use_container_width=True)
                st.caption("Attributes the prediction to input pixels relative to the baseline.")
            else:
                st.info("Disabled for speed. Check 'Compute Slow Visual Explainers' in sidebar to run.")
        with v4:
            st.markdown("**4. Occlusion Sensitivity**")
            if run_slow_explainers:
                st.image(results["occlusion"], clamp=True, use_container_width=True)
                st.caption("Measures prediction change when image regions are hidden.")
            else:
                st.info("Disabled for speed. Check 'Compute Slow Visual Explainers' in sidebar to run.")
                
        # Show verified faithfulness comparison table
        st.markdown("#### Published Attribution Faithfulness Benchmark")
        xai_file = "outputs/metrics/xai_comparison_table.json"
        if os.path.exists(xai_file):
            with open(xai_file, "r") as f:
                xai_data = json.load(f)
            df_xai = pd.DataFrame(xai_data)
            st.dataframe(df_xai, use_container_width=True)

    # Tab 3: SHAP Feature Attributions
    with tab_feat:
        st.subheader("Forensic Tabular Feature Attributions (TreeSHAP)")
        st.caption("Explaining the contribution of 64 engineered micro-texture, ELA, and edge features:")
        
        shap_feats = rep.get("top_contributing_features_shap", [])
        if shap_feats:
            df_shap = pd.DataFrame(shap_feats)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('#12151f')
            ax.set_facecolor('#12151f')
            
            y_pos = np.arange(len(df_shap))
            colors = ['#ff4d6d' if v > 0 else '#00d2aa' for v in df_shap["shap_value"]]
            
            ax.barh(y_pos, df_shap["shap_value"], color=colors, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df_shap["feature"], color='#f0f2f8', fontsize=10)
            ax.invert_yaxis()
            ax.tick_params(colors='#f0f2f8')
            ax.axvline(0, color='#8b91a7', linestyle='--', linewidth=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#8b91a7')
            ax.spines['bottom'].set_color('#8b91a7')
            ax.set_xlabel("SHAP Value (Positive = Pushes Forged, Negative = Pushes Authentic)", color='#f0f2f8', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("##### Detailed Feature Values & Attributions")
            df_shap_display = df_shap.copy()
            df_shap_display["Human-Readable Interpretation"] = df_shap_display["feature"].apply(
                lambda fn: map_feature_to_forensic_semantics(fn)["human_readable"]
            )
            df_shap_display = df_shap_display.rename(columns={
                "feature": "Forensic Feature Name",
                "shap_value": "SHAP Impact",
                "feature_value": "Observed Value"
            })
            display_cols = ["Forensic Feature Name", "SHAP Impact", "Observed Value", "Human-Readable Interpretation"]
            st.dataframe(df_shap_display[display_cols], use_container_width=True)
        else:
            st.info("SHAP explainer was not initialized or no tabular features were extracted.")

    # Tab 4: Counterfactual Explanations
    with tab_cf:
        st.subheader("Counterfactual Explanations")
        st.markdown("#### *'What minimal alterations would shift this document's prediction?'*")
        
        cf_c1, cf_c2 = st.columns(2)
        with cf_c1:
            st.markdown("**Actionable Feature Shifting:**")
            feat_cf = rep.get("feature_counterfactual", {})
            prob_before_cf = feat_cf.get("original_prob", prob_forged)
            prob_after_cf = feat_cf.get("counterfactual_prob", prob_forged)
            st.markdown(f"- **Current Active Model Probability:** `{prob_forged*100:.1f}%`")
            st.markdown(f"- **Feature Model Probability Before:** `{prob_before_cf*100:.1f}%`")
            st.markdown(f"- **Feature Model Probability After:** `{prob_after_cf*100:.1f}%`")
            st.write(rep.get("counterfactual_analysis", "N/A"))
            
            if feat_cf and feat_cf.get("perturbations"):
                st.markdown("##### Suggested Feature Adjustments:")
                df_pert_rows = []
                for p in feat_cf["perturbations"]:
                    fn = p.get("feature", "")
                    interp = map_feature_to_forensic_semantics(fn)["human_readable"]
                    df_pert_rows.append({
                        "Feature": fn,
                        "Forensic Characteristic": interp,
                        "Current Value": p.get("original_value", 0.0),
                        "Target Value": p.get("counterfactual_value", 0.0),
                        "Probability Delta": p.get("prob_reduction", 0.0)
                    })
                df_pert = pd.DataFrame(df_pert_rows)
                st.dataframe(df_pert, use_container_width=True)
                
        with cf_c2:
            st.markdown("**Spatial Inpainting Counterfactual:**")
            st.image(results["spatial_counterfactual"], caption="Synthesized Counterfactual Canvas (Tampered Regions Inpainted)", use_container_width=True)
            st.caption("Replacing suspected anomalous pixels with surrounding authentic texture.")

    # Tab 5: Similar Exemplars
    with tab_sim:
        st.subheader("Case-Based Latent Space Exemplars")
        st.caption("Retrieved from validation benchmark set using cosine similarity on penultimate feature embeddings:")
        
        exemplars = results.get("exemplars")
        if exemplars and isinstance(exemplars, dict):
            top_auth = exemplars.get("top_authentic", [])
            top_forg = exemplars.get("top_forged", [])
            
            ex_c1, ex_c2 = st.columns(2)
            with ex_c1:
                st.markdown("### Top Authentic Matches")
                if top_auth:
                    for ex in top_auth:
                        st.markdown(f"- **`{ex['image_name']}`** — Cosine Similarity: `{ex['similarity']:.4f}`")
                        st.caption(f"Template/Type: `{ex.get('fraud_type', 'none')}`")
                        if os.path.exists(ex.get("image_path", "")):
                            st.image(ex["image_path"], width=200)
                else:
                    st.info("No authentic exemplars retrieved.")
                    
            with ex_c2:
                st.markdown("### Top Forged Matches")
                if top_forg:
                    for ex in top_forg:
                        st.markdown(f"- **`{ex['image_name']}`** — Cosine Similarity: `{ex['similarity']:.4f}`")
                        st.caption(f"Fraud Type: `{ex.get('fraud_type', 'none')}`")
                        if os.path.exists(ex.get("image_path", "")):
                            st.image(ex["image_path"], width=200)
                else:
                    st.info("No forged exemplars retrieved.")
        else:
            st.info("Exemplar database is loading or no matches were found.")

    # Tab 6: Live Robustness Stress Test
    with tab_stress:
        st.subheader("Live Robustness Stress Testing")
        st.caption("Apply real physical degradations and evaluate prediction resilience:")
        
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            jpeg_q = st.slider("JPEG Compression Quality", 10, 100, 85, key="stress_jpeg")
            blur_k = st.slider("Gaussian Blur Kernel", 1, 15, 1, step=2, key="stress_blur")
            noise_amt = st.slider("Additive Gaussian Noise", 0.0, 0.10, 0.0, 0.01, key="stress_noise")
            
        with s_col2:
            robust_res = inspector.test_robustness(
                img_to_analyze,
                jpeg_q=jpeg_q,
                blur_k=blur_k,
                noise_amt=noise_amt
            )
            
            pert_prob = robust_res["perturbed_probability"]
            pert_verdict = robust_res["perturbed_verdict"]
            pert_conf = robust_res["perturbed_confidence"]
            
            st.markdown("**Live Perturbed Diagnostics:**")
            st.markdown(f"- Perturbed Verdict: **`{pert_verdict}`**")
            st.markdown(f"- Perturbed Forgery Probability: **{pert_prob*100:.1f}%**")
            st.markdown(f"- Confidence: **{pert_conf:.1f}%**")
            delta_p = (pert_prob - prob_forged) * 100
            st.markdown(f"- Probability Shift: **{delta_p:+.2f}%**")
            
        st.image(robust_res["perturbed_image"], caption=f"Perturbed Image Preview (JPEG Q={jpeg_q}, Blur={blur_k}, Noise={noise_amt})", use_container_width=True)

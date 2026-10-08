"""
Multi-Modal Forensic Decision Interpretation & XAI Dossier Generator
Answers:
1. "Is this document authentic or forged?"
2. "WHY did the model make this prediction?" (SHAP, LIME, Grad-CAM, Integrated Gradients, Occlusion)
3. "WHERE does the model believe the forgery is?" (Ground truth & predicted localization mask)
4. "WHICH features contributed most to the prediction?" (Forensic SHAP attributions)
5. "HOW confident and calibrated is the model?" (Confidence and calibration uncertainty interval)
6. "WHAT would need to change for the prediction to change?" (Counterfactual explanation)
7. "WHICH examples are most similar to this document?" (Latent space nearest neighbors)
"""

import os
import sys
import json
import argparse
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.deep_detector import MTDocForensics
from src.models.baseline_rf import ForensicFeatureExtractor, ForensicRandomForest
from src.explainability.gradcam.gradcam import GradCAMExplainer
from src.explainability.integrated_gradients.integrated_gradients import IntegratedGradientsExplainer
from src.explainability.occlusion.occlusion import OcclusionExplainer
from src.explainability.shap.shap_explainer import ForensicShapExplainer
from src.explainability.counterfactual.counterfactual import ForensicCounterfactualExplainer
from src.explainability.examples.example_retrieval import ExampleBasedExplainer
from src.localization.localization_eval import mask_to_bounding_boxes
from PIL import Image
import torchvision.transforms.functional as TF

from src.explainability.forensic_pipeline import ForensicInspectionSystem

def main():
    parser = argparse.ArgumentParser(description="Generate complete forensic XAI explanation dossier.")
    parser.add_argument("--image", type=str, required=True, help="Path to document image to inspect")
    parser.add_argument("--output_dir", type=str, default="outputs/explanations")
    args = parser.parse_args()
    
    inspector = ForensicInspectionSystem()
    res = inspector.inspect_document(args.image)
    
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    fig_path = os.path.join(args.output_dir, f"{base_name}_dossier.png")
    json_path = os.path.join(args.output_dir, f"{base_name}_report.json")
    
    inspector.render_dossier_figure(res, fig_path)
    with open(json_path, "w") as f:
        json.dump(res["report"], f, indent=2)
        
    rep = res["report"]
    expl = rep.get("forensic_explanation", {})
    
    print("\n" + "=" * 70)
    print("DOCFORENSICS EVIDENCE DOSSIER & VERDICT EXPLANATION")
    print("=" * 70)
    print(f"DOCUMENT FILE:   {rep['file']}")
    print(f"VERDICT:         {rep['decision']}")
    print(f"CONFIDENCE:      {rep['confidence']}%")
    print(f"FORGERY PROB:    {rep['forgery_probability']*100:.2f}%")
    print(f"REGIONS FOUND:   {rep['detected_tampering_regions_count']}")
    
    print("\n" + "-" * 70)
    print("[WHY WAS THIS DOCUMENT FLAGGED? / VERDICT RATIONALE]")
    print("-" * 70)
    print(expl.get("forensic_summary", rep.get("forensic_summary", "N/A")))
    
    print("\n" + "-" * 70)
    print("A. OVERALL FORENSIC EVIDENCE")
    print("-" * 70)
    print(expl.get("overall_evidence", "N/A"))
    
    print("\n" + "-" * 70)
    print("B. TOP CONTRIBUTING FORENSIC SIGNALS (TreeSHAP)")
    print("-" * 70)
    print(f"{'Feature':<20} | {'SHAP':<10} | {'Observed':<10} | {'Interpretation':<35}")
    print("-" * 80)
    for sig in expl.get("signals_table", [])[:6]:
        print(f"{sig['feature']:<20} | {sig['shap_formatted']:<10} | {sig['observed_formatted']:<10} | {sig['human_readable']:<35}")
        
    print("\n" + "-" * 70)
    print("C. STRONGEST EVIDENCE REGION")
    print("-" * 70)
    overlap = expl.get("overlap_info", {})
    print(f"Assessment: {overlap.get('statement', 'N/A')}")
    print(f"Details:    {overlap.get('details', 'N/A')}")
    if rep['bounding_boxes']:
        print(f"Bounding Boxes: {rep['bounding_boxes']}")
        
    print("\n" + "-" * 70)
    print("D. WHAT WOULD CHANGE THE DECISION? (Counterfactual)")
    print("-" * 70)
    cf = expl.get("counterfactual_details", {})
    print(f"Active Model Prob:         {cf.get('active_model_probability', 'N/A')}%")
    print(f"Feature Prob Before Shift: {cf.get('counterfactual_probability_before', 'N/A')}%")
    print(f"Target Prob After Shift:   {cf.get('counterfactual_probability_after', 'N/A')}%")
    print(f"Narrative:                 {cf.get('narrative', 'N/A')}")
    if cf.get("perturbations"):
        print("\nAdjustments:")
        for p in cf["perturbations"][:5]:
            print(f"  * {p['feature']} ({p['human_readable']}): {p['original_value']} -> {p['counterfactual_value']} (Delta: -{p['prob_reduction']})")

    print("\n" + "-" * 70)
    print("E. FORENSIC CONCLUSION & LIMITATIONS")
    print("-" * 70)
    print(f"Classification: {rep['decision']}")
    print(f"Confidence:     {rep['confidence']}%")
    print("Primary Evidence:")
    for item in expl.get("primary_evidence_items", []):
        print(f"  * {item}")
    print("\n" + expl.get("important_limitation", ""))
    print("=" * 70)
    print(f"SAVED DOSSIER FIGURE: {fig_path}")
    print(f"SAVED JSON REPORT:    {json_path}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()


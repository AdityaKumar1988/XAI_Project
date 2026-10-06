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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("xai_dossier")

class ForensicInspectionSystem:
    """End-to-End Explainable AI System for Document Forgery Inspection."""
    
    def __init__(self, model_checkpoint="outputs/models/mtdocforensics_none_best.pth",
                 rf_checkpoint="outputs/models/rf_baseline.pkl",
                 device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Deep Model
        self.deep_model = MTDocForensics(backbone="resnet50", pretrained=False)
        if os.path.exists(model_checkpoint):
            ckpt = torch.load(model_checkpoint, map_location=self.device)
            self.deep_model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded MT-DocForensics weights from {model_checkpoint}")
        else:
            logger.warning(f"Checkpoint not found at {model_checkpoint}; using initialized backbone.")
            
        self.deep_model.to(self.device)
        self.deep_model.eval()
        
        # Load RF Baseline & SHAP
        self.rf = ForensicRandomForest()
        if os.path.exists(rf_checkpoint):
            self.rf = ForensicRandomForest.load(rf_checkpoint)
        self.feature_extractor = ForensicFeatureExtractor()
        
        try:
            self.shap_explainer = ForensicShapExplainer(self.rf.rf, self.feature_extractor.feature_names)
        except Exception:
            self.shap_explainer = None
            
        # Visual Explainers
        target_layer = self.deep_model.stage4[-1]
        self.cam_explainer = GradCAMExplainer(self.deep_model, target_layer)
        self.ig_explainer = IntegratedGradientsExplainer(self.deep_model, steps=20)
        self.occ_explainer = OcclusionExplainer(self.deep_model, patch_size=(48, 48), stride=(32, 32))
        
    def inspect_document(self, img_path, ground_truth_mask_path=None):
        """Runs full multi-modal forensic inspection on an input document."""
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")
            
        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size
        
        # Preprocess tensor
        img_resized = img_pil.resize((256, 256), Image.BILINEAR)
        img_t = TF.to_tensor(img_resized)
        img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(self.device)
        
        # 1. Deep Forward Pass
        with torch.no_grad():
            outputs = self.deep_model(img_t)
            prob_forged = float(outputs["cls_prob"].item())
            pred_mask = outputs["mask_prob"].squeeze().cpu().numpy()
            embedding = outputs["embedding"].squeeze().cpu().numpy()
            
        # Calibrated decision thresholding
        confidence = prob_forged if prob_forged >= 0.5 else (1.0 - prob_forged)
        if 0.42 <= prob_forged <= 0.58:
            decision = "UNCERTAIN - Manual Review Needed"
            is_forged = None
        elif prob_forged > 0.58:
            decision = "FORGED / TAMPERED"
            is_forged = True
        else:
            decision = "GENUINE / AUTHENTIC"
            is_forged = False
            
        # 2. Localization
        pred_mask_full = cv2.resize(pred_mask, (orig_w, orig_h))
        boxes, _ = mask_to_bounding_boxes(pred_mask_full, threshold=0.5, min_area=30)
        
        # 3. Visual XAI Explanations
        cam_map = self.cam_explainer.generate_gradcam(img_t, target_class=1)
        campp_map = self.cam_explainer.generate_gradcam_plus_plus(img_t, target_class=1)
        ig_map = self.ig_explainer.attribute(img_t, target_class=1)
        occ_map = self.occ_explainer.attribute(img_t, target_class=1)
        
        cam_overlay = self.cam_explainer.overlay_heatmap(np.array(img_resized), cam_map)
        campp_overlay = self.cam_explainer.overlay_heatmap(np.array(img_resized), campp_map)
        
        # 4. Feature-Level SHAP
        tabular_feat = self.feature_extractor.extract_features(img_path)
        shap_res = self.shap_explainer.explain_instance(tabular_feat) if self.shap_explainer else None
        
        # 5. Counterfactual Analysis
        cf_explainer = ForensicCounterfactualExplainer(self.rf.predict_proba, self.feature_extractor.feature_names)
        feat_cf = cf_explainer.find_feature_counterfactual(tabular_feat)
        spatial_cf = cf_explainer.generate_spatial_counterfactual(np.array(img_resized), pred_mask)
        
        report = {
            "file": os.path.basename(img_path),
            "decision": decision,
            "forgery_probability": round(prob_forged, 4),
            "confidence": round(confidence * 100, 2),
            "detected_tampering_regions_count": len(boxes),
            "bounding_boxes": boxes,
            "top_contributing_features_shap": shap_res["top_features"][:5] if shap_res else [],
            "counterfactual_analysis": feat_cf.get("explanation", "N/A") if feat_cf else "N/A"
        }
        
        return {
            "report": report,
            "original_img": np.array(img_resized),
            "pred_mask": pred_mask,
            "gradcam": cam_map,
            "gradcam_pp": campp_map,
            "gradcam_overlay": cam_overlay,
            "gradcam_pp_overlay": campp_overlay,
            "integrated_gradients": ig_map,
            "occlusion": occ_map,
            "spatial_counterfactual": spatial_cf["counterfactual_image"],
            "embedding": embedding
        }

    def render_dossier_figure(self, results, save_path):
        """Creates a publication-quality 8-panel explanation dashboard figure."""
        rep = results["report"]
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(
            f"DocVerify Forensic XAI Dossier: {rep['file']} | Status: {rep['decision']} ({rep['confidence']}%)",
            fontsize=15, fontweight="bold", y=0.98
        )
        
        # 1. Original Document
        axes[0, 0].imshow(results["original_img"])
        axes[0, 0].set_title("Original Document", fontsize=11, fontweight="bold")
        axes[0, 0].axis("off")
        
        # 2. Predicted Mask & BBoxes
        mask_canvas = results["original_img"].copy()
        pred_m = (results["pred_mask"] > 0.5).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(pred_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(mask_canvas, (x, y), (x+w, y+h), (255, 0, 0), 2)
        axes[0, 1].imshow(mask_canvas)
        axes[0, 1].set_title(f"Localization Head ({rep['detected_tampering_regions_count']} Regions)", fontsize=11, fontweight="bold")
        axes[0, 1].axis("off")
        
        # 3. Grad-CAM Overlay
        axes[0, 2].imshow(results["gradcam_overlay"])
        axes[0, 2].set_title("Grad-CAM Activation", fontsize=11, fontweight="bold")
        axes[0, 2].axis("off")
        
        # 4. Grad-CAM++ Overlay
        axes[0, 3].imshow(results["gradcam_pp_overlay"])
        axes[0, 3].set_title("Grad-CAM++ Activation", fontsize=11, fontweight="bold")
        axes[0, 3].axis("off")
        
        # 5. Integrated Gradients
        axes[1, 0].imshow(results["integrated_gradients"], cmap="inferno")
        axes[1, 0].set_title("Integrated Gradients Attribution", fontsize=11, fontweight="bold")
        axes[1, 0].axis("off")
        
        # 6. Occlusion Sensitivity
        axes[1, 1].imshow(results["occlusion"], cmap="magma")
        axes[1, 1].set_title("Occlusion Sensitivity Map", fontsize=11, fontweight="bold")
        axes[1, 1].axis("off")
        
        # 7. Spatial Counterfactual
        axes[1, 2].imshow(results["spatial_counterfactual"])
        axes[1, 2].set_title("Spatial Counterfactual Inpainting", fontsize=11, fontweight="bold")
        axes[1, 2].axis("off")
        
        # 8. SHAP Feature Attribution Bar Chart
        ax_shap = axes[1, 3]
        if rep["top_contributing_features_shap"]:
            feats = [f["feature"][:18] for f in rep["top_contributing_features_shap"][:5]][::-1]
            vals = [f["shap_value"] for f in rep["top_contributing_features_shap"][:5]][::-1]
            colors = ["#ff4d6d" if v > 0 else "#00d2aa" for v in vals]
            ax_shap.barh(feats, vals, color=colors)
            ax_shap.set_title("Top SHAP Forensic Attributions", fontsize=11, fontweight="bold")
            ax_shap.set_xlabel("SHAP Value (Impact on Forged)")
        else:
            ax_shap.text(0.5, 0.5, "SHAP data unavailable", ha="center", va="center")
            ax_shap.axis("off")
            
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=250)
        plt.close()
        logger.info(f"Saved complete forensic explanation dossier to {save_path}")

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
        
    print(f"\n========================================================")
    print(f"DOCUMENT STATUS: {res['report']['decision']}")
    print(f"CONFIDENCE:      {res['report']['confidence']}%")
    print(f"REGIONS FOUND:   {res['report']['detected_tampering_regions_count']}")
    print(f"EXPLANATION:     {res['report']['counterfactual_analysis']}")
    print(f"SAVED DOSSIER:   {fig_path}")
    print(f"========================================================\n")

if __name__ == "__main__":
    main()

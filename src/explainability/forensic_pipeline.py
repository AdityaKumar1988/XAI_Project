"""
Unified Forensic Inspection Pipeline & Explainability Engine
Single Source of Truth for both CLI and Streamlit interfaces.
Integrates:
- MTDocForensics (Dual Classification & Pixel Localization)
- Handcrafted 64-dim Forensic Feature Extraction (LBP, ELA, Sobel, Color)
- ForensicRandomForest Classifier
- Grad-CAM & Grad-CAM++
- Integrated Gradients
- Occlusion Sensitivity
- TreeSHAP Feature Attributions
- Feature-Shift & Spatial Inpainting Counterfactuals
- Latent Space Nearest-Neighbor Exemplar Retrieval
- Live Physical Robustness Stress Testing
"""

import os
import sys
import io
import time
import json
import logging
import cv2
import numpy as np
from PIL import Image, ImageChops
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from src.models.deep_detector import MTDocForensics
from src.models.baseline_rf import ForensicFeatureExtractor, ForensicRandomForest
from src.explainability.gradcam.gradcam import GradCAMExplainer
from src.explainability.integrated_gradients.integrated_gradients import IntegratedGradientsExplainer
from src.explainability.occlusion.occlusion import OcclusionExplainer
from src.explainability.shap.shap_explainer import ForensicShapExplainer
from src.explainability.counterfactual.counterfactual import ForensicCounterfactualExplainer
from src.explainability.examples.example_retrieval import ExampleBasedExplainer
from src.localization.localization_eval import mask_to_bounding_boxes
from src.explainability.forensic_explainer import generate_forensic_explanation_dossier

logger = logging.getLogger("forensic_pipeline")

class ForensicInspectionSystem:
    """End-to-End Explainable AI System for Document Forgery Inspection."""
    
    def __init__(self, 
                 model_checkpoint="outputs/models/mtdocforensics_none_best.pth",
                 rf_checkpoint="outputs/models/rf_baseline.pkl",
                 device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Deep Model (MTDocForensics)
        self.deep_model = MTDocForensics(backbone="resnet50", pretrained=False)
        if os.path.exists(model_checkpoint):
            ckpt = torch.load(model_checkpoint, map_location=self.device)
            self.deep_model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded MT-DocForensics weights from {model_checkpoint}")
        else:
            logger.warning(f"Checkpoint not found at {model_checkpoint}; using unweighted backbone.")
            
        self.deep_model.to(self.device)
        self.deep_model.eval()
        
        # 1b. Load MobileNetV2 Baseline
        mobilenet_ckpt = "outputs/models/mobilenetv2_none_best.pth"
        self.mobilenet_model = None
        if os.path.exists(mobilenet_ckpt):
            try:
                from src.models.deep_detector import MobileNetV2Classifier
                self.mobilenet_model = MobileNetV2Classifier(pretrained=False)
                ckpt = torch.load(mobilenet_ckpt, map_location=self.device)
                self.mobilenet_model.load_state_dict(ckpt["model_state_dict"])
                self.mobilenet_model.to(self.device).eval()
                logger.info(f"Loaded MobileNetV2 baseline from {mobilenet_ckpt}")
            except Exception as e:
                logger.warning(f"Failed to load MobileNetV2: {e}")
                self.mobilenet_model = None
        
        # 2. Load Forensic Random Forest & TreeSHAP
        self.rf = ForensicRandomForest()
        if os.path.exists(rf_checkpoint):
            self.rf = ForensicRandomForest.load(rf_checkpoint)
        self.feature_extractor = ForensicFeatureExtractor()
        
        try:
            self.shap_explainer = ForensicShapExplainer(self.rf.rf, self.feature_extractor.feature_names)
        except Exception as e:
            logger.warning(f"Failed to initialize TreeSHAP explainer: {e}")
            self.shap_explainer = None
            
        # 3. Visual Explainers
        target_layer = self.deep_model.stage4[-1]
        self.cam_explainer = GradCAMExplainer(self.deep_model, target_layer)
        self.ig_explainer = IntegratedGradientsExplainer(self.deep_model, steps=15)
        self.occ_explainer = OcclusionExplainer(self.deep_model, patch_size=(48, 48), stride=(32, 32))
        
        # 4. Exemplar Retrieval Engine
        self.exemplar_explainer = None
        self._init_exemplars()
        
    def _init_exemplars(self):
        """Indexes or loads verified benchmark exemplars for latent nearest-neighbor retrieval."""
        cache_path = "outputs/models/exemplar_index.pkl"
        import joblib
        if os.path.exists(cache_path):
            try:
                data = joblib.load(cache_path)
                self.exemplar_explainer = ExampleBasedExplainer(data["embeddings"], data["metadata"])
                logger.info(f"Loaded {len(data['metadata'])} cached exemplars.")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached exemplars: {e}")
                
        # Build index from validation split if available
        val_csv = "data/splits/val.csv"
        if os.path.exists(val_csv):
            try:
                import pandas as pd
                df = pd.read_csv(val_csv)
                embeddings = []
                metadata = []
                for _, row in df.head(30).iterrows(): # Index first 30 balanced validation samples
                    img_p = row["image_path"]
                    if not os.path.exists(img_p):
                        continue
                    pil = Image.open(img_p).convert("RGB").resize((256, 256), Image.BILINEAR)
                    t = TF.normalize(TF.to_tensor(pil), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        out = self.deep_model(t)
                        emb = out["embedding"].cpu().numpy()
                        emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-7)
                    embeddings.append(emb_norm)
                    metadata.append({
                        "image_name": row["image_name"],
                        "image_path": row["image_path"],
                        "label": "forged" if str(row["label"]).lower() in ["fake", "forged", "1"] else "genuine",
                        "fraud_type": row.get("fraud_type", "none"),
                        "master_id": row.get("master_id", "none")
                    })
                if embeddings:
                    emb_mat = np.vstack(embeddings)
                    self.exemplar_explainer = ExampleBasedExplainer(emb_mat, metadata)
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    joblib.dump({"embeddings": emb_mat, "metadata": metadata}, cache_path)
                    logger.info(f"Indexed and saved {len(metadata)} benchmark exemplars.")
            except Exception as e:
                logger.warning(f"Could not build exemplar index: {e}")
                
    def inspect_document(self, img_input, model_name="MT-DocForensics", ground_truth_mask_path=None, run_slow_explainers=True):
        """
        Runs comprehensive multi-modal forensic inspection on an input document.
        Args:
            img_input: file path string, PIL Image, or numpy array
            model_name: "MT-DocForensics", "MobileNetV2 Baseline", or "Random Forest Baseline"
            ground_truth_mask_path: optional ground truth mask path for validation
            run_slow_explainers: whether to compute Integrated Gradients and Occlusion Sensitivity
        Returns:
            Dictionary containing reports, masks, bounding boxes, XAI heatmaps, SHAP, and counterfactuals.
        """
        # Load and normalize PIL Image
        if isinstance(img_input, str):
            if not os.path.exists(img_input):
                raise FileNotFoundError(f"Image not found at {img_input}")
            img_pil = Image.open(img_input).convert("RGB")
            doc_name = os.path.basename(img_input)
        elif isinstance(img_input, Image.Image):
            img_pil = img_input.convert("RGB")
            doc_name = getattr(img_input, "filename", "document.png") or "document.png"
        elif isinstance(img_input, np.ndarray):
            if img_input.ndim == 2:
                img_pil = Image.fromarray(cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB))
            else:
                img_pil = Image.fromarray(img_input)
            doc_name = "document.png"
        else:
            raise TypeError(f"Unsupported image input type: {type(img_input)}")
            
        orig_w, orig_h = img_pil.size
        
        # 1. Preprocess tensor for deep architectures
        img_resized = img_pil.resize((256, 256), Image.BILINEAR)
        img_t = TF.to_tensor(img_resized)
        img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(self.device)
        
        # Extract tabular features for RF and SHAP
        tabular_feat = self.feature_extractor.extract_features(img_pil)
        
        # 2. Forward Pass depending on selected architecture
        with torch.no_grad():
            outputs = self.deep_model(img_t)
            mtdoc_prob = float(outputs["cls_prob"].item())
            pred_mask = outputs["mask_prob"].squeeze().cpu().numpy()
            embedding = outputs["embedding"].squeeze().cpu().numpy()
            
        if "Random Forest" in str(model_name):
            try:
                rf_probs = self.rf.predict_proba(tabular_feat.reshape(1, -1))[0]
                prob_forged = float(rf_probs[1])
            except Exception:
                prob_forged = mtdoc_prob
        elif "MobileNet" in str(model_name) and self.mobilenet_model is not None:
            try:
                with torch.no_grad():
                    m_out = self.mobilenet_model(img_t)
                    prob_forged = float(m_out["prob"].item())
            except Exception:
                prob_forged = mtdoc_prob
        else:
            prob_forged = mtdoc_prob
            
        # Calibrated decision margin thresholding
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
            
        # 3. Pixel Localization & Bounding Box Extraction
        pred_mask_full = cv2.resize(pred_mask, (orig_w, orig_h))
        boxes, _ = mask_to_bounding_boxes(pred_mask_full, threshold=0.5, min_area=30)
        
        # Draw bounding boxes onto document copy
        annotated_img = np.array(img_pil).copy()
        for (bx, by, bw, bh) in boxes:
            cv2.rectangle(annotated_img, (bx, by), (bx + bw, by + bh), (255, 0, 70), 3)
            cv2.putText(annotated_img, "FORGERY DETECTED", (bx, max(20, by - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 70), 2)
                        
        # 4. Visual XAI Explanations
        cam_map = self.cam_explainer.generate_gradcam(img_t, target_class=1)
        campp_map = self.cam_explainer.generate_gradcam_plus_plus(img_t, target_class=1)
        cam_overlay = self.cam_explainer.overlay_heatmap(np.array(img_resized), cam_map)
        campp_overlay = self.cam_explainer.overlay_heatmap(np.array(img_resized), campp_map)
        
        if run_slow_explainers:
            ig_map = self.ig_explainer.attribute(img_t, target_class=1)
            occ_map = self.occ_explainer.attribute(img_t, target_class=1)
        else:
            ig_map = cam_map
            occ_map = campp_map
            
        # 5. Feature-Level SHAP
        shap_res = self.shap_explainer.explain_instance(tabular_feat) if self.shap_explainer else None
        
        # 6. Actionable Counterfactual Analysis
        cf_explainer = ForensicCounterfactualExplainer(self.rf.predict_proba, self.feature_extractor.feature_names)
        feat_cf = cf_explainer.find_feature_counterfactual(tabular_feat)
        spatial_cf = cf_explainer.generate_spatial_counterfactual(np.array(img_resized), pred_mask)
        
        # 7. Nearest Latent Exemplars
        exemplars = None
        if self.exemplar_explainer is not None:
            try:
                exemplars = self.exemplar_explainer.find_nearest_neighbors(embedding, k=3)
            except Exception as e:
                logger.warning(f"Exemplar lookup error: {e}")
                
        report = {
            "file": doc_name,
            "decision": decision,
            "forgery_probability": round(prob_forged, 4),
            "confidence": round(confidence * 100, 2),
            "detected_tampering_regions_count": len(boxes),
            "bounding_boxes": boxes,
            "top_contributing_features_shap": shap_res["top_features"][:6] if shap_res else [],
            "counterfactual_analysis": feat_cf.get("explanation", "N/A") if feat_cf else "N/A",
            "feature_counterfactual": feat_cf
        }
        
        # Generate human-readable forensic explanation dossier from real outputs
        forensic_dossier = generate_forensic_explanation_dossier(
            report, cam_map, pred_mask, orig_shape=(orig_h, orig_w)
        )
        report["forensic_explanation"] = forensic_dossier
        report["forensic_summary"] = forensic_dossier["forensic_summary"]
        report["overall_evidence"] = forensic_dossier["overall_evidence"]
        
        return {
            "report": report,
            "original_img": np.array(img_pil),
            "original_img_resized": np.array(img_resized),
            "annotated_img": annotated_img,
            "pred_mask": pred_mask,
            "pred_mask_full": pred_mask_full,
            "gradcam": cam_map,
            "gradcam_pp": campp_map,
            "gradcam_overlay": cam_overlay,
            "gradcam_pp_overlay": campp_overlay,
            "integrated_gradients": ig_map,
            "occlusion": occ_map,
            "spatial_counterfactual": spatial_cf["counterfactual_image"],
            "embedding": embedding,
            "exemplars": exemplars
        }
        
    def test_robustness(self, img_input, jpeg_q=85, blur_k=1, noise_amt=0.0):
        """
        Executes real stress-test on perturbed image and evaluates prediction retention.
        """
        if isinstance(img_input, str):
            img_pil = Image.open(img_input).convert("RGB")
        elif isinstance(img_input, Image.Image):
            img_pil = img_input.convert("RGB")
        else:
            img_pil = Image.fromarray(img_input).convert("RGB")
            
        pert_np = np.array(img_pil).copy()
        
        # 1. JPEG Compression
        if jpeg_q < 100:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)]
            _, enc = cv2.imencode(".jpg", cv2.cvtColor(pert_np, cv2.COLOR_RGB2BGR), encode_param)
            pert_np = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            
        # 2. Gaussian Blur
        if blur_k > 1:
            k = int(blur_k) if int(blur_k) % 2 == 1 else int(blur_k) + 1
            pert_np = cv2.GaussianBlur(pert_np, (k, k), 0)
            
        # 3. Additive Gaussian Noise
        if noise_amt > 0:
            noise = np.random.normal(0, noise_amt * 255.0, pert_np.shape).astype(np.float32)
            pert_np = np.clip(pert_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
        # Run actual forward pass on degraded image
        pert_resized = Image.fromarray(pert_np).resize((256, 256), Image.BILINEAR)
        pert_t = TF.normalize(TF.to_tensor(pert_resized), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.deep_model(pert_t)
            prob_pert = float(out["cls_prob"].item())
            
        conf_pert = prob_pert if prob_pert >= 0.5 else (1.0 - prob_pert)
        if 0.42 <= prob_pert <= 0.58:
            verdict_pert = "UNCERTAIN (MANUAL REVIEW)"
        elif prob_pert > 0.58:
            verdict_pert = "FORGED / TAMPERED"
        else:
            verdict_pert = "GENUINE / AUTHENTIC"
            
        return {
            "perturbed_image": pert_np,
            "perturbed_probability": round(prob_pert, 4),
            "perturbed_confidence": round(conf_pert * 100, 1),
            "perturbed_verdict": verdict_pert
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
        axes[0, 0].imshow(results["original_img_resized"])
        axes[0, 0].set_title("Input Document Image", fontsize=11, fontweight="bold")
        axes[0, 0].axis("off")
        
        # 2. Predicted Mask
        axes[0, 1].imshow(results["pred_mask"], cmap="gray")
        axes[0, 1].set_title(f"Localization Mask ({rep['detected_tampering_regions_count']} Regions)", fontsize=11, fontweight="bold")
        axes[0, 1].axis("off")
        
        # 3. Grad-CAM Overlay
        axes[0, 2].imshow(results["gradcam_overlay"])
        axes[0, 2].set_title("Grad-CAM Saliency Overlay", fontsize=11, fontweight="bold")
        axes[0, 2].axis("off")
        
        # 4. Grad-CAM++ Overlay
        axes[0, 3].imshow(results["gradcam_pp_overlay"])
        axes[0, 3].set_title("Grad-CAM++ Attribution Overlay", fontsize=11, fontweight="bold")
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

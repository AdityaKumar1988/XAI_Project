"""
Explainable AI: Counterfactual Explanations
Answers: "What minimal change would flip the model's prediction?"
Provides:
1. Feature-level counterfactual search (minimal feature shift across decision boundary)
2. Region-level spatial counterfactual (replacing suspected forgery patch with authentic reference background)
"""

import numpy as np
import torch
import cv2
import logging

logger = logging.getLogger("counterfactual")

class ForensicCounterfactualExplainer:
    """Computes actionable counterfactual explanations for document forgery decisions."""
    
    def __init__(self, model_predict_proba, feature_names=None):
        self.predict_proba = model_predict_proba # Function taking (N, D) -> (N, 2)
        self.feature_names = feature_names or []
        
    def find_feature_counterfactual(self, feature_vector, target_prob=0.45, max_iter=200, step_size=0.05):
        """
        Gradient-free / iterative perturbation counterfactual search.
        Finds the minimal perturbation vector delta such that P(Forged) < target_prob.
        """
        x_orig = feature_vector.copy().reshape(1, -1)
        orig_prob = self.predict_proba(x_orig)[0, 1]
        
        if orig_prob <= target_prob:
            return {
                "status": "already_target",
                "original_prob": float(orig_prob),
                "counterfactual_prob": float(orig_prob),
                "perturbations": [],
                "explanation": f"Document is classified as authentic ({orig_prob*100:.1f}% forgery probability). No counterfactual feature perturbation required to reach authenticity threshold."
            }
            
        # Iteratively perturb features towards authentic median
        x_cf = x_orig.copy()
        num_features = x_orig.shape[1]
        
        # Test sensitivity of each feature individually
        deltas = np.zeros(num_features)
        sensitivities = []
        for i in range(num_features):
            x_test_down = x_orig.copy()
            x_test_down[0, i] *= 0.8
            p_down = self.predict_proba(x_test_down)[0, 1]
            sens = orig_prob - p_down
            sensitivities.append((sens, i))
            
        sensitivities.sort(key=lambda s: s[0], reverse=True)
        
        # Sequentially modify the most impactful features
        current_prob = orig_prob
        modifications = []
        
        for sens, idx in sensitivities:
            if current_prob <= target_prob:
                break
            if sens <= 0.001:
                continue
                
            orig_val = float(x_cf[0, idx])
            # Scale down toward authentic distribution baseline
            target_val = orig_val * 0.5
            x_cf[0, idx] = target_val
            new_prob = self.predict_proba(x_cf)[0, 1]
            
            feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            modifications.append({
                "feature": feat_name,
                "original_value": round(orig_val, 4),
                "counterfactual_value": round(target_val, 4),
                "prob_reduction": round(float(current_prob - new_prob), 4)
            })
            current_prob = new_prob
            
        return {
            "status": "success" if current_prob <= target_prob else "partial",
            "original_prob": float(orig_prob),
            "counterfactual_prob": float(current_prob),
            "perturbations": modifications,
            "explanation": f"Reducing {len(modifications)} suspicious anomaly features drops forgery probability from {orig_prob*100:.1f}% to {current_prob*100:.1f}% (AUTHENTIC)."
        }

    @staticmethod
    def generate_spatial_counterfactual(img_rgb, localization_mask, reference_patch=None):
        """
        Creates visual counterfactual by blending/inpainting out the localized suspicious region.
        Labels clearly that this is an explanatory counterfactual visualization.
        """
        h, w = img_rgb.shape[:2]
        mask_binary = (localization_mask > 0.5).astype(np.uint8) * 255
        
        # Dilate mask slightly to cover boundaries
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask_binary, kernel, iterations=1)
        
        # Telea Inpainting to synthesize what document looks like without tampering anomaly
        counterfactual_img = cv2.inpaint(img_rgb, mask_dilated, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        
        # Annotate visual counterfactual
        annotated_cf = counterfactual_img.copy()
        cv2.putText(annotated_cf, "[COUNTERFACTUAL EXPLANATION: TAMPERED REGION INPAINTED]", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2, cv2.LINE_AA)
                    
        return {
            "counterfactual_image": annotated_cf,
            "inpainted_raw": counterfactual_img,
            "tampered_mask": mask_dilated
        }

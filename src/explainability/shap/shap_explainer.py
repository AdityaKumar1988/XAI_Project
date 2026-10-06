"""
Explainable AI: SHAP (SHapley Additive exPlanations) for Forensic Feature Attribution
Computes game-theoretic Shapley values on tabular forensic features (LBP, ELA, Gradients, Color).
Provides:
- Local sample feature attribution (Waterfall / Force explanation data)
- Global feature importance rankings
- Summary statistics for high-stakes decision interpretation
"""

import numpy as np
import shap
import logging

logger = logging.getLogger("shap_explainer")

class ForensicShapExplainer:
    """Explains forensic machine learning decisions using TreeSHAP."""
    
    def __init__(self, rf_model, feature_names):
        self.rf_model = rf_model
        self.feature_names = feature_names
        # TreeExplainer is exact and fast for Random Forests
        self.explainer = shap.TreeExplainer(self.rf_model)
        
    def explain_instance(self, feature_vector):
        """
        Computes SHAP values for a single document feature vector.
        Args:
            feature_vector: (D,) or (1, D) numpy array
        Returns:
            dict containing:
                shap_values: array of attribution values for the forged class (class 1)
                base_value: expected model value
                top_features: list of (feature_name, shap_val, feature_val) sorted by absolute importance
        """
        feat_2d = feature_vector.reshape(1, -1)
        raw_shap = self.explainer.shap_values(feat_2d)
        
        # Handle binary classification format across different SHAP versions
        if isinstance(raw_shap, list):
            # [class 0, class 1]
            vals = raw_shap[1][0]
            base_val = self.explainer.expected_value[1]
        elif len(raw_shap.shape) == 3:
            # (1, num_features, num_classes)
            vals = raw_shap[0, :, 1]
            base_val = self.explainer.expected_value[1]
        else:
            vals = raw_shap[0]
            base_val = self.explainer.expected_value
            
        ranked_indices = np.argsort(np.abs(vals))[::-1]
        top_features = []
        for idx in ranked_indices:
            name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            top_features.append({
                "feature": name,
                "shap_value": float(vals[idx]),
                "feature_value": float(feat_2d[0, idx])
            })
            
        return {
            "shap_values": vals,
            "base_value": float(base_val),
            "top_features": top_features
        }

    def explain_dataset(self, X_features):
        """Computes global SHAP summary across multiple documents."""
        raw_shap = self.explainer.shap_values(X_features)
        if isinstance(raw_shap, list):
            vals = raw_shap[1]
        elif len(raw_shap.shape) == 3:
            vals = raw_shap[:, :, 1]
        else:
            vals = raw_shap
            
        mean_abs_shap = np.mean(np.abs(vals), axis=0)
        global_ranking = []
        for idx in np.argsort(mean_abs_shap)[::-1]:
            name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            global_ranking.append({
                "feature": name,
                "mean_abs_shap": float(mean_abs_shap[idx])
            })
        return {
            "shap_matrix": vals,
            "global_ranking": global_ranking
        }

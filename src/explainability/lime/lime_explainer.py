"""
Explainable AI: LIME (Local Interpretable Model-agnostic Explanations)
Provides local sparse surrogate linear models around individual document samples.
Compares local feature attributions against SHAP for consistency analysis.
"""

import numpy as np
from lime import lime_tabular
import logging

logger = logging.getLogger("lime_explainer")

class ForensicLimeExplainer:
    """Computes LIME explanations on forensic tabular features."""
    
    def __init__(self, training_data, feature_names, class_names=["Genuine", "Forged"], mode="classification"):
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=class_names,
            mode=mode,
            random_state=42
        )
        
    def explain_instance(self, feature_vector, predict_fn, num_features=10):
        """
        Explains single document decision via LIME.
        Args:
            feature_vector: (D,) 1D numpy array
            predict_fn: Callable taking (N, D) and returning (N, 2) probabilities
            num_features: Number of top features to report
        Returns:
            dict with local linear weights and intercept
        """
        exp = self.explainer.explain_instance(
            data_row=feature_vector,
            predict_fn=predict_fn,
            num_features=num_features,
            labels=(1,) # Explain Forged class
        )
        
        lime_list = exp.as_list(label=1)
        results = []
        for feat_desc, weight in lime_list:
            results.append({
                "feature_description": feat_desc,
                "weight": float(weight)
            })
            
        return {
            "top_features": results,
            "intercept": float(exp.intercept[1]) if 1 in exp.intercept else 0.0,
            "score": float(exp.score)
        }

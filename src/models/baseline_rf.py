"""
Forensic Random Forest Baseline & Tabular Feature Extractor
Extracts deterministic, multi-scale texture, frequency, gradient, and ELA features.
Provides:
- Stable 64-dimensional forensic feature vector per document (no arbitrary contour truncation)
- Fully compatible with TreeSHAP and LIME for feature-level decision interpretation
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import logging

logger = logging.getLogger("baseline_rf")

class ForensicFeatureExtractor:
    """Extracts deterministic 64-dimensional forensic feature vector from document images."""
    
    def __init__(self):
        self.feature_names = self._generate_feature_names()
        
    def _generate_feature_names(self):
        names = []
        # 1. Multi-scale LBP (10 bins for R=1, 10 bins for R=2, 10 bins for R=3) -> 30 features
        for r in [1, 2, 3]:
            for b in range(10):
                names.append(f"lbp_r{r}_bin{b}")
        # 2. ELA statistics -> 8 features
        for s in ["mean", "std", "max", "p50", "p75", "p90", "p95", "skew"]:
            names.append(f"ela_{s}")
        # 3. High-frequency gradients & edges -> 8 features
        for g in ["sobel_mean", "sobel_std", "laplacian_var", "canny_dense_low", "canny_dense_med", "canny_dense_high", "grad_kurtosis", "edge_ratio"]:
            names.append(f"grad_{g}")
        # 4. Color & ink channel statistics -> 18 features (6 stats x 3 channels)
        for ch in ["r", "g", "b"]:
            for stat in ["mean", "std", "skew", "kurtosis", "p10", "p90"]:
                names.append(f"color_{ch}_{stat}")
        return names

    def extract_features(self, img_input):
        """Extracts exact 64-dim feature vector from an image file path, PIL Image, or numpy array."""
        import io
        if isinstance(img_input, str):
            if not os.path.exists(img_input):
                raise FileNotFoundError(f"Image not found at {img_input}")
            bgr = cv2.imread(img_input)
            if bgr is None:
                raise ValueError(f"Failed to load image at {img_input}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            pil_img = Image.fromarray(rgb)
        elif isinstance(img_input, Image.Image):
            pil_img = img_input.convert("RGB")
            rgb = np.array(pil_img)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        elif isinstance(img_input, np.ndarray):
            if img_input.ndim == 2:
                gray = img_input
                rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            elif img_input.shape[2] == 3:
                rgb = img_input
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            pil_img = Image.fromarray(rgb)
        else:
            raise TypeError(f"Unsupported image input type: {type(img_input)}")
            
        features = []
        
        # 1. Multi-Scale Uniform LBP (P=8, R in [1, 2, 3])
        for r in [1, 2, 3]:
            lbp = local_binary_pattern(gray, P=8, R=r, method="uniform")
            hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
            features.extend(hist.tolist())
            
        # 2. Error Level Analysis (ELA) Statistics
        try:
            buf = io.BytesIO()
            pil_img.save(buf, "JPEG", quality=90)
            buf.seek(0)
            resaved = Image.open(buf)
            diff = np.array(ImageChops.difference(pil_img, resaved), dtype=np.float32)
            ela_gray = np.mean(diff, axis=2)
            ela_stats = [
                float(np.mean(ela_gray)),
                float(np.std(ela_gray)),
                float(np.max(ela_gray)),
                float(np.percentile(ela_gray, 50)),
                float(np.percentile(ela_gray, 75)),
                float(np.percentile(ela_gray, 90)),
                float(np.percentile(ela_gray, 95)),
                float((np.mean(ela_gray) - np.median(ela_gray)) / (np.std(ela_gray) + 1e-6))
            ]
        except Exception:
            ela_stats = [0.0] * 8
        features.extend(ela_stats)
        
        # 3. High-Frequency Gradient & Edge Discontinuity
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        canny_low = cv2.Canny(gray, 30, 80)
        canny_med = cv2.Canny(gray, 50, 150)
        canny_high = cv2.Canny(gray, 100, 200)
        total_px = float(gray.shape[0] * gray.shape[1])
        
        grad_stats = [
            float(np.mean(grad_mag)),
            float(np.std(grad_mag)),
            float(np.var(laplacian)),
            float(np.sum(canny_low > 0) / total_px),
            float(np.sum(canny_med > 0) / total_px),
            float(np.sum(canny_high > 0) / total_px),
            float(np.mean((grad_mag - np.mean(grad_mag))**4) / ((np.std(grad_mag) + 1e-6)**4)),
            float((np.sum(canny_low > 0) + 1e-5) / (np.sum(canny_high > 0) + 1e-5))
        ]
        features.extend(grad_stats)
        
        # 4. Color Channel Disparities (Ink & Toner Consistency)
        for c in range(3):
            ch = rgb[:, :, c].astype(np.float32)
            m = np.mean(ch)
            s = np.std(ch) + 1e-6
            skew = np.mean(((ch - m) / s)**3)
            kurt = np.mean(((ch - m) / s)**4)
            features.extend([
                float(m),
                float(s),
                float(skew),
                float(kurt),
                float(np.percentile(ch, 10)),
                float(np.percentile(ch, 90))
            ])
            
        return np.array(features, dtype=np.float32)

class ForensicRandomForest:
    """Random Forest classifier on deterministic forensic feature representations."""
    
    def __init__(self, n_estimators=100, random_state=42):
        self.rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, class_weight="balanced")
        self.extractor = ForensicFeatureExtractor()
        self.is_fitted = False
        
    def fit(self, X_features, y):
        self.rf.fit(X_features, y)
        self.is_fitted = True
        return self
        
    def predict(self, X_features):
        return self.rf.predict(X_features)
        
    def predict_proba(self, X_features):
        return self.rf.predict_proba(X_features)
        
    def save(self, filepath="outputs/models/rf_baseline.pkl"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.rf, "feature_names": self.extractor.feature_names}, filepath)
        logger.info(f"Saved Random Forest baseline to {filepath}")
        
    @classmethod
    def load(cls, filepath="outputs/models/rf_baseline.pkl"):
        instance = cls()
        data = joblib.load(filepath)
        instance.rf = data["model"]
        instance.is_fitted = True
        return instance

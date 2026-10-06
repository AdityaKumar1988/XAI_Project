"""
Forensically Grounded Document Preprocessor
Provides:
1. Standard forensic preprocessing (non-destructive RGB normalization preserving micro-textures)
2. Error Level Analysis (ELA) conversion for compression disparity visualization
3. Legacy mode (Gaussian blur + CLAHE + deskew) strictly for scientific ablation experiments
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import logging

logger = logging.getLogger("preprocessor")

class ForensicPreprocessor:
    """Non-destructive image preprocessor preserving forensic tampering traces."""
    
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
        
    def load_image(self, path):
        """Loads image as RGB numpy array."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found at: {path}")
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to read image at: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    def preprocess_tensor(self, img_rgb, normalize=True):
        """
        Resizes using high-quality bilinear interpolation and normalizes for neural networks.
        Does NOT apply destructive smoothing or non-uniform histogram stretching.
        """
        resized = cv2.resize(img_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        img_float = resized.astype(np.float32) / 255.0
        if normalize:
            # Standard ImageNet mean and std
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_float = (img_float - mean) / std
        # (H, W, C) -> (C, H, W)
        tensor = np.transpose(img_float, (2, 0, 1))
        return tensor
        
    def preprocess_mask(self, mask_path):
        """Loads and binarizes ground-truth mask to (1, H, W)."""
        if not os.path.exists(mask_path):
            return np.zeros((1, self.target_size[0], self.target_size[1]), dtype=np.float32)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((1, self.target_size[0], self.target_size[1]), dtype=np.float32)
        resized = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        binary_mask = (resized > 127).astype(np.float32)
        return np.expand_dims(binary_mask, axis=0)
        
    def compute_ela(self, image_path, quality=90, scale=15):
        """
        Error Level Analysis (ELA).
        Re-saves image at known JPEG quality and computes absolute pixel difference.
        Forged or spliced regions compress differently than the authentic background.
        """
        original = Image.open(image_path).convert('RGB')
        ela_temp = "temp_ela.jpg"
        try:
            original.save(ela_temp, 'JPEG', quality=quality)
            resaved = Image.open(ela_temp)
            ela_diff = ImageChops.difference(original, resaved)
            
            # Enhance difference
            extrema = ela_diff.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0:
                max_diff = 1
            scale_factor = 255.0 / max_diff
            ela_enhanced = ImageEnhance.Brightness(ela_diff).enhance(scale_factor)
            return np.array(ela_enhanced)
        finally:
            if os.path.exists(ela_temp):
                try:
                    os.remove(ela_temp)
                except Exception:
                    pass

    def legacy_destructive_preprocess(self, image_path):
        """
        Legacy preprocessing pipeline (Gaussian blur + CLAHE + deskew).
        Used exclusively for ablation comparisons to measure forensic performance drop.
        """
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (7, 7), 1.0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Hough deskewing
        coords = np.column_stack(np.where(enhanced < 128))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = enhanced.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            deskewed = cv2.warpAffine(enhanced, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            deskewed = enhanced
            
        return deskewed

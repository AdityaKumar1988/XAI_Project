import cv2
import numpy as np
import pytesseract
from skimage.feature import local_binary_pattern
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureExtractor:
    """Extracts OCR, LBP, and structural features from document regions."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_ocr_features(self, roi_image):
        """Extract text and confidence features using pytesseract."""
        try:
            data = pytesseract.image_to_data(roi_image, output_type=pytesseract.Output.DICT)
            confidences = [int(c) for c in data['conf'] if c != '-1']
            
            avg_conf = np.mean(confidences) if confidences else 0
            num_words = len(confidences)
            low_conf_words = sum(1 for c in confidences if c < 50)
            
            # Character density
            text = pytesseract.image_to_string(roi_image)
            char_density = len(text.replace(" ", "")) / (roi_image.shape[0] * roi_image.shape[1]) if roi_image.size > 0 else 0
            
            return {
                'avg_ocr_conf': avg_conf,
                'num_words': num_words,
                'low_conf_words': low_conf_words,
                'char_density': char_density
            }
        except Exception as e:
            self.logger.warning(f"OCR extraction failed: {e}. Returning zeros.")
            return {'avg_ocr_conf': 0, 'num_words': 0, 'low_conf_words': 0, 'char_density': 0}

    def extract_lbp_features(self, roi_image, P=8, R=1):
        """Compute LBP histogram."""
        if len(roi_image.shape) > 2:
            roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            
        lbp = local_binary_pattern(roi_image, P, R, method='uniform')
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return hist

    def extract_structural_features(self, roi_image):
        """Compute structural features like intensity, edges, etc."""
        if len(roi_image.shape) > 2:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image
            
        mean_intensity = np.mean(gray)
        std_dev = np.std(gray)
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1]) if gray.size > 0 else 0
        
        # Blob count
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 10
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        blob_count = len(keypoints)
        
        aspect_ratio = gray.shape[1] / gray.shape[0] if gray.shape[0] > 0 else 0
        
        return {
            'mean_intensity': mean_intensity,
            'std_dev': std_dev,
            'edge_density': edge_density,
            'blob_count': blob_count,
            'aspect_ratio': aspect_ratio
        }

    def extract_all(self, roi_list, target_dim=512):
        """Extract and concatenate all features for a list of ROIs."""
        self.logger.info(f"Extracting features from {len(roi_list)} regions.")
        all_features = []
        
        for roi, _ in roi_list:
            ocr_feats = self.extract_ocr_features(roi)
            lbp_feats = self.extract_lbp_features(roi)
            struct_feats = self.extract_structural_features(roi)
            
            roi_vector = np.concatenate([
                list(ocr_feats.values()),
                lbp_feats,
                list(struct_feats.values())
            ])
            all_features.extend(roi_vector)
            
        feature_vector = np.array(all_features)
        
        # Pad or truncate to target_dim
        if len(feature_vector) > target_dim:
            feature_vector = feature_vector[:target_dim]
        else:
            feature_vector = np.pad(feature_vector, (0, target_dim - len(feature_vector)), 'constant')
            
        return feature_vector

if __name__ == "__main__":
    fe = FeatureExtractor()
    dummy_roi = np.ones((50, 150), dtype=np.uint8) * 200
    vec = fe.extract_all([(dummy_roi, (0,0,150,50))])
    print(f"Feature extractor tested. Vector shape: {vec.shape}")
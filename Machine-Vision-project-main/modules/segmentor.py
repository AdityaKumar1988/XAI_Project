import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RegionSegmentor:
    """Segments regions of interest from the document."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def adaptive_threshold(self, image):
        """Apply adaptive thresholding."""
        self.logger.info("Applying adaptive threshold.")
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )

    def detect_edges(self, binary_image):
        """Apply Canny edge detection."""
        self.logger.info("Detecting edges.")
        return cv2.Canny(binary_image, 50, 150)

    def extract_regions(self, edge_image, original_image):
        """Find contours and extract regions of interest."""
        self.logger.info("Extracting regions.")
        contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        min_area = 500  # Ignore small noise
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                roi = original_image[y:y+h, x:x+w]
                regions.append((roi, (x, y, w, h)))
                
        self.logger.info(f"Extracted {len(regions)} regions.")
        return regions

    def segment(self, preprocessed_image):
        """Run full segmentation pipeline."""
        thresh = self.adaptive_threshold(preprocessed_image)
        edges = self.detect_edges(thresh)
        regions = self.extract_regions(edges, preprocessed_image)
        
        # Draw annotated image
        annotated_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)
        for _, (x, y, w, h) in regions:
            cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        return regions, annotated_image

if __name__ == "__main__":
    seg = RegionSegmentor()
    dummy_img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(dummy_img, (50, 50), (150, 150), 255, -1)
    regions, ann = seg.segment(dummy_img)
    print(f"Segmentor module tested. Found {len(regions)} regions.")
import cv2
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentPreprocessor:
    """Handles initial image loading, cleaning, and geometric normalization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_image(self, image_path):
        """Load image using OpenCV and return original BGR image."""
        self.logger.info(f"Loading image from {image_path}")
        if not os.path.exists(image_path):
            self.logger.error("Image file not found.")
            raise FileNotFoundError(f"Could not find the image at {image_path}.")
            
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error("Failed to read image. File may be corrupted.")
            raise ValueError("Could not open or read the image.")
        return image

    def to_grayscale(self, image):
        """Convert BGR to grayscale."""
        self.logger.info("Converting to grayscale.")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def denoise(self, image, sigma=1.0):
        """Apply Gaussian blur."""
        self.logger.info("Applying denoising.")
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    def enhance_contrast(self, image):
        """Apply CLAHE for contrast enhancement."""
        self.logger.info("Enhancing contrast using CLAHE.")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)

    def correct_geometry(self, image):
        """Detects skew angle using Hough line transform and deskews."""
        self.logger.info("Correcting geometry (deskewing).")
        # Find all non-zero points (assuming dark text on light background, so we invert)
        coords = np.column_stack(np.where(image < 128))
        if len(coords) == 0:
             return image # Cannot determine skew
             
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed

    def preprocess(self, image_path):
        """Run full preprocessing pipeline."""
        img = self.load_image(image_path)
        gray = self.to_grayscale(img)
        denoised = self.denoise(gray)
        enhanced = self.enhance_contrast(denoised)
        final = self.correct_geometry(enhanced)
        self.logger.info("Preprocessing pipeline complete.")
        return final

if __name__ == "__main__":
    import tempfile
    # Simple test block
    proc = DocumentPreprocessor()
    # Create a dummy image
    dummy_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        cv2.imwrite(tmp.name, dummy_img)
        res = proc.preprocess(tmp.name)
        print("Preprocessor module tested successfully. Shape:", res.shape)
        os.remove(tmp.name)
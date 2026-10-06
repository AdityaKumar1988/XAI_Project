import cv2
import numpy as np
import logging
from utils.gradcam import GradCAM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DecisionEngine:
    """Handles final decision logic and visual annotations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def make_decision(self, label, confidence):
        """Interpret label and confidence into a string decision."""
        if confidence < 0.5:
            confidence = 1 - confidence # adjust if needed
            
        if 0.5 <= confidence <= 0.7:
            return "UNCERTAIN - Manual Review Needed"
        elif label == 1:
            return "GENUINE"
        else:
            return "FAKE"

    def generate_heatmap(self, model, image, layer_name='out_relu'):
        """Generate Grad-CAM heatmap."""
        self.logger.info("Generating Grad-CAM heatmap.")
        gradcam = GradCAM()
        
        # Preprocess image for model (Ensure RGB for MobileNetV2)
        img_resized = cv2.resize(image, (224, 224))
        if len(img_resized.shape) == 2:
            img_input = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        elif img_resized.shape[2] == 1:
            img_input = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        else:
            img_input = img_resized
            
        img_input = img_input / 255.0
        img_input = np.expand_dims(img_input, axis=0)
        
        try:
            heatmap = gradcam.compute(model, img_input, layer_name)
        except Exception as e:
            self.logger.error(f"Grad-CAM computation failed: {e}")
            return None
        
        # Resize heatmap to original image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            # If input was already RGB/BGR
            image_bgr = image.copy()
            
        overlay = cv2.addWeighted(image_bgr, 0.6, heatmap_colored, 0.4, 0)
        return overlay

    def annotate_result(self, original_image, bounding_boxes, decision, heatmap=None):
        """Annotate the image with boxes and final decision."""
        self.logger.info("Annotating final result.")
        
        if heatmap is not None:
            result_img = heatmap.copy()
        else:
            if len(original_image.shape) == 2:
                result_img = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            else:
                result_img = original_image.copy()

        # Draw bounding boxes
        for _, (x, y, w, h) in bounding_boxes:
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
        # Draw decision text
        color = (0, 255, 0) if "GENUINE" in decision else (0, 0, 255)
        if "UNCERTAIN" in decision:
            color = (0, 165, 255) # Orange
            
        cv2.putText(result_img, decision, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
        return result_img

if __name__ == "__main__":
    engine = DecisionEngine()
    print(engine.make_decision(1, 0.85))
import argparse
import os
import cv2
import logging
from modules.preprocessor import DocumentPreprocessor
from modules.segmentor import RegionSegmentor
from modules.feature_extractor import FeatureExtractor
from modules.classifier import DocumentClassifier
from modules.decision import DecisionEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def predict(image_path, model_type):
    if not os.path.exists(image_path):
        logging.error(f"Image not found: {image_path}")
        return

    os.makedirs("output", exist_ok=True)
    
    preproc = DocumentPreprocessor()
    seg = RegionSegmentor()
    clf = DocumentClassifier()
    engine = DecisionEngine()
    
    # 1. Preprocess
    img_gray = preproc.preprocess(image_path)
    
    # 2. Segment
    regions, annotated_seg = seg.segment(img_gray)
    
    heatmap = None
    # 3. Classify
    if model_type.lower() == 'cnn':
        label, conf = clf.predict_cnn(img_gray)
        try:
            from tensorflow.keras.models import load_model
            model = load_model(clf.cnn_path)
            heatmap = engine.generate_heatmap(model, img_gray)
        except Exception as e:
            logging.warning(f"Could not generate heatmap: {e}")
    else:
        fe = FeatureExtractor()
        if not regions:
            regions = [(img_gray, (0, 0, img_gray.shape[1], img_gray.shape[0]))]
        feature_vector = fe.extract_all(regions)
        label, conf = clf.predict_rf(feature_vector)
        
    # 4. Decision
    decision = engine.make_decision(label, conf)
    
    # 5. Output
    print(f"\nDocument: {os.path.basename(image_path)} | Result: {decision} | Confidence: {conf*100:.1f}%\n")
    
    final_img = engine.annotate_result(img_gray, regions, f"{decision} ({conf*100:.1f}%)", heatmap)
    output_path = os.path.join("output", "result.jpg")
    cv2.imwrite(output_path, final_img)
    logging.info(f"Saved annotated output to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the document image")
    parser.add_argument("--model", type=str, choices=["cnn", "rf"], default="cnn", help="Model to use (cnn or rf)")
    args = parser.parse_args()
    
    predict(args.image, args.model)
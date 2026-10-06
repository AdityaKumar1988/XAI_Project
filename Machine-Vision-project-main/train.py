import os
import numpy as np
import logging
import cv2
import joblib
from modules.preprocessor import DocumentPreprocessor
from modules.segmentor import RegionSegmentor
from modules.feature_extractor import FeatureExtractor
from modules.classifier import DocumentClassifier
from utils.data_loader import load_dataset, split_dataset
from utils.evaluate import plot_confusion_matrix, plot_roc_curve, print_classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_images_for_rf(image_paths):
    preproc = DocumentPreprocessor()
    seg = RegionSegmentor()
    fe = FeatureExtractor()
    
    X_features = []
    valid_indices = []
    
    for i, path in enumerate(image_paths):
        try:
            img = preproc.preprocess(path)
            regions, _ = seg.segment(img)
            if not regions:
                # Fallback if no regions detected: use whole image
                regions = [(img, (0, 0, img.shape[1], img.shape[0]))]
            features = fe.extract_all(regions)
            X_features.append(features)
            valid_indices.append(i)
        except Exception as e:
            logging.error(f"Error processing {path}: {e}")
            
    return np.array(X_features), valid_indices

def process_images_for_cnn(image_paths):
    X_images = []
    valid_indices = []
    for i, path in enumerate(image_paths):
        try:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            X_images.append(img)
            valid_indices.append(i)
        except Exception as e:
            logging.error(f"Error loading {path} for CNN: {e}")
    return np.array(X_images), valid_indices

def main():
    logger = logging.getLogger(__name__)
    logger.info("Starting training pipeline...")
    
    # 1. Load dataset
    dataset_path = "dataset/"
    # Create dummy dirs if they don't exist
    os.makedirs(os.path.join(dataset_path, "genuine"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "fake"), exist_ok=True)
    
    X_paths, y = load_dataset(dataset_path)
    
    if len(X_paths) == 0:
        logger.error("Dataset is empty. Please add images to dataset/genuine and dataset/fake.")
        return

    # 2. Split dataset
    X_train_p, X_val_p, X_test_p, y_train_all, y_val, y_test_all = split_dataset(X_paths, y)
    
    clf = DocumentClassifier()

    # --- Train Random Forest ---
    logger.info("Extracting features for Random Forest...")
    X_train_rf, valid_train = process_images_for_rf(X_train_p)
    y_train_rf = y_train_all[valid_train]
    
    if len(X_train_rf) > 0:
        clf.train_rf(X_train_rf, y_train_rf)
        
        logger.info("Evaluating Random Forest...")
        X_test_rf, valid_test = process_images_for_rf(X_test_p)
        y_test_rf = y_test_all[valid_test]
        
        if len(X_test_rf) > 0:
            rf = joblib.load(clf.rf_path)
            # Evaluate using predict_rf logic loop
            preds = []
            probs = []
            for feat in X_test_rf:
                p, prob = clf.predict_rf(feat)
                preds.append(p)
                probs.append(prob)
            
            print_classification_report(y_test_rf, preds)
            plot_confusion_matrix(y_test_rf, preds, ["Fake", "Genuine"], "output/rf")
            plot_roc_curve(y_test_rf, probs, "output/rf")
    else:
        logger.warning("No valid training data for Random Forest.")

    # --- Train CNN ---
    logger.info("Preparing images for CNN...")
    X_train_cnn, valid_train_cnn = process_images_for_cnn(X_train_p)
    y_train_cnn = y_train_all[valid_train_cnn]
    
    if len(X_train_cnn) > 0:
        clf.train_cnn(X_train_cnn, y_train_cnn)
        
        logger.info("Evaluating CNN...")
        X_test_cnn, valid_test_cnn = process_images_for_cnn(X_test_p)
        y_test_cnn = y_test_all[valid_test_cnn]
        
        if len(X_test_cnn) > 0:
            from tensorflow.keras.models import load_model
            model = load_model(clf.cnn_path)
            preds_prob = model.predict(X_test_cnn)
            preds = (preds_prob > 0.5).astype(int).flatten()
            probs = preds_prob.flatten()
            
            print_classification_report(y_test_cnn, preds)
            plot_confusion_matrix(y_test_cnn, preds, ["Fake", "Genuine"], "output/cnn")
            plot_roc_curve(y_test_cnn, probs, "output/cnn")
    else:
        logger.warning("No valid training data for CNN.")
        
    logger.info("Training script finished.")

if __name__ == "__main__":
    main()
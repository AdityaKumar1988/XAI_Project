import os
import cv2
import numpy as np
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dataset(dataset_path):
    """
    Loads all images from genuine/ and fake/ folders.
    Returns:
        X (images as paths or pre-loaded arrays), y (labels: 1 for genuine, 0 for fake)
    """
    logger = logging.getLogger(__name__)
    X_paths = []
    y = []
    
    genuine_dir = os.path.join(dataset_path, "genuine")
    fake_dir = os.path.join(dataset_path, "fake")
    
    if not os.path.exists(genuine_dir) or not os.path.exists(fake_dir):
        logger.error(f"Dataset directories not found in {dataset_path}. Ensure 'genuine' and 'fake' folders exist.")
        return np.array([]), np.array([])

    # Load Genuine (Label 1)
    for filename in os.listdir(genuine_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            X_paths.append(os.path.join(genuine_dir, filename))
            y.append(1)
            
    # Load Fake (Label 0)
    for filename in os.listdir(fake_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            X_paths.append(os.path.join(fake_dir, filename))
            y.append(0)
            
    logger.info(f"Loaded {len(X_paths)} image paths (Genuine: {y.count(1)}, Fake: {y.count(0)})")
    return np.array(X_paths), np.array(y)

def split_dataset(X, y, test_size=0.2, val_size=0.1):
    """
    Splits dataset into train, validation, and test sets.
    """
    # First split into train_val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Adjust val_size relative to train_val
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=relative_val_size, random_state=42, stratify=y_train_val)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    print("Data loader utility ready.")
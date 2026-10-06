"""
Unified Training Script for Document Forgery Detection and Localization
Supports:
- Random Forest Baseline (rf)
- MobileNetV2 (mobilenetv2)
- ResNet-50 (resnet50)
- Multi-Task MT-DocForensics (mtdocforensics)
- Ablation experiments: destructive_preproc, no_aug, cls_only
"""

import os
import sys
import time
import json
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import get_dataloaders
from src.models.baseline_rf import ForensicRandomForest, ForensicFeatureExtractor
from src.models.deep_detector import MobileNetV2Classifier, ResNet50Classifier, MTDocForensics, MultiTaskForensicLoss
from src.training.train_utils import Trainer
from src.evaluation.metrics import evaluate_classification, evaluate_localization, compute_calibration_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_script")

def train_rf_baseline(splits_dir="data/splits", output_dir="outputs/models", metrics_dir="outputs/metrics"):
    """Trains and evaluates deterministic Random Forest baseline."""
    import pandas as pd
    logger.info("--- Training Forensic Random Forest Baseline ---")
    start_time = time.time()
    
    train_df = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))
    
    extractor = ForensicFeatureExtractor()
    rf_model = ForensicRandomForest(n_estimators=100, random_state=42)
    
    logger.info(f"Extracting 64-dim forensic features for {len(train_df)} training documents...")
    X_train = []
    y_train = []
    for _, row in train_df.iterrows():
        feat = extractor.extract_features(row["image_path"])
        label = 1 if str(row["label"]).lower() in ["forged", "fake", "1"] else 0
        X_train.append(feat)
        y_train.append(label)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    logger.info(f"Fitting Random Forest on feature matrix shape: {X_train.shape}...")
    rf_model.fit(X_train, y_train)
    
    logger.info(f"Extracting features and evaluating on {len(test_df)} test documents...")
    X_test = []
    y_test = []
    t_inf_start = time.time()
    for _, row in test_df.iterrows():
        feat = extractor.extract_features(row["image_path"])
        label = 1 if str(row["label"]).lower() in ["forged", "fake", "1"] else 0
        X_test.append(feat)
        y_test.append(label)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    preds_prob = rf_model.predict_proba(X_test)[:, 1]
    inference_time_per_sample = (time.time() - t_inf_start) / len(test_df)
    training_time = time.time() - start_time
    
    metrics = evaluate_classification(y_test, preds_prob)
    calib = compute_calibration_error(y_test, preds_prob)
    metrics["ece"] = calib["ece"]
    metrics["brier_score"] = calib["brier_score"]
    metrics["training_time_seconds"] = round(training_time, 2)
    metrics["inference_time_seconds"] = round(inference_time_per_sample, 4)
    metrics["model_name"] = "RandomForest_Baseline"
    metrics["num_features"] = 64
    
    # Save model and metrics
    rf_path = os.path.join(output_dir, "rf_baseline.pkl")
    rf_model.save(rf_path)
    
    metrics_path = os.path.join(metrics_dir, "rf_test_metrics.json")
    os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    logger.info(f"Random Forest Test Results -> Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['roc_auc']:.4f} | ECE: {metrics['ece']:.4f}")
    return metrics

def train_deep_model(model_name="mtdocforensics", splits_dir="data/splits", epochs=15, batch_size=16,
                     lr=1e-4, device=None, output_dir="outputs/models", metrics_dir="outputs/metrics",
                     ablation="none"):
    """Trains and evaluates deep learning document forgery architecture."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"--- Training {model_name.upper()} on {device} (Ablation: {ablation}) ---")
    
    train_loader, val_loader, test_loader = get_dataloaders(splits_dir, img_size=(256, 256), batch_size=batch_size)
    
    # Model instantiation
    if model_name == "mobilenetv2":
        model = MobileNetV2Classifier(pretrained=True)
        criterion = nn.BCEWithLogitsLoss()
    elif model_name == "resnet50":
        model = ResNet50Classifier(pretrained=True)
        criterion = nn.BCEWithLogitsLoss()
    elif model_name == "mtdocforensics":
        model = MTDocForensics(backbone="resnet50", pretrained=True)
        if ablation == "cls_only":
            # Multi-task architecture trained without localization supervision
            criterion = MultiTaskForensicLoss(lambda_loc=0.0)
        else:
            criterion = MultiTaskForensicLoss(lambda_loc=1.0)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable Parameters: {param_count:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        output_dir=output_dir,
        model_name=f"{model_name}_{ablation}"
    )
    
    start_t = time.time()
    history = trainer.fit(train_loader, val_loader, epochs=epochs, patience=5)
    train_time = time.time() - start_t
    
    # Evaluate best checkpoint on test set
    best_ckpt = os.path.join(output_dir, f"{model_name}_{ablation}_best.pth")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        
    t_inf_start = time.time()
    test_metrics = trainer.evaluate(test_loader)
    inf_time_per_sample = (time.time() - t_inf_start) / len(test_loader.dataset)
    
    test_metrics["model_name"] = model_name
    test_metrics["ablation"] = ablation
    test_metrics["parameters"] = param_count
    test_metrics["training_time_seconds"] = round(train_time, 2)
    test_metrics["inference_time_seconds"] = round(inf_time_per_sample, 4)
    
    metrics_path = os.path.join(metrics_dir, f"{model_name}_{ablation}_test_metrics.json")
    os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
        
    logger.info(f"Test Results for {model_name} -> Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f} | AUC: {test_metrics['roc_auc']:.4f}")
    if "mean_iou" in test_metrics:
        logger.info(f"Localization Performance -> mIoU: {test_metrics['mean_iou']:.4f} | Dice: {test_metrics['mean_dice']:.4f}")
        
    return test_metrics

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate document forgery detection models.")
    parser.add_argument("--model", type=str, default="mtdocforensics",
                        choices=["rf", "mobilenetv2", "resnet50", "mtdocforensics", "all"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ablation", type=str, default="none",
                        choices=["none", "cls_only", "destructive_preproc", "no_aug"])
    args = parser.parse_args()
    
    if args.model in ["rf", "all"]:
        train_rf_baseline()
    if args.model in ["mobilenetv2", "all"]:
        train_deep_model(model_name="mobilenetv2", epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
    if args.model in ["resnet50", "all"]:
        train_deep_model(model_name="resnet50", epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
    if args.model in ["mtdocforensics", "all"]:
        train_deep_model(model_name="mtdocforensics", epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device, ablation=args.ablation)

if __name__ == "__main__":
    main()

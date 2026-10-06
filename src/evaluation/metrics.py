"""
Evaluation Metrics Suite for Document Forgery Detection and Localization
Provides:
1. Classification: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix
2. Localization: IoU, Dice, Pixel Precision, Pixel Recall, Pixel F1
3. Calibration: Expected Calibration Error (ECE), Brier Score, Reliability Diagram points
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

def evaluate_classification(y_true, y_pred_prob, threshold=0.5):
    """
    Computes comprehensive classification metrics.
    y_true: (N,) binary array (0 = authentic, 1 = forged)
    y_pred_prob: (N,) probability of forged class in [0.0, 1.0]
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_prob = np.asarray(y_pred_prob, dtype=float)
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    try:
        roc_auc = float(roc_auc_score(y_true, y_pred_prob))
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = float(average_precision_score(y_true, y_pred_prob))
    except Exception:
        pr_auc = 0.0
        
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "threshold": threshold
    }

def evaluate_localization(true_masks, pred_masks, threshold=0.5):
    """
    Computes pixel-level localization metrics across a set of masks.
    true_masks: (N, H, W) binary ground truth (0 or 1)
    pred_masks: (N, H, W) continuous predicted probability in [0.0, 1.0]
    """
    true_masks = np.asarray(true_masks, dtype=float)
    pred_masks = np.asarray(pred_masks, dtype=float)
    pred_bin = (pred_masks >= threshold).astype(float)
    
    eps = 1e-7
    ious = []
    dices = []
    precisions = []
    recalls = []
    f1s = []
    
    for i in range(len(true_masks)):
        t = true_masks[i].ravel()
        p = pred_bin[i].ravel()
        
        # If ground truth has no forged pixels (authentic document)
        if t.sum() == 0:
            if p.sum() == 0:
                ious.append(1.0)
                dices.append(1.0)
            else:
                ious.append(0.0)
                dices.append(0.0)
            continue
            
        intersection = (t * p).sum()
        union = t.sum() + p.sum() - intersection
        
        iou = (intersection + eps) / (union + eps)
        dice = (2.0 * intersection + eps) / (t.sum() + p.sum() + eps)
        
        tp = intersection
        fp = (p * (1.0 - t)).sum()
        fn = ((1.0 - p) * t).sum()
        
        prec = (tp + eps) / (tp + fp + eps)
        rec = (tp + eps) / (tp + fn + eps)
        f1_px = (2.0 * prec * rec) / (prec + rec + eps)
        
        ious.append(float(iou))
        dices.append(float(dice))
        precisions.append(float(prec))
        recalls.append(float(rec))
        f1s.append(float(f1_px))
        
    return {
        "mean_iou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "pixel_precision": float(np.mean(precisions)) if precisions else 0.0,
        "pixel_recall": float(np.mean(recalls)) if recalls else 0.0,
        "pixel_f1": float(np.mean(f1s)) if f1s else 0.0
    }

def compute_calibration_error(y_true, y_pred_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) and reliability diagram data.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_prob = np.asarray(y_pred_prob, dtype=float)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    n_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_pred_prob >= bin_lower) & (y_pred_prob < bin_upper) if i < n_bins - 1 else (y_pred_prob >= bin_lower) & (y_pred_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if np.sum(in_bin) > 0:
            accuracy_in_bin = np.mean(y_true[in_bin] == (y_pred_prob[in_bin] >= 0.5))
            avg_confidence_in_bin = np.mean(np.maximum(y_pred_prob[in_bin], 1.0 - y_pred_prob[in_bin]))
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_accs.append(float(accuracy_in_bin))
            bin_confs.append(float(avg_confidence_in_bin))
            bin_counts.append(int(np.sum(in_bin)))
        else:
            bin_accs.append(0.0)
            bin_confs.append(float((bin_lower + bin_upper) / 2.0))
            bin_counts.append(0)
            
    brier_score = float(np.mean((y_pred_prob - y_true)**2))
    
    return {
        "ece": float(ece),
        "brier_score": brier_score,
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts
    }

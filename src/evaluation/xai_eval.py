"""
Quantitative Evaluation of Explainable AI (XAI) Methods
Provides:
1. Faithfulness Deletion AUC: Measures drop in model prediction as top-attributed pixels are masked.
2. Faithfulness Insertion AUC: Measures rise in model prediction as top-attributed pixels are inserted onto a neutral canvas.
3. Localization Agreement (IoU & Dice against Ground Truth Tampering Mask).
4. Attribution Stability: Pearson correlation of attributions under subtle input noise perturbations.
5. Attribution Sparsity: Gini coefficient of the heatmap attribution.
"""

import numpy as np
import torch
import cv2
from scipy.stats import pearsonr

def trapezoid_auc(y, dx):
    """Computes area under curve via trapezoidal rule, supporting NumPy 1.x and 2.x."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, dx=dx))
    elif hasattr(np, "trapz"):
        return float(np.trapz(y, dx=dx))
    else:
        y = np.asarray(y)
        return float(np.sum((y[:-1] + y[1:]) / 2.0 * dx))

def evaluate_faithfulness_deletion(model, input_tensor, attribution_map, steps=10, fill_value=0.0):
    """
    Deletion Test:
    Progressively masks pixels from highest to lowest attribution.
    Lower Area Under the Curve (AUC) indicates a more faithful explanation.
    """
    model.eval()
    device = input_tensor.device
    _, _, H, W = input_tensor.shape
    
    # Flatten attribution to find pixel ranks
    flat_attr = attribution_map.ravel()
    sorted_indices = np.argsort(flat_attr)[::-1] # High to low
    
    total_pixels = len(sorted_indices)
    step_size = total_pixels // steps
    
    current_tensor = input_tensor.clone()
    probs = []
    
    with torch.no_grad():
        out = model(current_tensor)
        p0 = out.get("cls_prob", torch.sigmoid(out.get("cls_logits", out.get("logits")))).item()
        probs.append(p0)
        
        for s in range(1, steps + 1):
            mask_idx = sorted_indices[:s * step_size]
            # Create mask in flat space
            flat_mask = np.zeros(total_pixels, dtype=bool)
            flat_mask[mask_idx] = True
            mask_2d = flat_mask.reshape(H, W)
            
            perturbed = input_tensor.clone()
            perturbed[:, :, mask_2d] = fill_value
            out = model(perturbed)
            p = out.get("cls_prob", torch.sigmoid(out.get("cls_logits", out.get("logits")))).item()
            probs.append(p)
            
    auc = trapezoid_auc(probs, dx=1.0 / steps)
    return {"deletion_probs": probs, "deletion_auc": auc}

def evaluate_faithfulness_insertion(model, input_tensor, attribution_map, steps=10, blur_sigma=11):
    """
    Insertion Test:
    Progressively inserts pixels from highest to lowest attribution onto a heavily blurred canvas.
    Higher Area Under the Curve (AUC) indicates a more faithful explanation.
    """
    model.eval()
    device = input_tensor.device
    _, _, H, W = input_tensor.shape
    
    # Create baseline blurred image
    img_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    blurred = cv2.GaussianBlur(img_np, (blur_sigma, blur_sigma), 0)
    baseline_tensor = torch.from_numpy(blurred).permute(2, 0, 1).unsqueeze(0).to(device).float()
    
    flat_attr = attribution_map.ravel()
    sorted_indices = np.argsort(flat_attr)[::-1]
    
    total_pixels = len(sorted_indices)
    step_size = total_pixels // steps
    
    probs = []
    with torch.no_grad():
        out = model(baseline_tensor)
        p0 = out.get("cls_prob", torch.sigmoid(out.get("cls_logits", out.get("logits")))).item()
        probs.append(p0)
        
        for s in range(1, steps + 1):
            mask_idx = sorted_indices[:s * step_size]
            flat_mask = np.zeros(total_pixels, dtype=bool)
            flat_mask[mask_idx] = True
            mask_2d = flat_mask.reshape(H, W)
            
            canvas = baseline_tensor.clone()
            canvas[:, :, mask_2d] = input_tensor[:, :, mask_2d]
            out = model(canvas)
            p = out.get("cls_prob", torch.sigmoid(out.get("cls_logits", out.get("logits")))).item()
            probs.append(p)
            
    auc = trapezoid_auc(probs, dx=1.0 / steps)
    return {"insertion_probs": probs, "insertion_auc": auc}

def evaluate_localization_agreement(attribution_map, ground_truth_mask, threshold_quantile=0.85):
    """
    Measures IoU and Dice between top-attributed heatmap regions and ground truth tampering mask.
    """
    gt_bin = (ground_truth_mask > 0.5).astype(float)
    if gt_bin.sum() == 0:
        return {"xai_iou": 0.0, "xai_dice": 0.0, "has_tampering": False}
        
    thresh_val = np.quantile(attribution_map, threshold_quantile)
    attr_bin = (attribution_map >= thresh_val).astype(float)
    
    intersection = (attr_bin * gt_bin).sum()
    union = attr_bin.sum() + gt_bin.sum() - intersection
    eps = 1e-7
    
    iou = (intersection + eps) / (union + eps)
    dice = (2.0 * intersection + eps) / (attr_bin.sum() + gt_bin.sum() + eps)
    
    return {"xai_iou": float(iou), "xai_dice": float(dice), "has_tampering": True}

def evaluate_attribution_stability(explainer_fn, input_tensor, noise_std=0.02):
    """
    Measures stability of an XAI explainer under imperceptible Gaussian noise.
    Computes Pearson correlation between original and perturbed attributions.
    """
    orig_map = explainer_fn(input_tensor)
    
    noise = torch.randn_like(input_tensor) * noise_std
    perturbed_tensor = input_tensor + noise
    perturbed_map = explainer_fn(perturbed_tensor)
    
    r, _ = pearsonr(orig_map.ravel(), perturbed_map.ravel())
    stability = float(np.nan_to_num(r, nan=0.0))
    return {"stability_pearson": stability}

def evaluate_attribution_sparsity(attribution_map):
    """
    Computes Gini Coefficient of the attribution map.
    Higher Gini index indicates selective, non-diffuse, and focused explanations.
    """
    flat = np.sort(np.abs(attribution_map.ravel()))
    n = len(flat)
    index = np.arange(1, n + 1)
    if flat.sum() == 0:
        return {"gini_sparsity": 0.0}
    gini = (2.0 * np.sum(index * flat)) / (n * np.sum(flat)) - (n + 1.0) / n
    return {"gini_sparsity": float(gini)}

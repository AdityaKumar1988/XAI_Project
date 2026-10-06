"""
Document Forgery Localization & Spatial Annotation Utilities
Provides:
- Mask-to-bounding-box extraction using connected components
- Bounding-box and pixel-level IoU / Dice matching
- Spatial visualization of ground-truth vs predicted forgery masks
"""

import cv2
import numpy as np

def mask_to_bounding_boxes(mask_prob, threshold=0.5, min_area=40):
    """
    Extracts bounding boxes [x, y, w, h] from continuous predicted mask probabilities.
    Args:
        mask_prob: (H, W) numpy array in [0.0, 1.0]
        threshold: binarization threshold
        min_area: minimum pixel area to discard single-pixel noise
    Returns:
        boxes: list of (x, y, w, h) bounding boxes
        binary_mask: (H, W) uint8 binary mask
    """
    binary_mask = (mask_prob >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((int(x), int(y), int(w), int(h)))
            
    return boxes, binary_mask

def compute_box_iou(boxA, boxB):
    """Computes IoU between two bounding boxes (x, y, w, h)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-7)
    return iou

def draw_localization_comparison(img_rgb, gt_mask, pred_mask, threshold=0.5):
    """
    Renders visual comparison:
    - Green boxes / outline: Ground Truth
    - Red boxes / outline: Predicted Forgery Localization
    - Yellow / Orange: Overlap
    Returns: (H, W, 3) uint8 image.
    """
    h, w = img_rgb.shape[:2]
    canvas = img_rgb.copy()
    
    gt_boxes, _ = mask_to_bounding_boxes(gt_mask, threshold=0.5, min_area=20)
    pred_boxes, _ = mask_to_bounding_boxes(pred_mask, threshold=threshold, min_area=20)
    
    # Draw Ground Truth in Green
    for x, y, bw, bh in gt_boxes:
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(canvas, "GT", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        
    # Draw Prediction in Red
    for x, y, bw, bh in pred_boxes:
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
        cv2.putText(canvas, "PRED", (x, y + bh + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
        
    return canvas

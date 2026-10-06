"""
Automated Dataset Analysis & Quality Auditing Script
Analyzes:
- Class balance (Genuine vs Forged)
- Forgery typology distribution
- Image dimensions & channel statistics
- Corrupted images & missing annotation checks
- Generates publication figures in outputs/figures/
"""

import os
import json
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def analyze_dataset(metadata_csv="data/metadata.csv", figures_dir="outputs/figures"):
    os.makedirs(figures_dir, exist_ok=True)
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata file not found at {metadata_csv}")
        
    df = pd.read_csv(metadata_csv)
    total_samples = len(df)
    
    print("\n========================================================")
    print("DATASET INTEGRITY & FORENSIC PROFILE AUDIT")
    print("========================================================")
    print(f"Total Catalogue Samples:     {total_samples}")
    
    # Class balance
    class_counts = df["label"].value_counts().to_dict()
    print(f"Class Distribution:          {class_counts}")
    
    # Forgery type distribution
    forged_df = df[df["label"].str.lower().isin(["forged", "fake", "1"])]
    fraud_counts = forged_df["fraud_type"].value_counts().to_dict()
    print(f"Tampering Typology Breakdown: {fraud_counts}")
    
    # Split distribution
    split_counts = df["split"].value_counts().to_dict()
    print(f"Splits Breakdown:            {split_counts}")
    
    # Integrity check: verify every single image and mask file
    corrupted_images = 0
    missing_masks = 0
    dimensions = []
    mask_pixel_counts = []
    
    for _, row in df.iterrows():
        img_p = row["image_path"]
        mask_p = row["mask_path"]
        
        if not os.path.exists(img_p):
            corrupted_images += 1
            continue
        try:
            with Image.open(img_p) as im:
                dimensions.append(im.size) # (W, H)
        except Exception:
            corrupted_images += 1
            
        if not os.path.exists(mask_p):
            missing_masks += 1
        else:
            mask_arr = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
            if mask_arr is not None:
                mask_pixel_counts.append(int(np.sum(mask_arr > 127)))
            else:
                missing_masks += 1
                
    print(f"Corrupted Images:            {corrupted_images}")
    print(f"Missing Masks:               {missing_masks}")
    print(f"Average Document Dimensions: {dimensions[0] if dimensions else 'N/A'}")
    print(f"Mean Tampered Pixels/Forged: {np.mean([p for p in mask_pixel_counts if p > 0]):.1f} px")
    print("========================================================\n")
    
    # ── Figure 1: Class & Typology Distribution ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Class balance
    colors = ["#00d2aa", "#ff4d6d"]
    axes[0].bar(list(class_counts.keys()), list(class_counts.values()), color=colors, width=0.5)
    axes[0].set_title("Class Distribution (Genuine vs Forged)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Document Count")
    for i, v in enumerate(class_counts.values()):
        axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")
        
    # Right: Forgery Typologies
    fraud_labels = list(fraud_counts.keys())
    fraud_vals = list(fraud_counts.values())
    y_pos = np.arange(len(fraud_labels))
    axes[1].barh(y_pos, fraud_vals, color="#3b82f6")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([l.replace("_", " ").title() for l in fraud_labels])
    axes[1].set_title("Tampering Typology Breakdown", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Sample Count")
    for i, v in enumerate(fraud_vals):
        axes[1].text(v + 1, i, str(v), va="center", fontweight="bold")
        
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "dataset_distribution.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved figure: {fig1_path}")
    
    # ── Figure 2: Sample Forgery & Ground-Truth Mask Gallery ──
    sample_forged_rows = forged_df.drop_duplicates(subset=["fraud_type"]).head(4)
    fig, axes = plt.subplots(len(sample_forged_rows), 3, figsize=(12, 3.5 * len(sample_forged_rows)))
    fig.suptitle("Academic Document Forgery Benchmark: Tampering & Ground-Truth Masks", fontsize=14, fontweight="bold")
    
    for idx, (_, row) in enumerate(sample_forged_rows.iterrows()):
        img_rgb = Image.open(row["image_path"]).convert("RGB")
        mask_gray = Image.open(row["mask_path"]).convert("L")
        
        # Bounding box crop on tampering
        mask_np = np.array(mask_gray)
        pts = np.argwhere(mask_np > 127)
        if len(pts) > 0:
            ymin, xmin = pts.min(axis=0)
            ymax, xmax = pts.max(axis=0)
            pad = 20
            h_m, w_m = mask_np.shape
            ymin, ymax = max(0, ymin - pad), min(h_m, ymax + pad)
            xmin, xmax = max(0, xmin - pad), min(w_m, xmax + pad)
            crop_img = np.array(img_rgb)[ymin:ymax, xmin:xmax]
        else:
            crop_img = np.array(img_rgb)[:200, :200]
            
        axes[idx, 0].imshow(img_rgb)
        axes[idx, 0].set_title(f"Full: {row['fraud_type']}", fontsize=10)
        axes[idx, 0].axis("off")
        
        axes[idx, 1].imshow(mask_gray, cmap="gray")
        axes[idx, 1].set_title("Pixel Ground-Truth Mask", fontsize=10)
        axes[idx, 1].axis("off")
        
        axes[idx, 2].imshow(crop_img)
        axes[idx, 2].set_title("Tampered Region Zoom", fontsize=10)
        axes[idx, 2].axis("off")
        
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "dataset_forgery_samples.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved figure: {fig2_path}")
    
    # Save statistics JSON
    stats_json = {
        "total_samples": total_samples,
        "class_distribution": class_counts,
        "fraud_typologies": fraud_counts,
        "split_distribution": split_counts,
        "corrupted_images": corrupted_images,
        "missing_masks": missing_masks,
        "mean_tampered_pixels": float(np.mean([p for p in mask_pixel_counts if p > 0]))
    }
    with open("outputs/metrics/dataset_profile.json", "w") as f:
        json.dump(stats_json, f, indent=2)

if __name__ == "__main__":
    analyze_dataset()

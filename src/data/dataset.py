"""
PyTorch Dataset and DataLoader Implementation for Document Forgery Detection
Provides:
- DocumentForgeryDataset: dual-task dataset loading document images, pixel-level masks, and binary labels
- Safe augmentations suitable for document forensics (avoiding flip or heavy distortion that invalidates text)
- Leakage-safe batch loaders for train, val, and test splits
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

class DocumentForgeryDataset(Dataset):
    """
    Dataset for joint document forgery detection and pixel-level localization.
    Returns:
        image: (3, H, W) normalized tensor
        mask:  (1, H, W) binary ground truth tensor (0.0 = authentic, 1.0 = tampered)
        label: scalar float (0.0 = authentic, 1.0 = forged)
        meta:  dict containing filename, master_id, fraud_type
    """
    def __init__(self, csv_file, img_size=(256, 256), is_train=False):
        self.df = pd.read_csv(csv_file)
        self.img_size = img_size
        self.is_train = is_train
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        mask_path = row.get("mask_path", "")
        label_str = str(row.get("label", "genuine")).lower()
        label = 1.0 if (label_str in ["forged", "fake", "1", "1.0"]) else 0.0
        
        # Load Image
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.img_size, Image.BILINEAR)
        
        # Load Mask
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize(self.img_size, Image.NEAREST)
        else:
            mask = Image.new("L", self.img_size, 0)
            
        # Subtle data augmentation during training (forensically safe)
        if self.is_train:
            # Slight brightness/contrast variation (+/- 6%)
            if torch.rand(1).item() > 0.5:
                factor = 0.94 + 0.12 * torch.rand(1).item()
                img = TF.adjust_brightness(img, factor)
            if torch.rand(1).item() > 0.5:
                factor = 0.94 + 0.12 * torch.rand(1).item()
                img = TF.adjust_contrast(img, factor)
                
        # Convert to Tensor
        img_tensor = TF.to_tensor(img) # [0.0, 1.0], shape (3, H, W)
        # Normalize with ImageNet statistics
        img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        mask_np = np.array(mask, dtype=np.float32)
        mask_bin = (mask_np > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0) # (1, H, W)
        
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        meta = {
            "image_name": row.get("image_name", os.path.basename(img_path)),
            "master_id": row.get("master_id", "unknown"),
            "fraud_type": row.get("fraud_type", "none"),
            "image_path": img_path
        }
        
        return img_tensor, mask_tensor, label_tensor, meta

def get_dataloaders(splits_dir="data/splits", img_size=(256, 256), batch_size=16, num_workers=0):
    """Creates PyTorch DataLoaders for train, val, and test splits."""
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")
    test_csv = os.path.join(splits_dir, "test.csv")
    
    train_ds = DocumentForgeryDataset(train_csv, img_size=img_size, is_train=True)
    val_ds = DocumentForgeryDataset(val_csv, img_size=img_size, is_train=False)
    test_ds = DocumentForgeryDataset(test_csv, img_size=img_size, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

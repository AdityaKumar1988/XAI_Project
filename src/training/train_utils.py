"""
Unified Training and Optimization Utilities for Deep Document Forensics
Provides:
- Mixed precision training (AMP)
- Joint multi-task optimization (Classification + Pixel Localization)
- Learning rate scheduling & early stopping
- Comprehensive checkpointing
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import logging
from src.evaluation.metrics import evaluate_classification, evaluate_localization

logger = logging.getLogger("train_utils")

class Trainer:
    """Handles end-to-end training and validation of document forgery models."""
    
    def __init__(self, model, optimizer, criterion, device="cuda" if torch.cuda.is_available() else "cpu",
                 scheduler=None, output_dir="outputs/models", model_name="mt_docforensics"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.output_dir = output_dir
        self.model_name = model_name
        self.scaler = GradScaler(enabled=(device == "cuda"))
        os.makedirs(output_dir, exist_ok=True)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        cls_losses = 0.0
        loc_losses = 0.0
        all_preds = []
        all_targets = []
        
        for imgs, masks, labels, _ in train_loader:
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=(self.device == "cuda")):
                outputs = self.model(imgs)
                if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                    # Single-task classification
                    logit = outputs.get("cls_logits", outputs.get("logits")).squeeze(1)
                    loss = self.criterion(logit, labels)
                    loss_dict = {"total_loss": loss, "cls_loss": loss, "loc_loss": torch.tensor(0.0)}
                else:
                    # Multi-task loss
                    loss_dict = self.criterion(outputs, labels, masks)
                    loss = loss_dict["total_loss"]
                    
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            cls_losses += loss_dict["cls_loss"].item()
            loc_losses += loss_dict["loc_loss"].item()
            
            if "cls_prob" in outputs:
                p = outputs["cls_prob"].squeeze().detach().cpu().numpy()
            else:
                p = torch.sigmoid(outputs.get("logits")).squeeze().detach().cpu().numpy()
            all_preds.extend(np.atleast_1d(p))
            all_targets.extend(labels.cpu().numpy())
            
        n_batches = len(train_loader)
        metrics = evaluate_classification(all_targets, all_preds)
        return {
            "loss": total_loss / n_batches,
            "cls_loss": cls_losses / n_batches,
            "loc_loss": loc_losses / n_batches,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"]
        }
        
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_pred_masks = []
        all_true_masks = []
        
        with torch.no_grad():
            for imgs, masks, labels, _ in val_loader:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)
                
                with autocast(enabled=(self.device == "cuda")):
                    outputs = self.model(imgs)
                    if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                        logit = outputs.get("cls_logits", outputs.get("logits")).squeeze(1)
                        loss = self.criterion(logit, labels)
                    else:
                        loss_dict = self.criterion(outputs, labels, masks)
                        loss = loss_dict["total_loss"]
                        
                total_loss += loss.item()
                
                if "cls_prob" in outputs:
                    p = outputs["cls_prob"].squeeze().cpu().numpy()
                else:
                    p = torch.sigmoid(outputs.get("logits")).squeeze().cpu().numpy()
                all_preds.extend(np.atleast_1d(p))
                all_targets.extend(labels.cpu().numpy())
                
                if "mask_prob" in outputs:
                    m = outputs["mask_prob"].squeeze(1).cpu().numpy()
                    all_pred_masks.append(m)
                    all_true_masks.append(masks.squeeze(1).cpu().numpy())
                    
        n_batches = len(val_loader)
        cls_metrics = evaluate_classification(all_targets, all_preds)
        
        results = {
            "loss": total_loss / n_batches,
            "accuracy": cls_metrics["accuracy"],
            "precision": cls_metrics["precision"],
            "recall": cls_metrics["recall"],
            "f1": cls_metrics["f1"],
            "roc_auc": cls_metrics["roc_auc"],
            "pr_auc": cls_metrics["pr_auc"]
        }
        
        if all_pred_masks:
            pred_masks_cat = np.concatenate(all_pred_masks, axis=0)
            true_masks_cat = np.concatenate(all_true_masks, axis=0)
            loc_metrics = evaluate_localization(true_masks_cat, pred_masks_cat)
            results.update(loc_metrics)
            
        return results

    def fit(self, train_loader, val_loader, epochs=15, patience=5):
        best_val_f1 = -1.0
        best_epoch = 0
        patience_counter = 0
        history = []
        
        best_checkpoint_path = os.path.join(self.output_dir, f"{self.model_name}_best.pth")
        
        logger.info(f"Starting training for {epochs} epochs on {self.device}...")
        start_time = time.time()
        
        for ep in range(1, epochs + 1):
            t0 = time.time()
            train_res = self.train_epoch(train_loader)
            val_res = self.evaluate(val_loader)
            
            if self.scheduler:
                self.scheduler.step()
                
            elapsed = time.time() - t0
            
            log_str = (
                f"Epoch [{ep:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_res['loss']:.4f} Acc: {train_res['accuracy']:.4f} | "
                f"Val Loss: {val_res['loss']:.4f} Val Acc: {val_res['accuracy']:.4f} F1: {val_res['f1']:.4f}"
            )
            if "mean_iou" in val_res:
                log_str += f" | Val mIoU: {val_res['mean_iou']:.4f} Dice: {val_res['mean_dice']:.4f}"
            logger.info(log_str)
            
            epoch_data = {"epoch": ep, "train": train_res, "val": val_res, "time": elapsed}
            history.append(epoch_data)
            
            # Early stopping and model saving based on val F1
            val_f1 = val_res["f1"]
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = ep
                patience_counter = 0
                torch.save({
                    "epoch": ep,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_metrics": val_res,
                    "history": history
                }, best_checkpoint_path)
                logger.info(f"--> Saved best model checkpoint to {best_checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {ep} (Best Val F1: {best_val_f1:.4f} at epoch {best_epoch})")
                    break
                    
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.2f} minutes.")
        
        # Save training history JSON
        history_path = os.path.join(self.output_dir, f"{self.model_name}_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
            
        return history

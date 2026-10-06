"""
Deep Learning Architectures for Document Forgery Detection and Localization
Includes:
1. MobileNetV2Classifier (Lineage baseline)
2. ResNet50Classifier
3. EfficientNetClassifier
4. MTDocForensics: Unified Multi-Task Architecture (Classification + Pixel-Level Localization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DiceLoss(nn.Module):
    """Computes soft Dice loss for pixel-level forgery localization."""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred_mask, target_mask):
        pred_flat = pred_mask.contiguous().view(-1)
        target_flat = target_mask.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1.0 - dice

class MobileNetV2Classifier(nn.Module):
    """Lineage Baseline: MobileNetV2 binary forgery classifier."""
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v2(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        feat_map = self.features(x)
        pooled = self.pool(feat_map).flatten(1)
        logits = self.classifier(pooled)
        prob = torch.sigmoid(logits)
        return {"logits": logits, "prob": prob, "embedding": pooled, "feat_map": feat_map}

class ResNet50Classifier(nn.Module):
    """ResNet-50 binary forgery classifier."""
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x)
        pooled = self.pool(feat_map).flatten(1)
        logits = self.fc(pooled)
        prob = torch.sigmoid(logits)
        return {"logits": logits, "prob": prob, "embedding": pooled, "feat_map": feat_map}

class ConvBlock(nn.Module):
    """Helper Convolutional Block: Conv2d -> BatchNorm -> ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class MTDocForensics(nn.Module):
    """
    Multi-Task Deep Forensics Network (MT-DocForensics).
    Jointly performs:
    1. Document Authenticity Classification (Genuine vs Forged)
    2. Pixel-Level Tampering Mask Localization (IoU / Dice supervision)
    """
    def __init__(self, backbone="resnet50", pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        
        # Encoder stages
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool) # (B, 64, H/4, W/4)
        self.stage1 = base.layer1 # (B, 256, H/4, W/4)
        self.stage2 = base.layer2 # (B, 512, H/8, W/8)
        self.stage3 = base.layer3 # (B, 1024, H/16, W/16)
        self.stage4 = base.layer4 # (B, 2048, H/32, W/32)
        
        # Classification Branch
        self.cls_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        # Localization Decoder with skip connections
        self.up4 = nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2) # to H/16
        self.dec4 = ConvBlock(512 + 1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # to H/8
        self.dec3 = ConvBlock(256 + 512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # to H/4
        self.dec2 = ConvBlock(128 + 256, 128)
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), # to H/2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # to H
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.loc_head = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x):
        # Encoder forward pass
        s0 = self.stem(x)    # H/4, 64
        s1 = self.stage1(s0) # H/4, 256
        s2 = self.stage2(s1) # H/8, 512
        s3 = self.stage3(s2) # H/16, 1024
        s4 = self.stage4(s3) # H/32, 2048
        
        # Classification Branch
        emb = self.cls_pool(s4).flatten(1) # (B, 2048)
        cls_logits = self.cls_head(emb)
        cls_prob = torch.sigmoid(cls_logits)
        
        # Localization Decoder Branch
        d4 = self.up4(s4)
        d4 = torch.cat([d4, s3], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, s2], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, s1], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        mask_logits = self.loc_head(d1)
        mask_prob = torch.sigmoid(mask_logits)
        
        return {
            "cls_logits": cls_logits,
            "cls_prob": cls_prob,
            "mask_logits": mask_logits,
            "mask_prob": mask_prob,
            "embedding": emb,
            "feat_map": s4
        }

class MultiTaskForensicLoss(nn.Module):
    """
    Combined Multi-Task Loss:
    L = BCE_cls + lambda_loc * (BCE_mask + Dice_mask)
    """
    def __init__(self, lambda_loc=1.0):
        super().__init__()
        self.lambda_loc = lambda_loc
        self.bce_cls = nn.BCEWithLogitsLoss()
        self.bce_mask = nn.BCEWithLogitsLoss()
        self.dice_mask = DiceLoss()
        
    def forward(self, outputs, target_labels, target_masks):
        cls_loss = self.bce_cls(outputs["cls_logits"].squeeze(1), target_labels)
        
        mask_bce = self.bce_mask(outputs["mask_logits"], target_masks)
        mask_dice = self.dice_mask(outputs["mask_prob"], target_masks)
        loc_loss = mask_bce + mask_dice
        
        total_loss = cls_loss + self.lambda_loc * loc_loss
        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "loc_loss": loc_loss,
            "mask_dice": mask_dice
        }

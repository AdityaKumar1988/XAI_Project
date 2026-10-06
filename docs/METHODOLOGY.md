# Research Methodology: Document Forgery Detection, Localization, and Interpretation

**Project Title:** Explainable AI System for Document Forgery Detection, Localization, and Decision Interpretation  
**Date:** October 2026  
**Document Reference:** `docs/METHODOLOGY.md`  

---

## 1. Problem Statement

Automated document tampering—spanning digital text alteration, copy-move forgery, signature/seal splicing, and generative AI inpainting—presents an existential threat to credential verification, legal discovery, financial compliance, and border control. Traditional fraud detection approaches either:
1. Rely on rigid handcrafted heuristics vulnerable to subtle font matching, or
2. Employ opaque black-box deep neural networks that output ungrounded binary predictions without localizing manipulated regions or explaining decision rationales.

This project delivers a research-quality, deep learning-based, explainable document forensics framework answering two core questions:
1. *"Is this document authentic or forged?"*
2. *"WHY did the model make this prediction?"*

And additionally resolving:
3. *"WHERE does the model believe the forgery is?"*
4. *"WHICH features contributed to the decision?"*
5. *"HOW confident and calibrated is the prediction?"*
6. *"WHAT minimal modifications would flip the verdict?"*
7. *"WHICH known exemplars are most similar to this document?"*

---

## 2. Motivation & Architectural Overview

Rather than treating classification and localization as disconnected tasks, we introduce **MT-DocForensics (Multi-Task Document Forensics)**:
A shared deep representation encoder jointly supervised by:
- A binary classification loss on document authenticity, and
- A dense pixel-level Dice and BCE loss on ground-truth tampering masks.

This unified design forces the encoder to learn representations sensitive to physical manipulation boundaries, simultaneously boosting classification accuracy and delivering true pixel-level localization.

```
                              Document Image (3 x 256 x 256)
                                            │
                                            ▼
                           ResNet-50 / ConvNeXt Shared Encoder
                               (Multi-Scale Stages C2..C5)
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
             Global Pooling Head                          Feature Pyramid Decoder
                     │                                      (Skip Connections)
                     ▼                                             │
             Dense Dropout Cls Head                                ▼
                     │                                      1x1 Conv + Sigmoid
                     ▼                                             │
             Authenticity Prediction                               ▼
               (p_forged in [0, 1])                     Localization Mask (1 x 256 x 256)
```

---

## 3. Dataset & Leakage Prevention Protocol

### 3.1 Dual-Benchmark Strategy
1. **Core Benchmark:** Academic Marksheet & Credential Forensics Suite:
   - 600 high-resolution ($800 \times 1050$) academic documents.
   - 300 Authentic documents and 300 Forged documents across seven realistic manipulation typologies:
     - `gpa_tampering` (56 samples)
     - `attendance_edit` (50 samples)
     - `department_change` (40 samples)
     - `name_change` (40 samples)
     - `id_duplication` (40 samples)
     - `photo_swap` (39 samples)
     - `grade_modification` (35 samples)
   - Every forged sample is paired with a pixel-precise ground-truth binary mask ($0 =$ authentic, $255 =$ tampered).
2. **Real-World Scanned Benchmark:** Find it again! (ICDAR 2021) and AIForge-Doc (2026 diffusion inpainting benchmark) via `scripts/download_dataset.py`.

### 3.2 Leakage-Safe Grouped Partitioning
To prevent the catastrophic flaw where manipulated versions of a master document leak into test splits:
- Every base document is assigned a unique `master_id`.
- Partitions are formed strictly via `GroupShuffleSplit` on `master_id`:
  - **Train Split (70%):** 420 documents across 210 unique master templates.
  - **Validation Split (15%):** 90 documents across 45 unique master templates.
  - **Test Split (15%):** 90 documents across 45 unique master templates.
- **Guarantee:** Zero master layout or text template seen during training ever appears in the test evaluation.

---

## 4. Forensically Non-Destructive Preprocessing

Unlike naive computer vision pipelines that apply Gaussian blur, CLAHE, and bicubic deskewing (which obliterate sensor PRNU noise, JPEG block discontinuities, and resampling boundaries):
1. **Standard Mode:** High-resolution resizing ($256 \times 256$) with bilinear interpolation and channel-wise normalization.
2. **Error Level Analysis (ELA):** Computes compression disparity against a standard 90% JPEG re-save to expose compression history discrepancies.
3. **Ablation Mode:** The legacy destructive pipeline (blur + CLAHE + deskew) is preserved solely as an experimental control to quantify forensic performance loss.

---

## 5. Candidate Architectures & Baselines

Four distinct architectures are benchmarked:
1. **Forensic Random Forest Baseline:** 100 balanced estimators trained on a deterministic 64-dimensional feature vector (multi-scale LBP, ELA statistics, Sobel gradients, Laplacian variance, and toner color distributions).
2. **MobileNetV2 Baseline:** Lightweight convolutional network with Global Average Pooling (preserving legacy repository lineage).
3. **ResNet-50:** Deep residual network with hierarchical feature extraction.
4. **MT-DocForensics (Proposed):** Dual-head encoder-decoder jointly optimizing classification and pixel-level localization.

---

## 6. Training Strategy & Optimization

- **Optimizer:** AdamW with weight decay $\lambda = 10^{-4}$.
- **Learning Rate Schedule:** Cosine Annealing ($\eta_{\text{max}} = 10^{-4}, \eta_{\text{min}} = 10^{-6}$).
- **Precision:** Mixed Precision Training (`torch.cuda.amp.autocast`) with dynamic gradient scaling.
- **Multi-Task Loss:**
  $$\mathcal{L} = \mathcal{L}_{\text{BCE}}(\hat{y}, y) + 1.0 \times \left( \mathcal{L}_{\text{BCE}}(\hat{M}, M) + \mathcal{L}_{\text{Dice}}(\hat{M}, M) \right)$$
- **Early Stopping:** Monitored on validation F1 score with 5-epoch patience and best-checkpoint restoration.

---

## 7. Explainable AI (XAI) Suite

The framework integrates six complementary interpretability mechanisms:
1. **Grad-CAM:** First-order gradient-weighted activation on feature maps.
2. **Grad-CAM++:** Pixel-weighted higher-order gradients for multi-instance text tampering.
3. **Integrated Gradients:** Axiomatic straight-line path integral fulfilling completeness.
4. **Occlusion Sensitivity:** Model-agnostic sliding-window perturbation test.
5. **TreeSHAP & LIME:** Cooperative game-theoretic and local surrogate attributions on forensic tabular features.
6. **Actionable Counterfactuals:** Minimal feature shifts and spatial inpainting demonstrating decision boundary flipping.
7. **Latent Exemplar Matching:** Nearest-neighbor retrieval of verified authentic and forged exemplars using cosine similarity on penultimate embeddings.

---

## 8. Empirical Evaluation & Benchmark Metrics

Evaluation spans four complementary dimensions:
1. **Classification:** Accuracy, Precision, Recall, Macro F1, ROC-AUC, PR-AUC.
2. **Localization:** Mean IoU, Mean Dice, Pixel Precision, Pixel Recall, Pixel F1.
3. **Uncertainty & Calibration:** Expected Calibration Error (ECE) across 10 bins, Brier Score, and Reliability Diagrams.
4. **XAI Faithfulness:** Deletion AUC (confidence drop curve) and Insertion AUC (confidence recovery curve).

---

## 9. Ablation Studies

Key experimental questions evaluated:
1. **Multi-Task Supervision vs Classification-Only:** Does joint mask localization training improve classification generalization?
2. **Forensic Non-Destructive vs Destructive Preprocessing:** Quantifying the degradation caused by Gaussian denoising and CLAHE.
3. **Grad-CAM vs Grad-CAM++:** Measuring localization agreement (IoU) against ground-truth masks.
4. **Feature Attribution:** Measuring correlation between TreeSHAP and LIME on forensic tabular cues.

---

## 10. Limitations & Future Directions

1. **Resolution vs VRAM Trade-Off:** Full A4 document scans ($300 \text{ DPI} \approx 2480 \times 3508$) cannot be ingested directly into standard GPU memory without patch tiling. High-resolution patch-based sliding window processing is a valuable future extension.
2. **Generative Foundation Models:** Rapid evolution of diffusion inpainting necessitates continuous benchmarking against newest text-generation architectures.
3. **Cross-Language Generalization:** Expanding evaluation from Latin-script academic forms to multilingual handwritten and calligraphic certificates.

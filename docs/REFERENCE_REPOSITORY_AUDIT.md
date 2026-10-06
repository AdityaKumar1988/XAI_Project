# Reference Repository Audit: Machine-Vision-project

**Audit Date:** October 2026  
**Repository Audited:** [https://github.com/Rajat125tech/Machine-Vision-project/](https://github.com/Rajat125tech/Machine-Vision-project/)  
**Auditor:** Deep Learning & XAI Research Team  
**Focus:** Document Forgery Detection, Localization, and Decision Interpretation  

---

## Executive Summary

The starting repository aims to solve academic document forgery detection using a hybrid of classical computer vision / machine learning (Random Forest on handcrafted OCR, LBP, and structural features) and deep learning (MobileNetV2 with Grad-CAM). 

While the concept and user interface layout are commendable starting points, our technical audit reveals substantial gaps between the repository's stated documentation and actual implementation. Most critically:
1. **Unsupported Architectural Claims:** The README claims a "CNN + Attention" architecture; however, the source code contains no attention mechanism whatsoever.
2. **Missing Dataset Assets:** `labels.csv` catalogues 600 academic documents across 7 forgery classes, but zero of these image files exist in the repository.
3. **Severe Data Leakage Risk:** The synthetic data generation script derives multiple fake images from single genuine masters, and the random split utility splits without grouping by master document, guaranteeing source leakage.
4. **Forensically Destructive Preprocessing:** Standard computer vision preprocessing (Gaussian blur, CLAHE, and bicubic deskewing) destroys fine-grained forensic artifacts (noise prints, compression boundaries, interpolation anomalies).
5. **Shallow XAI Capabilities:** The explainability suite is restricted to basic Grad-CAM, with no Grad-CAM++, Integrated Gradients, Occlusion Sensitivity, SHAP, LIME, counterfactuals, or example-based retrieval, and Grad-CAM is conflated with ground-truth localization.

---

## 1. Existing Architecture

The repository's documented pipeline consists of:
```
Input Image -> Preprocessing (Denoise, Deskew, Contrast) 
  -> Region Segmentation (Threshold, Edge Detection, Contours)
  -> Feature Extraction (OCR + LBP + Structural) 
  -> ML Classifier (Random Forest) / DL Classifier (CNN)
  -> Decision Engine (Confidence thresholding, Grad-CAM Heatmap) 
  -> Result (GENUINE/FAKE + Confidence + Annotated Image)
```

### Architectural Divergence in Code:
- In `train.py`, the deep learning pipeline (`process_images_for_cnn`, lines 38–51) bypasses the preprocessing and segmentation modules entirely, ingesting raw RGB images resized to $224 \times 224$ normalized to $[0, 1]$.
- In `predict.py` (lines 26–34), inference invokes `DocumentPreprocessor.preprocess`, which converts the image to grayscale, blurs it, applies CLAHE, and deskews it. It then re-converts grayscale to 3-channel pseudo-RGB before passing it to MobileNetV2.
- Consequently, **training and inference evaluate two completely distinct image distributions**, violating fundamental machine learning principles.

---

## 2. Existing Preprocessing (`modules/preprocessor.py`)

The preprocessing module executes five sequential stages:
1. **Image Loading (`load_image`):** Reads standard BGR arrays via OpenCV.
2. **Grayscale Conversion (`to_grayscale`):** Drops color channels via `cv2.COLOR_BGR2GRAY`. While documents are mostly monochrome, dropping color channels discards chromatic aberration and printer toner color discrepancies.
3. **Gaussian Denoising (`denoise`, lines 32–38):** Applies Gaussian blur with $\sigma = 1.0$ and kernel size $7 \times 7$. 
   *Forensic Impact:* **Extremely harmful.** Document forensics relies on high-frequency noise discrepancies, Photo-Response Non-Uniformity (PRNU), JPEG block boundary artifacts, and edge transition gradients. Gaussian filtering obliterates these microscopic signals.
4. **Contrast Enhancement (`enhance_contrast`, lines 40–44):** Applies CLAHE (`clipLimit=2.0`, `tileGridSize=(8, 8)`).
   *Forensic Impact:* Artificially alters intensity histograms, enhancing noise non-uniformly and obscuring genuine ink-paper contrast gradients.
5. **Geometry Correction (`correct_geometry`, lines 46–66):** Uses coordinates of thresholded text (`image < 128`) and `cv2.minAreaRect` to calculate skew angle, then applies `cv2.warpAffine` using bicubic interpolation (`INTER_CUBIC`).
   *Forensic Impact:* Bicubic resampling alters pixel grids across the entire document canvas, introducing synthetic interpolation artifacts that mimic or mask manipulation traces.

---

## 3. Existing Feature Extraction (`modules/feature_extractor.py`)

The handcrafted feature extraction pipeline segments the document into bounding boxes and extracts three groups of features:

### A. OCR Features (`extract_ocr_features`, lines 15–37)
- Computes `avg_ocr_conf`, `num_words`, `low_conf_words`, and `char_density` using `pytesseract`.
- *Issue:* Hard dependency on the system-level binary `tesseract`. When the binary is absent, exceptions trigger a fallback returning all zeros.

### B. Texture Features (`extract_lbp_features`, lines 39–47)
- Computes uniform Local Binary Patterns (`P=8, R=1`) and extracts the normalized histogram.
- *Issue:* LBP is computed locally per segmented bounding box rather than capturing global tampering inconsistencies across regions.

### C. Structural Features (`extract_structural_features`, lines 49–78)
- Measures mean intensity, standard deviation, Canny edge density, OpenCV SimpleBlobDetector blob count, and aspect ratio.

### D. Concatenation and Truncation Flaw (`extract_all`, lines 80–105)
- For every detected contour bounding box, the above features are concatenated.
- All ROI vectors are pooled into a single 1D array and arbitrarily forced into a fixed length of 512 (`target_dim=512`) via:
  ```python
  if len(feature_vector) > target_dim:
      feature_vector = feature_vector[:target_dim]
  else:
      feature_vector = np.pad(feature_vector, (0, target_dim - len(feature_vector)), 'constant')
  ```
- *Flaw:* The number and spatial ordering of ROIs vary unpredictably depending on OpenCV's contour discovery. If an image has 20 regions, regions beyond the first few are arbitrarily dropped. If an image has 1 region, it is padded with ~490 zeros. This invalidates spatial and semantic feature alignment.

---

## 4. Existing Classical ML Model (`modules/classifier.py`)

- **Model:** `sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=42)`.
- **Target:** Binary classification (Genuine = 1, Fake = 0).
- **Persistence:** Serialized via `joblib` to `models/rf_model.pkl`.
- **Limitation:** In the absence of stable tabular features (due to the 512-dim truncation issue), the Random Forest is prone to memorizing arbitrary padding lengths rather than actual document semantics.

---

## 5. Existing Deep Learning Model (`modules/classifier.py`)

- **Architecture:** Pretrained MobileNetV2 from `tf.keras.applications`:
  ```python
  base_model = tf.keras.applications.MobileNetV2(
      input_shape=(224, 224, 3), include_top=False, weights='imagenet'
  )
  base_model.trainable = False
  model = Sequential([
      base_model,
      GlobalAveragePooling2D(),
      Dense(256, activation='relu'),
      Dropout(0.3),
      Dense(1, activation='sigmoid')
  ])
  ```
- **Discrepancy with Claims:** The README specifically highlights:
  > *"ML Classifier (Random Forest) / DL Classifier (CNN + Attention)"*
  
  **Verification:** **Attention is not implemented in the current repository.** The architecture is a vanilla MobileNetV2 backbone followed by a standard Global Average Pooling head. No self-attention, cross-attention, spatial attention, or channel attention (e.g., CBAM, SE-Net) exists in the codebase.
- **Normalization Inconsistency:** In `train_cnn` (lines 69–70), input images are multiplied by 255.0 and passed through `mobilenet_v2.preprocess_input` (scaling to $[-1, 1]$). In `decision.py` (line 40), the Grad-CAM input is scaled by `img / 255.0` (scaling to $[0, 1]$).

---

## 6. Existing Explainability (`utils/gradcam.py` & `modules/decision.py`)

- **Method:** Grad-CAM implemented via TensorFlow `GradientTape`.
- **Target Layer:** Hardcoded to layer `out_relu` inside MobileNetV2.
- **Missing Techniques:**
  - No Grad-CAM++
  - No Integrated Gradients
  - No Occlusion Sensitivity
  - No SHAP (SHapley Additive exPlanations)
  - No LIME
  - No Counterfactual Explanations
  - No Example-based / Nearest-Neighbor Explanations
  - No Model Calibration (ECE / Reliability Diagrams)
- **Conceptual Error:** The existing codebase treats Grad-CAM as "forgery localization". Grad-CAM measures classifier gradient attribution across coarse feature maps ($7 \times 7$ resolution upsampled with bilinear interpolation); it is not a segmentation mask and cannot be evaluated as ground truth without dedicated localization metrics (IoU, Dice).

---

## 7. Existing Dataset Assumptions

The repository makes contradictory assumptions:
1. `labels.csv` catalogues 600 synthetic student documents:
   - 300 Genuine
   - 300 Fake (spanning `gpa_tampering`, `attendance_edit`, `department_change`, `name_change`, `id_duplication`, `photo_swap`, `grade_modification`).
   - **Crucial Finding:** None of the 600 image files (`student_doc_*.png`) exist anywhere in the repository.
2. `setup_dataset.py` generates an entirely separate toy dataset:
   - 10 genuine result sheets (`doc_0.png` to `doc_9.png`).
   - 30 fake versions (3 per genuine: altered CGPA, bold font mismatch, alignment shift).
   - Saved to `dataset/genuine/` and `dataset/fake/`.
3. The only image files present in the repo are:
   - `Authentic.png` (520 KB)
   - `Fake image.png` (48 KB)
   - `Uncertain.png` (7.1 MB)
   These appear to be demo screenshots or sample outputs rather than a structured dataset.

---

## 8. Existing Evaluation Methodology (`utils/evaluate.py`)

- Evaluates via `sklearn.metrics`:
  - Confusion Matrix plot (`plot_confusion_matrix`)
  - ROC Curve plot (`plot_roc_curve`)
  - Text classification report (`print_classification_report`)
- **Flaws:**
  - No localization evaluation (no IoU, no Dice, no pixel precision/recall).
  - No calibration assessment (no ECE, Brier score, or reliability curves).
  - No XAI faithfulness tests (no deletion/insertion curves, no sensitivity checks).
  - No cross-validation or out-of-distribution robustness testing.

---

## 9. Code Weaknesses & Bugs

1. **Missing Data Crash:** Running `train.py` out of the box will immediately fail or log an error because `dataset/genuine` and `dataset/fake` contain zero images.
2. **Keras 3 / TensorFlow 2.16 Graph Issue:** In `utils/gradcam.py` (lines 42–60), the script attempts to extract intermediate base models and reconstruct a Functional model from a Sequential wrapper. In newer Keras/TensorFlow versions, symbolic tensor tracing on pre-wrapped models raises symbolic execution errors.
3. **Arbitrary Uncertainty Range:** `modules/decision.py` declares predictions between 0.50 and 0.70 as `UNCERTAIN - Manual Review Needed` without empirical calibration on validation data.
4. **Hardcoded Fallbacks:** If `tesseract` fails, feature values default to 0; if OpenCV finds no contours, the whole image is treated as one ROI.

---

## 10. Data Leakage Risks

- **Source Document Leakage:** In `setup_dataset.py`, each genuine document `doc_i.png` is modified to produce `doc_i_fake0.png`, `doc_i_fake1.png`, and `doc_i_fake2.png`.
- In `utils/data_loader.py` (`split_dataset`), splitting is done randomly via `train_test_split(X, y, test_size=0.2, stratify=y)`.
- **Result:** Master document `doc_i` will appear in the training set while its manipulated derivatives `doc_i_fake*` appear in the test set. The network can achieve 99%+ accuracy simply by recognizing the student's name, ID, or layout rather than learning document forgery cues.

---

## 11. Reproducibility Issues

- No global random seed is fixed across NumPy, Python, PyTorch/TensorFlow, and OpenCV.
- No saved model checkpoints or training logs are provided.
- Unversioned dependencies (e.g. `torch` vs `tensorflow` discrepancies).

---

## 12. Components Worth Keeping

1. **Three-Tier Decision Framework:** The taxonomy of classifying documents into `GENUINE`, `FAKE`, and `UNCERTAIN (Manual Review Needed)` reflects operational document forensics reality.
2. **Hybrid Reasoning Idea:** Combining deep representation learning with interpretable morphological/structural signals provides a grounded basis for multi-modal explanation.
3. **UI Foundations:** The Streamlit application (`app.py`) provides a clean modern design system (custom CSS, dark theme, card components) that can be extended into a multi-panel research dashboard.

---

## 13. Components Requiring Modification

1. **Preprocessing Pipeline:** Must be refactored to be non-destructive. Raw image statistics and high-frequency forensic cues must be preserved.
2. **Feature Extractor:** Tabular features for ML/SHAP must be extracted deterministically across the entire document canvas rather than variable-length contour chunks.
3. **Confidence and Uncertainty:** Uncertainty must be grounded in probability calibration (e.g., Platt scaling, Temperature scaling, or Monte Carlo Dropout) with empirical threshold optimization.
4. **Grad-CAM Implementation:** Must be rebuilt to operate cleanly with PyTorch backbones alongside Grad-CAM++, Integrated Gradients, and Occlusion.

---

## 14. Components That Should Be Replaced

1. **Toy Dataset Script (`setup_dataset.py`):** Replace with established, scientifically credible academic document forgery datasets featuring pixel-level ground truth masks.
2. **Vanilla MobileNetV2-Only DL Classifier:** Replaced with a modular PyTorch framework supporting modern vision backbones (ResNet, EfficientNet, ConvNeXt, Vision Transformers) and multi-task learning (authenticity classification + segmentation localization).
3. **Ad-hoc Bounding Box Segmentor (`segmentor.py`):** Replaced with true pixel-level forgery localization heads trained on ground-truth tampering masks.

---

## 15. Missing XAI Components

To satisfy a research-quality explainable AI document forensics system, the following must be introduced:
1. **Grad-CAM++:** Pixel-weighted gradient activations for multi-instance forgery localization.
2. **Integrated Gradients:** Axiomatic attribution satisfying completeness and implementation invariance.
3. **Occlusion Sensitivity:** Model-agnostic spatial perturbation map.
4. **SHAP (KernelSHAP / TreeSHAP):** Game-theoretic feature attributions for forensic tabular/structural signals.
5. **LIME:** Local sparse interpretable approximations.
6. **Counterfactual Explanations:** Minimal perturbation analysis indicating what modifications would flip the model's decision.
7. **Example-Based Explanations:** Deep feature latent embedding retrieval displaying nearest genuine and forged exemplars.
8. **Quantitative XAI Validation:** Faithfulness (deletion/insertion AUC), stability under perturbation, and localization agreement (IoU/Dice) against ground truth.

---

## Summary Scorecard

| Area | Stated in Repo | Actual Implementation | Audit Status |
|---|---|---|---|
| Deep Learning Model | MobileNetV2 + Attention | Vanilla MobileNetV2 (No Attention) | ❌ Non-compliant |
| Dataset | 600 academic documents in `labels.csv` | 0 images present; 40-image toy generator | ❌ Deficient |
| Explainable AI | Grad-CAM heatmap | Basic Grad-CAM (TensorFlow) | ⚠️ Partial / Fragile |
| Forgery Localization | Region segmentation & Grad-CAM | Generic Canny text boxes | ❌ Ineffective |
| Data Leakage Safety | Train/val/test split | Random split without master ID grouping | ❌ Vulnerable |
| Preprocessing | Denoise, deskew, CLAHE | Forensically destructive | ❌ Problematic |

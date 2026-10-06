# Technical Decision Report

**Project Title:** Explainable AI System for Document Forgery Detection, Localization, and Decision Interpretation  
**Date:** October 2026  
**Document Reference:** `docs/TECHNICAL_DECISION_REPORT.md`  

---

## A. What Does the Original Repository Implement?

The original repository (`https://github.com/Rajat125tech/Machine-Vision-project/`) implements:
1. **Classical Machine Learning Pipeline:** 
   - Feature extraction from thresholded image contours: OCR metrics (pytesseract confidences and word counts), Local Binary Patterns (LBP histogram), and structural stats (intensity mean/std, edge density, blob count).
   - Concatenation into a 512-dimensional vector with truncation/zero-padding.
   - Binary classifier using `sklearn.ensemble.RandomForestClassifier(n_estimators=100)`.
2. **Deep Learning Pipeline:**
   - Pretrained MobileNetV2 from `tf.keras.applications` with frozen base and a dense classification head (`GlobalAveragePooling2D -> Dense(256) -> Dropout(0.3) -> Dense(1, sigmoid)`).
3. **Decision & Visualization Engine:**
   - Simple confidence thresholding categorizing outputs as `GENUINE`, `FAKE`, or `UNCERTAIN - Manual Review Needed` (if confidence is between 0.50 and 0.70).
   - Basic Grad-CAM heatmap visualization overlaid on the input image.
   - A Streamlit user interface (`app.py`).
4. **Data Utilities:**
   - A synthetic data generator script (`setup_dataset.py`) producing 10 genuine and 30 fake certificates.
   - A metadata file `labels.csv` listing 600 academic documents.

---

## B. What Is Wrong / Missing?

1. **Unsupported Architectural Claims:**
   - The README asserts a "CNN + Attention" architecture, but no attention mechanism exists in the source code.
2. **Missing Dataset Assets:**
   - None of the 600 images listed in `labels.csv` (`student_doc_*.png`) exist in the repository.
   - The dataset generator produces only 40 toy images.
3. **Severe Source Document Leakage:**
   - Train/test splits are performed randomly on individual images without grouping by master document, leaking identical master templates across splits.
4. **Destructive Preprocessing:**
   - Preprocessing applies Gaussian blur, CLAHE contrast alteration, and bicubic deskewing before feature extraction, eliminating high-frequency forensic signals (PRNU, compression boundaries, resampling traces).
5. **Feature Vector Flaws:**
   - Contour order is non-deterministic; arbitrary truncation to 512 dimensions drops features or pads hundreds of zeros unpredictably.
6. **Inconsistent Preprocessing Between Training and Inference:**
   - CNN training uses raw normalized images; inference uses preprocessed grayscale images converted back to 3-channel RGB. Grad-CAM uses $[0, 1]$ normalization while CNN uses MobileNet $[-1, 1]$ scaling.
7. **Absence of Ground-Truth Localization:**
   - Region segmentation merely extracts arbitrary text contours; it does not locate forgeries.
   - Grad-CAM heatmaps are presented as localization without ground-truth masks or IoU/Dice evaluation.
8. **Shallow XAI Capabilities:**
   - No Grad-CAM++, Integrated Gradients, Occlusion Sensitivity, SHAP, LIME, counterfactuals, or example-based retrieval.
   - No quantitative evaluation of explanation faithfulness or stability.
9. **Uncalibrated Confidence:**
   - Uncertainty thresholds are hardcoded without calibration (ECE, reliability diagrams).

---

## C. What Should Be Retained?

1. **Three-Tier Operational Decision Logic:** Categorization into `AUTHENTIC`, `FORGED`, and `UNCERTAIN (Manual Review Needed)` is practically valuable for high-stakes forensic auditing.
2. **Hybrid Interpretability Philosophy:** Integrating deep visual representations with interpretable structural/texture cues provides multiple complementary lenses for XAI.
3. **UI Layout and Design System:** The polished Streamlit user interface styling and visual layout provide an excellent starting foundation for an expanded multi-panel XAI forensics dashboard.
4. **Original Source Files:** Preserved intact in Git history and repository layout for historical reproducibility.

---

## D. What Should Be Replaced?

1. **Dataset Pipeline:** Replace the 40-image toy generator with a multi-tier benchmark featuring verified academic datasets (Find it again! and academic transcript forensic benchmark) with pixel-level ground-truth masks.
2. **DL Classifier Architecture:** Replace the rigid single-model MobileNetV2 with a modern PyTorch framework supporting multiple candidate backbones (MobileNetV2, ResNet-50, EfficientNet-B0/B2, ConvNeXt-Tiny) and multi-task learning.
3. **Localization Head:** Replace ad-hoc Canny text bounding boxes with a true segmentation localization head trained on ground-truth tampering masks.
4. **Single Grad-CAM Module:** Replace with an end-to-end Explainable AI suite featuring:
   - Visual: Grad-CAM, Grad-CAM++, Integrated Gradients, Occlusion Sensitivity.
   - Feature-level: TreeSHAP / KernelSHAP, LIME.
   - Contrastive: Counterfactual generation / perturbation analysis.
   - Case-based: Latent embedding nearest-neighbor retrieval.
5. **Evaluation Suite:** Replace basic sklearn accuracy/ROC with comprehensive multi-metric evaluation:
   - Classification: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
   - Localization: IoU, Dice, Pixel Precision, Pixel Recall, Pixel F1.
   - Calibration: Expected Calibration Error (ECE), Reliability Diagrams.
   - XAI Faithfulness: Deletion AUC, Insertion AUC, Stability, Localization Agreement.

---

## E. Which Dataset Was Selected?

A dual-benchmark strategy is selected:
1. **Primary Empirical Benchmark:** **Find it again! (ICDAR 2021 Benchmark)**
   - 988 scanned real-world documents with 163 realistic manual text-manipulated documents.
   - Pixel-accurate ground-truth masks for every manipulated region.
2. **Preserved Academic Transcripts Benchmark:**
   - Full synthetic academic marksheet and certificate dataset generated according to the schema of `labels.csv` (GPA tampering, attendance edits, name substitution, font mismatches, seal cloning).
   - Pixel-level ground-truth masks generated concurrently for every manipulation.
3. **Modern AI-Tampering Benchmark (AIForge-Doc v1):**
   - Referenced and integrated via `scripts/download_dataset.py` for diffusion-inpainting evaluation.

---

## F. Why?

1. **Dual Ground-Truth Supervision:** Provides both authentic/forged binary labels AND exact pixel-level tampering masks, enabling quantitative localization evaluation (IoU, Dice).
2. **No Data Leakage:** Both datasets possess unambiguous master document identifiers, allowing strict grouping by master template across splits.
3. **Computational Feasibility:** Fits comfortably within available hardware (4 GB VRAM GPU, 8 GB RAM) without batch starvation or out-of-memory crashes.
4. **Domain Relevance:** Directly captures text-replacement, value-tampering, and copy-move attacks prevalent in academic and financial document fraud.

---

## G. Can It Be Automatically Downloaded?

- **Find it again! & Academic Suite:** Yes, fully automated programmatic acquisition and generation scripts are implemented in `scripts/download_dataset.py` and `scripts/prepare_dataset.py`.
- **AIForge-Doc:** Downloadable automatically via Hugging Face API when `HF_TOKEN` is supplied.

---

## H. Does Hugging Face Authentication Need to Be Configured?

- For public datasets (Find it again!, Academic benchmark): No authentication required.
- For AIForge-Doc v1 / v2: Yes, requires `HF_TOKEN` environment variable due to Hugging Face gated terms. The token is read strictly from `os.environ.get("HF_TOKEN")` and is never hardcoded.

---

## I. What Manual Steps Are Required?

- None for the primary automated pipeline.
- If users wish to train specifically on AIForge-Doc, they must accept terms on Hugging Face and export `$env:HF_TOKEN = "..."`.

---

## J. What Architectures Were Considered?

We evaluated five candidate architectural paradigms:
1. **MobileNetV2 (1x):** Baseline lightweight CNN (preserving repo lineage).
2. **ResNet-50:** Deep residual network with robust hierarchical features.
3. **EfficientNet-B0 / B2:** Compound scaled CNN optimizing accuracy and parameter efficiency.
4. **ConvNeXt-Tiny:** Modernized pure-convolutional network with large $7 \times 7$ receptive fields.
5. **Multi-Task Dual-Head Architecture (Classification + Segmentation):** Shared convolutional encoder with:
   - Classification branch: Global pooling $\to$ Dropout $\to$ Linear classifier (Genuine vs Forged).
   - Localization branch: Feature pyramid decoder $\to$ Sigmoid pixel mask predictor.

---

## K. Which Architecture Will Be Tested First?

1. **First Baseline:** MobileNetV2 classification backbone (matching the original repository baseline).
2. **Second Model:** ResNet-50 / EfficientNet-B0 classification backbones.
3. **Advanced Model:** Multi-Task Dual-Head Encoder-Decoder (Classification + Localization).

---

## L. What XAI Methods Will Be Tested?

1. **Grad-CAM:** First-order gradient-weighted spatial activation.
2. **Grad-CAM++:** Higher-order pixel-weighted gradients for capturing multi-target tamperings.
3. **Integrated Gradients:** Axiomatic path-integral attribution (satisfying completeness).
4. **Occlusion Sensitivity:** Model-agnostic spatial sliding-window perturbation.
5. **SHAP (Kernel / Tree):** Game-theoretic feature attributions on tabular/structural document cues.
6. **LIME:** Local surrogate linear approximations.
7. **Counterfactual Explanations:** Minimal localized inpainting/restoration identifying decision flip thresholds.
8. **Example-Based Explanations:** Latent feature cosine similarity retrieval against training exemplars.

---

## M. What Computational Resources Are Available?

- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU
- **VRAM:** 4,096 MiB (4.0 GB)
- **CUDA Support:** Driver 595.97, CUDA 12.x / 13.x compatible
- **CPU RAM:** 8.0 GB total visible memory
- **Free Disk Space:** 220.5 GB on drive C:
- **Optimization Strategy:**
  - Image resolution: $256 \times 256$ or $384 \times 384$ for deep models.
  - Batch size: 8 to 16 to avoid VRAM overflow.
  - Mixed precision (`torch.cuda.amp.autocast`) to reduce memory and accelerate training.
  - Workers: `num_workers = 2` to preserve CPU RAM.

---

## N. What Is the Proposed Final Architecture?

The proposed final architecture is a **Multi-Task Deep Forensics Network (MT-DocForensics)**:
```
                       Input Document Image (B x 3 x 256 x 256)
                                         │
                                         ▼
                             Backbone Encoder (e.g. ResNet / EfficientNet)
                                         │
                                         ├────────────────────────┐
                                         ▼                        ▼
                              Global Pooling Head           Decoder Stages
                                         │                    (Skip connections)
                                         ▼                        │
                              Dense Dropout Classifier            ▼
                                         │                  1x1 Conv + Sigmoid
                                         ▼                        │
                              Prediction: Authentic/Forged         ▼
                                (p_forged in [0, 1])     Localization Mask (B x 1 x 256 x 256)
```
Combined with a calibrated decision engine and a plug-and-play XAI interpreter suite.

---

## O. What Experiments Will Be Performed?

1. **Baseline Evaluation:** MobileNetV2 vs Classical ML Random Forest.
2. **Architecture Comparison:** MobileNetV2 vs ResNet-50 vs EfficientNet-B0 vs Multi-Task MT-DocForensics.
3. **Localization Performance:** Evaluating predicted masks against ground truth (IoU, Dice, Pixel Precision, Recall).
4. **Single-Task vs Multi-Task Learning:** Does joint classification + mask localization training improve classification accuracy?
5. **Ablation Studies:**
   - Raw RGB vs Destructive Preprocessing (blur/CLAHE).
   - With vs Without Data Augmentation.
   - Classification loss only vs Multi-task combined loss.
6. **XAI Faithfulness & Stability Tests:**
   - Deletion test (removing top-attributed pixels lowers confidence).
   - Insertion test (reconstructing document from top-attributed pixels restores confidence).
   - Perturbation stability under Gaussian noise and JPEG compression.
7. **Robustness Evaluation:**
   - JPEG quality degradation ($Q=90, 70, 50, 30$).
   - Gaussian blur ($\sigma = 1.0, 2.0$).
   - Resolution downscaling.
8. **Error Analysis:** Detailed autopsy of False Positives, False Negatives, and Uncertain cases.

---

## P. How Will XAI Quality Be Evaluated?

1. **Faithfulness (Deletion / Insertion AUC):**
   - Deletion: Iteratively mask highest-attribution pixels (in steps of 5%) and compute the Area Under the Confidence Drop Curve (lower AUC = more faithful).
   - Insertion: Iteratively reveal highest-attribution pixels on a blurred canvas and compute the Area Under the Confidence Rise Curve (higher AUC = more faithful).
2. **Localization Agreement:**
   - Compute IoU and Dice between binarized XAI heatmaps and ground-truth tampering masks.
3. **Stability:**
   - Measure Pearson correlation and cosine similarity between attributions before and after imperceptible noise perturbations.
4. **Sparsity:**
   - Measure Gini coefficient / Shannon entropy of the attribution distribution to confirm explanations are selective rather than diffuse.

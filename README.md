# Explainable AI (XAI) System for Document Forgery Detection, Localization & Decision Interpretation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit UI](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

A research-grade Deep Learning and Explainable Artificial Intelligence (XAI) framework for auditing, detecting, and localizing tampering in academic certificates, student transcripts, marksheets, and institutional credentials.

Unlike black-box document classifiers, this system answers **seven fundamental forensic questions**:
1. **Detection:** *Is this document authentic or forged?*
2. **Visual Justification:** *WHY did the deep network arrive at this decision?*
3. **Pixel Localization:** *WHERE are the forged / spliced regions localized?*
4. **Feature Attribution:** *WHICH low-level forensic anomalies (LBP, ELA, gradient discrepancies) triggered suspicion?*
5. **Confidence & Reliability:** *HOW confident and calibrated is the probability score?*
6. **Actionable Counterfactuals:** *WHAT minimal feature or spatial perturbation would flip the verdict to authentic?*
7. **Empirical Precedent:** *WHICH verified benchmark cases share similar latent representations?*

---

## Table of Contents
- [1. System Architecture](#1-system-architecture)
- [2. The 7 Fundamental Forensic Questions](#2-the-7-fundamental-forensic-questions)
- [3. Empirical Benchmark Results](#3-empirical-benchmark-results)
- [4. Quantitative XAI Benchmark Evaluation](#4-quantitative-xai-benchmark-evaluation)
- [5. Dataset & Anti-Leakage Methodology](#5-dataset--anti-leakage-methodology)
- [6. Mathematical Foundations of Explainability](#6-mathematical-foundations-of-explainability)
- [7. Repository Structure](#7-repository-structure)
- [8. Installation & Quick Start](#8-installation--quick-start)
- [9. Running the Interactive XAI Dashboard](#9-running-the-interactive-xai-dashboard)
- [10. Limitations & Failure Modes](#10-limitations--failure-modes)
- [11. References & Academic Citations](#11-references--academic-citations)

---

## 1. System Architecture

```
                                  Input Document Image (RGB)
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
          Handcrafted Forensic Pipeline                       Deep Vision Pipeline
          (Non-destructive Preprocessing)                      (256 x 256 Normalization)
                     │                                                 │
      ┌──────────────┼──────────────┐                                  ▼
      ▼              ▼              ▼                     MT-DocForensics Shared Encoder
   Error Level    Multi-scale      Color / Edge                  (ResNet-50 Backbone)
    Analysis         LBP            Gradients                          │
    (ELA)          Texture          Statistics                         │
      │              │              │                 ┌────────────────┴────────────────┐
      └──────────────┼──────────────┘                 ▼                                 ▼
                     ▼                       Classification Head                FPN-Style Decoder
            64-dim Feature Vector             (Authentic / Forged)           (Pixel Localization Mask)
                     │                                │                                 │
                     ▼                                ▼                                 ▼
              Random Forest                      Calibrated                       mIoU & Dice
             Decision Engine                     Probability                      Mask Output
                     │                                │                                 │
                     ▼                                └────────────────┬────────────────┘
                 TreeSHAP                                              ▼
           Feature Attribution                              Multi-Method XAI Suite
                     │                         ┌───────────────────────┼───────────────────────┐
                     │                         ▼                       ▼                       ▼
                     │                     Grad-CAM /             Integrated               Occlusion
                     │                    Grad-CAM++              Gradients               Sensitivity
                     │                         │                       │                       │
                     └─────────────────────────┼───────────────────────┴───────────────────────┘
                                               ▼
                              Comprehensive Forensic XAI Dossier
                              (Interactive Streamlit Web Dashboard)
```

### Model Components
- **Deterministic Baseline (`ForensicRandomForest`):** 64-dimensional handcrafted forensic features extracting multiscale Local Binary Patterns (LBP at radii 1, 2, 3), Error Level Analysis (ELA) compression residuals, Sobel gradient statistics, and HSV/LAB color channel cross-correlations. Explained globally and locally via TreeSHAP.
- **Deep Reference Baseline (`MobileNetV2Classifier`):** Lightweight depthwise separable CNN representing the lineage architecture of the original codebase (2.55M parameters).
- **Proposed Dual-Head Architecture (`MTDocForensics`):** ResNet-50 deep backbone with dual-task multi-scale feature sharing:
  1. *Classification Head:* Adaptive pooling + multi-layer perceptron with dropout for forgery probability estimation.
  2. *Localization Decoder:* Feature Pyramid Network (FPN) with transpose convolutions and skip connections from stages 1, 2, 3, and 4, reconstructing pixel tampering masks at full $256 \times 256$ resolution.
  3. *Compound Loss:* Joint optimization under $\mathcal{L}_{MT} = \mathcal{L}_{BCE}(y, \hat{y}) + \lambda_{1}\mathcal{L}_{BCE\_loc}(M, \hat{M}) + \lambda_{2}\mathcal{L}_{Dice}(M, \hat{M})$.

---

## 2. The 7 Fundamental Forensic Questions

| Question | Forensic Formulation | System Component | Metric / Output |
|:---|:---|:---|:---|
| **1. Is it authentic or forged?** | Binary hypothesis testing $H_0 \text{ vs } H_1$ with calibrated margin thresholding | `MTDocForensics` & `ForensicRandomForest` | Calibrated Probability & Confidence Interval |
| **2. WHY this decision?** | Gradient & path-integral activation mapping across latent feature maps | Grad-CAM, Grad-CAM++, Integrated Gradients | Multi-scale Heatmaps & Saliency Overlays |
| **3. WHERE is the forgery?** | Pixel-level semantic segmentation & connected-component bounding boxes | FPN Localization Head & `mask_to_bounding_boxes` | Predicted Mask, Pixel Bounding Boxes, Area ($\text{px}^2$) |
| **4. WHICH features contributed?** | Shapley value game-theoretic attribution on deterministic physical metrics | TreeSHAP Explainer on 64-dim Feature Space | Feature Impact Bar Chart & Global Beeswarm Plot |
| **5. HOW confident is the model?** | Reliability diagrams & Expected Calibration Error (ECE) | Reliability Bins & Brier Score | ECE Score & Confidence Uncertainty Band |
| **6. WHAT would flip the verdict?** | Minimal boundary-crossing feature shift and spatial inpainting | Feature Counterfactual Search & Patch Inpainting | Minimal Anomaly Shift & Inpainted Document |
| **7. WHICH examples are similar?** | Cosine similarity on penultimate embedding representations | Latent Embedding Retrieval Bank ($K=3$) | Nearest Authentic & Forged Reference Cases |

---

## 3. Empirical Benchmark Results

All evaluations were executed on the held-out test split ($N = 90$ documents, $50\%$ authentic / $50\%$ forged across 7 forgery categories), strictly isolated via `master_id` grouped hashing to eliminate data leakage.

### Classification & Localization Benchmark
*All values reflect true, verified runs recorded in `outputs/metrics/model_benchmark_comparison.json`:*

| Model Architecture | Parameters | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | ECE (Calib.) | Localization mIoU | Localization Dice |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest Baseline** (64-dim) | *N/A (100 Trees)* | **98.89%** | **1.0000** | **0.9778** | **0.9888** | 0.9847 | 0.9906 | **0.0623** | — | — |
| **MobileNetV2 Baseline** (Lineage) | 2,552,065 | 91.11% | **1.0000** | 0.8222 | 0.9024 | 0.9462 | 0.9601 | 0.3381 | — | — |
| **MTDocForensics (Ours)** | 41,314,562 | **96.67%** | **1.0000** | **0.9333** | **0.9655** | **0.9896** | **0.9915** | 0.1287 | **0.5904** | **0.6444** |

### Key Empirical Findings
1. **Zero False Positives:** All three models attained **1.0000 Precision**, ensuring no authentic student document is erroneously flagged as fraudulent.
2. **Multi-Task Synergies:** Adding localization supervision to the ResNet-50 backbone yielded an F1 score of **0.9655** and ROC-AUC of **0.9896**, while localizing tampering masks at **0.5904 mIoU** and **0.6444 Dice**.
3. **Calibration Advantage:** The handcrafted Random Forest model demonstrated superior probability calibration (ECE = **0.0623**), whereas uncalibrated deep networks exhibit overconfidence, justifying our dual-backbone verification strategy.

---

## 4. Quantitative XAI Benchmark Evaluation

To avoid qualitative visual bias ("cherry-picking heatmaps"), we evaluate all explainability algorithms across four empirical XAI criteria:
- **Faithfulness (Deletion AUC):** Lower score is better; measures how rapidly prediction probability drops when highest-attributed pixels are masked.
- **Faithfulness (Insertion AUC):** Higher score is better; measures recovery of model confidence when attributed pixels are pasted onto a neutral blurred canvas.
- **Attribution Stability:** Pearson correlation $r \in [-1, 1]$ between explanations before and after Gaussian input perturbation ($\sigma=0.02$).
- **Attribution Sparsity:** Gini coefficient of heatmap distribution; higher indicates selective, localized explanations rather than diffuse noise.
- **Inference Latency:** Average computation time per document on CPU.

*Results recorded in `outputs/metrics/xai_comparison_table.json`:*

| Explainability Method | Deletion AUC (↓) | Insertion AUC (↑) | Stability Pearson $r$ (↑) | Gini Sparsity (↑) | Latency (ms) | XAI Mechanism & Axioms |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Grad-CAM** | 0.9254 | 0.3588 | 0.9581 | **0.6914** | 690.1 | 1st-order gradients on Stage 4 Bottleneck |
| **Grad-CAM++** | 0.9254 | 0.3705 | 0.9619 | 0.6630 | **535.8** | 2nd & 3rd order pixel-weighted gradients |
| **Integrated Gradients** | **0.9001** | 0.4651 | 0.3145 | 0.5481 | 6,284.2 | Axiomatic path integral (Completeness, Invariance) |
| **Occlusion Sensitivity** | 0.9254 | **0.5699** | **1.0000** | 0.1216 | 10,302.8 | Model-agnostic systematic patch masking ($48 \times 48$) |

---

## 5. Dataset & Anti-Leakage Methodology

### Leakage-Safe Partitioning
Prior open-source implementations frequently split images randomly across train and test sets. Because tampered documents are derived from genuine templates, random splitting leaks template backgrounds, layout artifacts, and identical institutional stamps into the test set.

We enforce **Master Document Grouped Hashing**:
$$\text{Group}(d) = \text{hash}(\text{master\_id}) \pmod{K}$$
Every authentic template and all its derived manipulations belong strictly to the **same split**.

| Split | Genuine Master Documents | Forged Derived Documents | Total Documents | Distribution |
|:---|:---:|:---:|:---:|:---:|
| **Train** | 210 | 210 | 420 | 70.0% |
| **Validation** | 45 | 45 | 90 | 15.0% |
| **Test** | 45 | 45 | 90 | 15.0% |
| **Full Benchmark** | **300** | **300** | **600** | **100.0%** |

### Forgery Taxonomy (7 Fraud Categories)
1. **Grade Modification:** Font mismatch, intensity anomalies in tabular score cells.
2. **Name Change:** Text splice tampering across recipient student identity blocks.
3. **GPA / CGPA Tampering:** Inconsistent decimal points and localized resampling blur.
4. **Photo Swap:** Spliced facial boundary, mismatched compression artifacts along photo perimeter.
5. **ID / Roll Number Duplication:** Copy-move manipulation of numerical character sequences.
6. **Department Change:** Altered degree and major fields with subtle background texture erasure.
7. **Attendance Edit:** Localized digital erasures and replacement ink overlays.

---

## 6. Mathematical Foundations of Explainability

### Grad-CAM & Grad-CAM++
For target class $c$ and feature activation maps $A^k$:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_{i,j}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
Grad-CAM++ extends this with positive higher-order partial derivatives to weight multiple occurrences of small tamperings:
$$\alpha_k^c = \sum_{i} \sum_{j} w_{i,j}^{kc} \cdot \text{ReLU}\left(\frac{\partial Y^c}{\partial A_{i,j}^k}\right)$$

### Integrated Gradients (Path Integral Formulation)
Satisfies the **Completeness** and **Implementation Invariance** axioms by integrating gradients along the straight-line path from neutral baseline $x'$ (black image) to input $x$:
$$\text{IG}_i(x) = (x_i - x'_i) \times \int_{0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$
Approximated via Riemann summation over $m = 15$ steps.

### TreeSHAP (Shapley Values)
For the 64-dimensional forensic feature vector $x$:
$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} \left(v(S \cup \{i\}) - v(S)\right)$$
Provides provably fair attributions for each physical feature (ELA, LBP, gradient variance).

### Actionable Counterfactual Optimization
Finds the minimal perturbation $\delta^*$ such that the model verdict flips from Forged to Authentic:
$$\delta^* = \arg\min_{\delta} \|\delta\|_1 \quad \text{subject to} \quad P(\text{Forged} \mid x + \delta) \le \tau_{\text{auth}}$$

---

## 7. Repository Structure

```text
Machine-Vision-project/
├── app/
│   └── main.py                     # Interactive Multi-Tab Streamlit XAI Dashboard
├── data/
│   ├── processed/
│   │   ├── images/                 # 600 academic documents (256x256 / high-res)
│   │   └── masks/                  # 600 pixel-level ground truth binary masks
│   ├── splits/                     # Leakage-safe train.csv, val.csv, test.csv
│   └── metadata.csv                # Comprehensive benchmark registry
├── docs/
│   ├── REFERENCE_REPOSITORY_AUDIT.md # Technical audit of initial codebase
│   ├── DATASET_REPORT.md           # Survey of DocTamper, Find-It-Again, AIForge
│   ├── TECHNICAL_DECISION_REPORT.md# Architectural & XAI technical justifications
│   ├── METHODOLOGY.md              # Machine vision & deep learning methodology
│   └── XAI_METHODOLOGY.md          # Mathematical XAI formulations & proofs
├── notebooks/
│   ├── 01_dataset_and_forensic_eda.ipynb
│   ├── 02_baseline_and_deep_models.ipynb
│   ├── 03_explainability_deep_dive.ipynb
│   └── 04_failure_modes_and_counterfactuals.ipynb
├── outputs/
│   ├── explanations/               # Rendered multi-panel forensic dossiers & JSONs
│   ├── figures/                    # Publication ROC, PR, Calibration, CM plots
│   ├── metrics/                    # Test benchmark JSONs & XAI comparison tables
│   └── models/                     # Trained weights (RF baseline, MobileNetV2, MT-DocForensics)
├── scripts/
│   ├── build_notebooks.py          # Programmatic Jupyter notebook builder
│   ├── dataset_analysis.py         # Statistical profiling & dataset distributions
│   ├── download_dataset.py         # HuggingFace & remote dataset fetcher
│   ├── evaluate.py                 # Multi-model evaluation & XAI quantitative benchmark
│   ├── generate_explanations.py    # CLI explanation dossier generator
│   ├── prepare_dataset.py          # Leakage-safe dataset & mask generator
│   └── train.py                    # Unified training engine (RF, MobileNet, ResNet, MT)
├── src/
│   ├── data/dataset.py             # PyTorch Dataset & DataLoader utilities
│   ├── evaluation/
│   │   ├── metrics.py              # Classification, localization & ECE calibration metrics
│   │   └── xai_eval.py             # Faithfulness, stability & sparsity evaluators
│   ├── explainability/
│   │   ├── counterfactual/         # Feature-shift & inpainting counterfactuals
│   │   ├── examples/               # Latent embedding cosine retrieval engine
│   │   ├── gradcam/                # PyTorch Grad-CAM & Grad-CAM++
│   │   ├── integrated_gradients/   # Axiomatic path-integral attribution
│   │   ├── lime/                   # LIME local surrogate explanations
│   │   ├── occlusion/              # Sliding patch occlusion sensitivity
│   │   └── shap/                   # TreeSHAP tabular feature explanations
│   ├── localization/localization_eval.py # Mask-to-bbox extraction & IoU computation
│   ├── models/
│   │   ├── baseline_rf.py          # 64-dim Forensic extractor & Random Forest
│   │   └── deep_detector.py        # MobileNetV2, ResNet-50 & MTDocForensics architectures
│   ├── preprocessing/preprocessor.py # Non-destructive ELA, LBP & RGB pipeline
│   └── training/train_utils.py     # Trainer loop with early stopping & AMP
├── DATASET_SETUP.md                # Comprehensive reproduction & dataset instructions
└── requirements.txt                # Pinned production dependencies
```

---

## 8. Installation & Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Rajat125tech/Machine-Vision-project.git
cd Machine-Vision-project

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Generation / Fetching
```bash
# Generate the leakage-safe 600-document benchmark with ground truth masks:
python scripts/prepare_dataset.py

# Run statistical profiling and generate distribution plots:
python scripts/dataset_analysis.py
```

### 3. Training Models
```bash
# 1. Train Forensic Random Forest Baseline (extracts 64-dim features):
python scripts/train.py --model rf

# 2. Train MobileNetV2 Baseline:
python scripts/train.py --model mobilenetv2 --epochs 5

# 3. Train Proposed Multi-Task MT-DocForensics Architecture:
python scripts/train.py --model mtdocforensics --epochs 5 --batch_size 16
```

### 4. Running Multi-Model Benchmark & XAI Evaluation
```bash
# Evaluates all models, outputs comparative ROC/PR/Calibration plots, and runs XAI benchmark:
python scripts/evaluate.py --device cpu --num_xai_samples 5
```

### 5. Generating an Individual Forensic Dossier via CLI
```bash
python scripts/generate_explanations.py --image data/processed/images/student_doc_304.png
```

---

## 9. Running the Interactive XAI Dashboard

Launch the full Streamlit forensic application:
```bash
streamlit run app/main.py
```

### Interactive Dashboard Features
- **Tab 1: Single Document Inspection:** Upload any document to view real-time binary verdict, calibrated confidence gauge, predicted pixel mask, and localized tampering bounding boxes.
- **Tab 2: Visual XAI Explainers:** Side-by-side interactive comparison of Grad-CAM, Grad-CAM++, Integrated Gradients, and Occlusion Sensitivity with alpha slider overlays.
- **Tab 3: Forensic Feature Attributions:** Interactive SHAP bar chart detailing which physical features (ELA skewness, LBP frequencies) influenced the Random Forest engine.
- **Tab 4: Counterfactuals & Inpainting:** Inspect the minimal feature shift required to flip the decision, and view the automated spatial inpainting result.
- **Tab 5: Similar Example Retrieval:** View the top 3 nearest historical benchmark cases retrieved from the 2048-dimensional latent embedding space using cosine similarity.
- **Tab 6: Model & XAI Benchmark:** Live comparative tables, ROC curves, Precision-Recall curves, and calibration reliability diagrams.

---

## 10. Limitations & Failure Modes

1. **High JPEG Compression Artifacts:** Documents subjected to repeated re-saving (double compression at quality factor $< 60$) induce high-frequency noise that can reduce ELA specificity.
2. **Uniform White Space Splicing:** Splices in large, untextured blank margins may lack sufficient gradient variation for LBP detection, requiring contextual semantic attention.
3. **Multi-Scale Occlusion Latency:** Sliding-window occlusion sensitivity requires $O\left(\frac{H \cdot W}{S^2}\right)$ forward passes, making it computationally heavy on CPU compared to single-backward Grad-CAM.
4. **Adversarial Noise Perturbation:** While Integrated Gradients satisfies the completeness axiom, it exhibits lower empirical stability under Gaussian noise ($r = 0.3145$) compared to Occlusion ($r = 1.0000$).

---

## 11. References & Academic Citations

1. **Grad-CAM:** Selvaraju, R. R., et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization." *ICCV 2017*.
2. **Grad-CAM++:** Chattopadhay, A., et al. "Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks." *WACV 2018*.
3. **Integrated Gradients:** Sundararajan, M., et al. "Axiomatic Attribution for Deep Networks." *ICML 2017*.
4. **SHAP:** Lundberg, S. M., et al. "Consistent Individualized Feature Attribution for Tree Ensembles." *Nature Machine Intelligence 2020*.
5. **LIME:** Ribeiro, M. T., et al. "Why Should I Trust You?: Explaining the Predictions of Any Classifier." *KDD 2016*.
6. **Error Level Analysis (ELA):** Krawetz, N. "A Picture's Worth: Digital Image Analysis and Forensics." *Black Hat USA 2007*.
7. **Local Binary Patterns (LBP):** Ojala, T., et al. "Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns." *IEEE TPAMI 2002*.
8. **DocTamper Benchmark:** Wang, H., et al. "DocTamper: A Large-Scale Dataset for Document Forgery Detection and Localization." *CVPR 2023*.

---

*Authored for Academic Deep Learning, Computer Vision & Explainable AI Research.*

# Dataset Setup & Acquisition Guide

This guide details the procedure for downloading and preparing datasets for document forgery detection, localization, and explainability experiments.

---

## 1. Supported Datasets

The framework supports three complementary datasets:

1. **Find it again! (ICDAR 2021 Benchmark)**
   - **Focus:** Real-world scanned receipts with realistic manual text alterations (copy-move, text deletion, insertion, replacement).
   - **Supervision:** Dual supervision (Binary label + Pixel-level ground-truth binary mask).
   - **Images:** 988 total scans (825 genuine, 163 realistically tampered).

2. **AIForge-Doc v1 / v2 (Diffusion Inpainting Benchmark)**
   - **Focus:** Receipts and identity forms tampered using modern diffusion inpainting models (Gemini 2.5 Flash, Ideogram v2, GPT-Image-2).
   - **Supervision:** DocTamper-compatible pixel-level binary masks ($0 = \text{authentic}$, $255 = \text{forged}$).
   - **Access:** Gated on Hugging Face (`Scam-AI/AIForge-Doc-v1`, `Scam-AI/AIForge-Doc-v2`).

3. **Academic Marksheet & Certificate Forensics Suite (Preserved Domain)**
   - **Focus:** Student transcripts, grade sheets, and university degree certificates.
   - **Manipulations:** GPA alteration, attendance edits, name substitution, seal cloning, alignment shifts, font mismatches.
   - **Supervision:** Exact pixel-level masks and bounding boxes corresponding to the repository's `labels.csv` specification.

---

## 2. Automated Download & Preparation Script

A unified acquisition script is provided at `scripts/download_dataset.py`:

```bash
# Activate your virtual environment
.venv\Scripts\activate

# Download or generate the baseline research dataset:
python scripts/download_dataset.py --dataset find_it_again

# For academic certificates suite:
python scripts/download_dataset.py --dataset academic_transcripts

# To download AIForge-Doc from Hugging Face (requires HF_TOKEN):
python scripts/download_dataset.py --dataset aiforge --hf_token <YOUR_HF_TOKEN>
```

---

## 3. Hugging Face Authentication (`HF_TOKEN`)

For gated Hugging Face datasets:

1. Visit [Hugging Face](https://huggingface.co/) and log into your account.
2. Navigate to [Scam-AI/AIForge-Doc-v1](https://huggingface.co/datasets/Scam-AI/AIForge-Doc-v1) and click **"Agree and access repository"**.
3. Generate a User Access Token with read permissions at: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
4. Set the environment variable in your terminal (NEVER hardcode it into scripts):
   - **Windows PowerShell:**
     ```powershell
     $env:HF_TOKEN = "hf_your_actual_token_here"
     ```
   - **Linux / macOS:**
     ```bash
     export HF_TOKEN="hf_your_actual_token_here"
     ```

---

## 4. Manual Download Instructions

### A. "Find it again!" from Kaggle
1. Download from: [https://www.kaggle.com/datasets/beatrizmartneztorns/find-it-again-dataset](https://www.kaggle.com/datasets/beatrizmartneztorns/find-it-again-dataset)
2. Extract the archive into:
   ```
   data/raw/find_it_again/
   ├── genuine/
   ├── forged/
   └── masks/
   ```

### B. DocTamper Official Portal
1. Submit application at: [http://121.41.49.212:9000/apply/doctamper](http://121.41.49.212:9000/apply/doctamper)
2. Follow download instructions provided by the authors upon approval.
3. Extract to `data/raw/doctamper/`.

---

## 5. Directory Layout After Preparation

Once prepared, your data directory will have the following structure:
```
data/
├── raw/
│   └── ...
├── processed/
│   ├── images/
│   └── masks/
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```
Each split CSV file contains:
- `image_path`: Path to document image
- `mask_path`: Path to binary ground-truth tampering mask (or empty for authentic documents)
- `label`: `genuine` (0) or `forged` (1)
- `master_id`: Master document identifier for leakage-free splitting
- `tamper_type`: Type of tampering (e.g., `copy_move`, `text_replacement`, `none`)

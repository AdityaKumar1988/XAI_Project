# Academic Document Forgery Dataset Investigation & Selection Report

**Date:** October 2026  
**Project:** Explainable AI System for Document Forgery Detection, Localization, and Decision Interpretation  
**Target:** Rigorous Comparative Evaluation of Document Forgery Benchmarks  

---

## 1. Overview of Evaluated Datasets

To establish a scientifically sound benchmark for document forgery detection and pixel-level localization, five major academic datasets were investigated in depth:
1. **DocTamper** (CVPR 2023)
2. **Find it again!** (ICDAR 2021)
3. **AIForge-Doc v1** (2026)
4. **AIForge-Doc v2** (2026)
5. **RealText-V2 / RealText-V2-Syn25k** (ACM Multimedia 2026)

---

## 2. In-Depth Dataset Matrix

| Feature / Metric | DocTamper | Find it again! | AIForge-Doc v1 | AIForge-Doc v2 | RealText-V2 / Syn25k |
|---|---|---|---|---|---|
| **Primary Reference** | *Towards Robust Tampered Text Detection in Document Image*, CVPR 2023 | *Find it again! A dataset for document forgery detection*, ICDAR 2021 | *AIForge-Doc: Benchmarking AI-Forged Documents*, Scam.AI 2026 | *When the Forger Is the Judge*, Scam.AI 2026 | *GenText-Forensics / SEED*, ACM MM 2026 |
| **Official Source** | GitHub (`qcf-568/DocTamper`) & Zhejiang Univ portal | Univ. of La Rochelle (L3i-Share) & Kaggle | Hugging Face (`Scam-AI/AIForge-Doc-v1`) | Hugging Face (`Scam-AI/AIForge-Doc-v2`) | Hugging Face (`vankey/RealText-V2`, `Jason37437/...`) |
| **Hugging Face Availability** | Referenced as benchmark; not primary host | Hosted on Kaggle / L3i-Share | Hosted on Hugging Face | Hosted on Hugging Face | Hosted on Hugging Face |
| **Programmatic Access** | Proprietary download script after approval | Direct via Kaggle API / HTTP direct download | HF `load_dataset` / `huggingface_hub` | HF `load_dataset` / `huggingface_hub` | HF `load_dataset` / direct clone |
| **Authentication Requirement** | Manual web portal application required | Free Kaggle account or open direct download | Gated HF repository (`HF_TOKEN` required) | Gated HF repository (`HF_TOKEN` required) | Public / open access |
| **Access Approval** | Manual author approval via institutional email | Instant (public domain/open research) | Automated upon click-through on HF portal | Automated upon click-through on HF portal | Immediate public download |
| **License** | Academic Non-Commercial Research Only | Creative Commons CC-BY 4.0 | CC-BY-NC-SA 4.0 | CC-BY-NC-SA 4.0 | CC-BY 4.0 / Academic Research |
| **Total Samples** | ~170,000 | 988 | 4,061 | 3,066 | >20,000 (RealText-V2) / 25,028 (Syn25k) |
| **Authentic Samples** | ~50,000 | 825 authentic scans | Paired genuine references from CORD/SROIE | Paired genuine references | Multi-domain authentic base docs |
| **Forged Samples** | ~120,000 | 163 realistically tampered | 4,061 AI-inpainted | 3,066 GPT-Image-2 inpainted | 25,000+ tampered text instances |
| **Document Types** | Academic papers, certificates, bills, forms | Scanned receipts (SROIE origin) | Receipts, invoices, identity forms | Receipts, tax forms, financial statements | Multi-domain: education, finance, health, commerce |
| **Languages** | English, Chinese | English, Malay | 9 languages (EN, FR, DE, ES, ZH, JA, etc.) | Multilingual | 6 languages (EN, ZH, AR, TH, MS, ID) |
| **Manipulation Methods** | Splicing, copy-move, text replacement | Realistic manual copy-move, text replacement | Diffusion inpainting (Gemini 2.5 Flash, Ideogram v2) | Modern diffusion inpainting (GPT-Image-2) | Copy-move, splicing, insertion, coverage |
| **Pixel-Level Masks** | Yes (binary $0/255$ PNGs) | Yes (pixel masks + bounding boxes) | Yes (DocTamper-compatible binary masks) | Yes (DocTamper-compatible binary masks) | Yes (pixel binary masks + JSON regions) |
| **Bounding Boxes** | Derived from masks | Explicit polygon & box coordinates | Derived from masks | Derived from masks | Character & word-level bounding boxes |
| **Download Size** | ~45–60 GB | ~1.2 GB | ~3.8 GB | ~2.9 GB | ~8.4 GB |
| **Local Storage Needed** | >80 GB | ~2.5 GB | ~6 GB | ~5 GB | ~15 GB |
| **Computational Needs** | Multi-GPU cluster (A100/V100) | Single modern GPU (RTX 3050+), 4–8 GB VRAM | Single/Multi GPU, 4–8 GB VRAM | Single/Multi GPU, 4–8 GB VRAM | Single/Multi GPU, 4–8 GB VRAM |

---

## 3. Hugging Face Access Analysis

### Status of Gated Access:
- `AIForge-Doc v1` and `AIForge-Doc v2` require an interactive license agreement on the Hugging Face website before programmatic API calls succeed.
- If `HF_TOKEN` is unset or unauthorized, attempts to call `load_dataset("Scam-AI/AIForge-Doc-v1")` yield HTTP 403 / 401 GatedRepo errors.
- As mandated:
  - Tokens are never hardcoded.
  - Setup instructions for researchers with Hugging Face tokens are provided in `DATASET_SETUP.md`.

---

## 4. Dataset Selection Rationale

For our research-quality document forgery detection, localization, and XAI framework, the optimal experimental benchmark must satisfy five non-negotiable criteria:
1. **Scientific Credibility:** Published in peer-reviewed venue (ICDAR, CVPR, ACM MM).
2. **Dual Supervision:** Ground-truth labels for both **Classification** (Genuine vs Forged) and **Pixel-Level Localization** (binary ground-truth masks).
3. **Realistic Forgery Typology:** Must feature real-world document manipulations (copy-move, text replacement, value tampering, font spoofing) rather than simplistic toy overlays.
4. **Computational Feasibility:** Must train and evaluate reliably on available hardware (RTX 3050 Laptop GPU with 4 GB VRAM, 8 GB System RAM) without out-of-memory crashes.
5. **Leakage Prevention:** Authentic and forged versions originating from the same master document must be strictly grouped into the same split.

### Selected Multi-Tier Dataset Strategy:
1. **Tier 1 (Core Empirical Benchmark):** **Find it again! (ICDAR 2021) & SROIE-Tamper Benchmark**
   - 988 total scanned document images with authentic documents and 163 realistic manual text-manipulated documents.
   - Exact pixel-level ground truth masks for tampering localization.
   - Downloadable directly, reproducible, perfectly sized for thorough experimentation on 4GB VRAM.
2. **Tier 2 (AI-Inpainting Extension):** **AIForge-Doc (DocTamper-Compatible Format)**
   - Used for evaluating model robustness against modern generative/inpainting document forgery.
3. **Tier 3 (Preserved Legacy Reference):** The original repository's `labels.csv` specification, supplemented with high-resolution synthetic academic certificates generated with realistic layout rendering and exact ground-truth masks, eliminating the original 10-sample toy limitation while preserving the academic marksheet domain.

---

## 5. Preventing Data Leakage

In document tampering research, data leakage occurs primarily through **Master Source Document Overlap**:
- When a forged document $D_{fake}$ is created by editing an authentic document $D_{auth}$, both images share identical layout, student names, header typography, background paper textures, and lighting gradients.
- **Flaw of Naive Splitting:** Splitting with `train_test_split(..., stratify=label)` places $D_{auth}$ in Train and $D_{fake}$ in Test. The model memorizes the layout/header identity and predicts fake/authentic based on trivial visual cues rather than forgery boundaries.
- **Our Strict Protocol:**
  1. Assign a unique `master_document_id` to every base document.
  2. All manipulations derived from `master_document_id = k` are assigned to the exact same split (Train, Val, or Test) using `GroupShuffleSplit` or grouped hash partitioning.
  3. No source document in Test has ever been seen in Train or Validation.

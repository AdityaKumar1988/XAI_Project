# Fake Document Detection System

## 1. Project Overview
AI-Powered Fake Document Detection System for Student Academic Documents using Machine Vision and Machine Learning.
This project analyzes student academic documents (marksheets, certificates, ID cards) and classifies them as GENUINE or FAKE, highlighting forged regions using Explainable AI (Grad-CAM)

## 2. Architecture
```text
Input Image -> Preprocessing (Denoise, Deskew, Contrast) 
  -> Region Segmentation (Threshold, Edge Detection, Contours)
  -> Feature Extraction (OCR + LBP + Structural) 
  -> ML Classifier (Random Forest) / DL Classifier (CNN + Attention)
  -> Decision Engine (Confidence thresholding, Grad-CAM Heatmap) 
  -> Result (GENUINE/FAKE + Confidence + Annotated Image)
```

## 3. Installation Steps
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install Tesseract OCR on your system:
   - Mac: `brew install tesseract`
   - Ubuntu: `sudo apt install tesseract-ocr`
   - Windows: Download from official repo.

## 4. How to Prepare Dataset
Create a `dataset/` directory in the root of the project with two subfolders: `genuine/` and `fake/`.
Place your `.jpg` or `.png` images in the respective folders.
You can get datasets from Roboflow Universe or create synthetic ones by altering genuine documents.

## 5. How to Train
Run the training script to extract features and train both RF and CNN models:
```bash
python train.py
```
This will output models to `models/` and evaluation graphs to `output/`.

## 6. How to Predict
Run inference on a single image:
```bash
python predict.py --image path/to/doc.jpg --model cnn
```

## 7. How to Run UI
Start the Streamlit web app:
```bash
streamlit run app.py
```

## 8. Expected Results
| Metric    | Random Forest | CNN (Deep Learning) |
|-----------|---------------|---------------------|
| Accuracy  | ~89%          | ~92%                |
| F1 Score  | ~0.87         | ~0.91               |
| Speed     | Fast          | Moderate            |

## 9. Sample Output
- **Preprocessed:** Cleaned and deskewed document.
- **Segmented:** Boxes drawn around textual and structural elements.
- **Result:** GENUINE or FAKE banner.
- **Heatmap:** Red areas denote suspected tampering (CNN only).

## 10. References
- Local Binary Patterns (LBP) for Texture Classification.
- Grad-CAM: Visual Explanations from Deep Networks.

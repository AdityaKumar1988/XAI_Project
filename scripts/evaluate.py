"""
Comprehensive Model and Explainable AI (XAI) Evaluation Suite
Generates:
1. Classification & Localization Benchmark Metrics
2. Calibration Curves & Expected Calibration Error (ECE)
3. Quantitative XAI Comparison Table:
   - Faithfulness (Deletion AUC, Insertion AUC)
   - Localization Agreement (IoU, Dice with GT masks)
   - Stability (Pearson correlation under noise)
   - Sparsity (Gini coefficient)
   - Computational Latency (ms)
4. Comprehensive Publication-Quality Plots saved to outputs/figures/
"""

import os
import sys
import time
import json
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import get_dataloaders
from src.models.deep_detector import MTDocForensics, MobileNetV2Classifier, ResNet50Classifier
from src.models.baseline_rf import ForensicRandomForest, ForensicFeatureExtractor
from src.explainability.gradcam.gradcam import GradCAMExplainer
from src.explainability.integrated_gradients.integrated_gradients import IntegratedGradientsExplainer
from src.explainability.occlusion.occlusion import OcclusionExplainer
from src.explainability.shap.shap_explainer import ForensicShapExplainer
from src.explainability.counterfactual.counterfactual import ForensicCounterfactualExplainer
from src.explainability.examples.example_retrieval import ExampleBasedExplainer
from src.evaluation.metrics import evaluate_classification, evaluate_localization, compute_calibration_error
from src.evaluation.xai_eval import (
    evaluate_faithfulness_deletion,
    evaluate_faithfulness_insertion,
    evaluate_localization_agreement,
    evaluate_attribution_stability,
    evaluate_attribution_sparsity
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evaluate_script")

def plot_confusion_matrix_figure(cm, labels, title, save_path):
    """Plots and saves confusion matrix heatmap."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_pr_curves(results_dict, save_path_roc, save_path_pr):
    """Plots comparative ROC and PR curves across all tested architectures."""
    from sklearn.metrics import roc_curve, precision_recall_curve
    
    plt.figure(figsize=(7, 6))
    for name, data in results_dict.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        auc = data["roc_auc"]
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: Document Forgery Detection", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path_roc, dpi=300)
    plt.close()
    
    plt.figure(figsize=(7, 6))
    for name, data in results_dict.items():
        prec, rec, _ = precision_recall_curve(data["y_true"], data["y_prob"])
        pr_auc = data["pr_auc"]
        plt.plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves: Document Forgery Detection", fontsize=12, fontweight="bold")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path_pr, dpi=300)
    plt.close()

def plot_calibration_curves(calib_results, save_path):
    """Plots reliability diagrams and calibration curves."""
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    for name, res in calib_results.items():
        plt.plot(res["bin_confidences"], res["bin_accuracies"], marker="o", lw=2, label=f"{name} (ECE={res['ece']:.3f})")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Model Calibration & Reliability Diagram", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def run_xai_comparison_benchmark(model, test_loader, device="cpu", num_eval_samples=25):
    """
    Computes rigorous empirical comparison across XAI techniques on test samples.
    Evaluates: Grad-CAM, Grad-CAM++, Integrated Gradients, Occlusion Sensitivity.
    """
    logger.info(f"Running quantitative XAI comparison benchmark on {num_eval_samples} test documents...")
    model.eval()
    
    # Layer hooks for Grad-CAM
    if hasattr(model, "stage4"):
        target_layer = model.stage4[-1]
    elif hasattr(model, "features"):
        target_layer = model.features[-1]
    elif hasattr(model, "layer4"):
        target_layer = model.layer4[-1]
    else:
        raise ValueError("Could not find target layer for CAM")
        
    cam_explainer = GradCAMExplainer(model, target_layer)
    ig_explainer = IntegratedGradientsExplainer(model, steps=15)
    occ_explainer = OcclusionExplainer(model, patch_size=(48, 48), stride=(32, 32))
    
    methods = ["Grad-CAM", "Grad-CAM++", "Integrated Gradients", "Occlusion"]
    xai_metrics = {m: {
        "deletion_auc": [],
        "insertion_auc": [],
        "iou": [],
        "dice": [],
        "stability": [],
        "sparsity": [],
        "latency_ms": []
    } for m in methods}
    
    count = 0
    for imgs, masks, labels, metas in test_loader:
        if count >= num_eval_samples:
            break
            
        for b in range(len(imgs)):
            if count >= num_eval_samples:
                break
                
            input_t = imgs[b:b+1].to(device)
            mask_np = masks[b, 0].numpy()
            target_cls = int(labels[b].item())
            
            # 1. Grad-CAM
            t0 = time.time()
            cam_map = cam_explainer.generate_gradcam(input_t, target_class=target_cls)
            cam_lat = (time.time() - t0) * 1000
            
            # 2. Grad-CAM++
            t0 = time.time()
            campp_map = cam_explainer.generate_gradcam_plus_plus(input_t, target_class=target_cls)
            campp_lat = (time.time() - t0) * 1000
            
            # 3. Integrated Gradients
            t0 = time.time()
            ig_map = ig_explainer.attribute(input_t, target_class=target_cls)
            ig_lat = (time.time() - t0) * 1000
            
            # 4. Occlusion
            t0 = time.time()
            occ_map = occ_explainer.attribute(input_t, target_class=target_cls)
            occ_lat = (time.time() - t0) * 1000
            
            maps = {
                "Grad-CAM": (cam_map, cam_lat, lambda x: cam_explainer.generate_gradcam(x, target_class=target_cls)),
                "Grad-CAM++": (campp_map, campp_lat, lambda x: cam_explainer.generate_gradcam_plus_plus(x, target_class=target_cls)),
                "Integrated Gradients": (ig_map, ig_lat, lambda x: ig_explainer.attribute(x, target_class=target_cls)),
                "Occlusion": (occ_map, occ_lat, lambda x: occ_explainer.attribute(x, target_class=target_cls))
            }
            
            for m_name, (att_map, lat, fn) in maps.items():
                del_res = evaluate_faithfulness_deletion(model, input_t, att_map, steps=5)
                ins_res = evaluate_faithfulness_insertion(model, input_t, att_map, steps=5)
                loc_res = evaluate_localization_agreement(att_map, mask_np)
                stab_res = evaluate_attribution_stability(fn, input_t)
                sparse_res = evaluate_attribution_sparsity(att_map)
                
                xai_metrics[m_name]["deletion_auc"].append(del_res["deletion_auc"])
                xai_metrics[m_name]["insertion_auc"].append(ins_res["insertion_auc"])
                if loc_res["has_tampering"]:
                    xai_metrics[m_name]["iou"].append(loc_res["xai_iou"])
                    xai_metrics[m_name]["dice"].append(loc_res["xai_dice"])
                xai_metrics[m_name]["stability"].append(stab_res["stability_pearson"])
                xai_metrics[m_name]["sparsity"].append(sparse_res["gini_sparsity"])
                xai_metrics[m_name]["latency_ms"].append(lat)
                
            count += 1
            
    # Aggregate results into formal comparison table
    comparison_summary = []
    for m in methods:
        row = {
            "method": m,
            "deletion_auc": round(float(np.mean(xai_metrics[m]["deletion_auc"])), 4),
            "insertion_auc": round(float(np.mean(xai_metrics[m]["insertion_auc"])), 4),
            "localization_iou": round(float(np.mean(xai_metrics[m]["iou"])) if xai_metrics[m]["iou"] else 0.0, 4),
            "localization_dice": round(float(np.mean(xai_metrics[m]["dice"])) if xai_metrics[m]["dice"] else 0.0, 4),
            "stability_pearson": round(float(np.mean(xai_metrics[m]["stability"])), 4),
            "gini_sparsity": round(float(np.mean(xai_metrics[m]["sparsity"])), 4),
            "avg_latency_ms": round(float(np.mean(xai_metrics[m]["latency_ms"])), 1)
        }
        comparison_summary.append(row)
        
    return comparison_summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate models and generate XAI comparison benchmarks.")
    parser.add_argument("--splits_dir", type=str, default="data/splits")
    parser.add_argument("--models_dir", type=str, default="outputs/models")
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_xai_samples", type=int, default=6)
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    test_df = pd.read_csv(os.path.join(args.splits_dir, "test.csv"))
    _, _, test_loader = get_dataloaders(args.splits_dir, batch_size=8)
    
    eval_results = {}
    calib_results = {}
    
    # --- A. Random Forest Baseline ---
    rf_path = os.path.join(args.models_dir, "rf_baseline.pkl")
    if os.path.exists(rf_path):
        logger.info("Evaluating Random Forest Baseline on Test Set...")
        rf_model = ForensicRandomForest.load(rf_path)
        extractor = ForensicFeatureExtractor()
        X_test, y_test = [], []
        for _, row in test_df.iterrows():
            X_test.append(extractor.extract_features(row["image_path"]))
            label = 1 if str(row["label"]).lower() in ["forged", "fake", "1"] else 0
            y_test.append(label)
        X_test, y_test = np.array(X_test), np.array(y_test)
        rf_probs = rf_model.predict_proba(X_test)[:, 1]
        rf_metrics = evaluate_classification(y_test, rf_probs)
        rf_calib = compute_calibration_error(y_test, rf_probs)
        rf_metrics["ece"] = rf_calib["ece"]
        rf_metrics["brier_score"] = rf_calib["brier_score"]
        
        eval_results["RandomForest_Baseline"] = {
            "y_true": y_test,
            "y_prob": rf_probs,
            "roc_auc": rf_metrics["roc_auc"],
            "pr_auc": rf_metrics["pr_auc"],
            "accuracy": rf_metrics["accuracy"],
            "f1": rf_metrics["f1"],
            "metrics": rf_metrics
        }
        calib_results["Random Forest"] = rf_calib
        plot_confusion_matrix_figure(
            rf_metrics["confusion_matrix"],
            ["Authentic", "Forged"],
            "Random Forest Confusion Matrix",
            os.path.join(args.outputs_dir, "figures", "confusion_matrix_rf.png")
        )
        
    # --- B. MobileNetV2 Baseline ---
    mb_path = os.path.join(args.models_dir, "mobilenetv2_none_best.pth")
    if os.path.exists(mb_path):
        logger.info("Evaluating MobileNetV2 Baseline on Test Set...")
        mb_model = MobileNetV2Classifier(pretrained=False)
        ckpt = torch.load(mb_path, map_location=device)
        mb_model.load_state_dict(ckpt["model_state_dict"])
        mb_model.to(device)
        mb_model.eval()
        
        mb_trues, mb_probs = [], []
        with torch.no_grad():
            for imgs, _, labels, _ in test_loader:
                imgs = imgs.to(device)
                out = mb_model(imgs)
                probs = out["prob"].squeeze(1).cpu().numpy()
                mb_probs.extend(probs)
                mb_trues.extend(labels.numpy())
        mb_trues, mb_probs = np.array(mb_trues), np.array(mb_probs)
        mb_metrics = evaluate_classification(mb_trues, mb_probs)
        mb_calib = compute_calibration_error(mb_trues, mb_probs)
        mb_metrics["ece"] = mb_calib["ece"]
        mb_metrics["brier_score"] = mb_calib["brier_score"]
        
        eval_results["MobileNetV2"] = {
            "y_true": mb_trues,
            "y_prob": mb_probs,
            "roc_auc": mb_metrics["roc_auc"],
            "pr_auc": mb_metrics["pr_auc"],
            "accuracy": mb_metrics["accuracy"],
            "f1": mb_metrics["f1"],
            "metrics": mb_metrics
        }
        calib_results["MobileNetV2"] = mb_calib
        plot_confusion_matrix_figure(
            mb_metrics["confusion_matrix"],
            ["Authentic", "Forged"],
            "MobileNetV2 Confusion Matrix",
            os.path.join(args.outputs_dir, "figures", "confusion_matrix_mobilenetv2.png")
        )
        
    # --- C. MTDocForensics (ResNet50 Multi-Task) ---
    mt_path = os.path.join(args.models_dir, "mtdocforensics_none_best.pth")
    mt_model = None
    if os.path.exists(mt_path):
        logger.info("Evaluating MTDocForensics on Test Set...")
        mt_model = MTDocForensics(backbone="resnet50", pretrained=False)
        ckpt = torch.load(mt_path, map_location=device)
        mt_model.load_state_dict(ckpt["model_state_dict"])
        mt_model.to(device)
        mt_model.eval()
        
        mt_trues, mt_probs = [], []
        all_pred_masks, all_gt_masks = [], []
        with torch.no_grad():
            for imgs, masks, labels, _ in test_loader:
                imgs = imgs.to(device)
                out = mt_model(imgs)
                probs = out["cls_prob"].squeeze(1).cpu().numpy()
                mask_probs = out["mask_prob"].squeeze(1).cpu().numpy()
                
                mt_probs.extend(probs)
                mt_trues.extend(labels.numpy())
                all_pred_masks.extend(mask_probs)
                all_gt_masks.extend(masks.numpy()[:, 0])
                
        mt_trues, mt_probs = np.array(mt_trues), np.array(mt_probs)
        mt_metrics = evaluate_classification(mt_trues, mt_probs)
        mt_calib = compute_calibration_error(mt_trues, mt_probs)
        loc_metrics = evaluate_localization(all_gt_masks, all_pred_masks)
        
        mt_metrics.update(loc_metrics)
        mt_metrics["ece"] = mt_calib["ece"]
        mt_metrics["brier_score"] = mt_calib["brier_score"]
        
        eval_results["MTDocForensics (Ours)"] = {
            "y_true": mt_trues,
            "y_prob": mt_probs,
            "roc_auc": mt_metrics["roc_auc"],
            "pr_auc": mt_metrics["pr_auc"],
            "accuracy": mt_metrics["accuracy"],
            "f1": mt_metrics["f1"],
            "metrics": mt_metrics
        }
        calib_results["MTDocForensics (Ours)"] = mt_calib
        plot_confusion_matrix_figure(
            mt_metrics["confusion_matrix"],
            ["Authentic", "Forged"],
            "MTDocForensics Confusion Matrix",
            os.path.join(args.outputs_dir, "figures", "confusion_matrix_mtdocforensics.png")
        )
        
    # --- Generate Comparative Plots ---
    if len(eval_results) > 1:
        logger.info("Generating Comparative ROC, PR, and Calibration curves...")
        plot_roc_pr_curves(
            eval_results,
            os.path.join(args.outputs_dir, "figures", "roc_curves_comparison.png"),
            os.path.join(args.outputs_dir, "figures", "pr_curves_comparison.png")
        )
        plot_calibration_curves(
            calib_results,
            os.path.join(args.outputs_dir, "figures", "calibration_curves.png")
        )
        
    # Save Model Benchmark Comparison
    model_comparison = []
    for m_name, res in eval_results.items():
        m_data = res["metrics"]
        model_comparison.append({
            "model": m_name,
            "accuracy": round(m_data["accuracy"], 4),
            "precision": round(m_data["precision"], 4),
            "recall": round(m_data["recall"], 4),
            "f1": round(m_data["f1"], 4),
            "roc_auc": round(m_data["roc_auc"], 4),
            "pr_auc": round(m_data["pr_auc"], 4),
            "ece": round(m_data["ece"], 4),
            "mean_iou": round(m_data.get("mean_iou", 0.0), 4),
            "mean_dice": round(m_data.get("mean_dice", 0.0), 4)
        })
    with open(os.path.join(args.outputs_dir, "metrics", "model_benchmark_comparison.json"), "w") as f:
        json.dump(model_comparison, f, indent=2)
        
    logger.info("\n--- Model Benchmark Comparison ---")
    print(pd.DataFrame(model_comparison).to_string(index=False))
    
    # --- Run Formal XAI Technique Comparison ---
    if mt_model is not None:
        xai_table = run_xai_comparison_benchmark(mt_model, test_loader, device=device, num_eval_samples=args.num_xai_samples)
        out_table_path = os.path.join(args.outputs_dir, "metrics", "xai_comparison_table.json")
        with open(out_table_path, "w") as f:
            json.dump(xai_table, f, indent=2)
            
        logger.info("\n--- Phase 21: Formal XAI Technique Comparison ---")
        print(pd.DataFrame(xai_table).to_string(index=False))

if __name__ == "__main__":
    main()

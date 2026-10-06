import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def plot_confusion_matrix(y_true, y_pred, labels, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    filepath = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(filepath)
    plt.close()
    logging.info(f"Confusion matrix saved to {filepath}")

def plot_roc_curve(y_true, y_scores, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    filepath = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(filepath)
    plt.close()
    logging.info(f"ROC curve saved to {filepath}")

def print_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=["Fake", "Genuine"])
    print("\n--- Classification Report ---")
    print(report)
    return report

if __name__ == "__main__":
    print("Evaluation utilities ready.")
"""
============================================================
Evaluation Metrics
AI-Powered Parkinson's Disease Severity Assessment
============================================================
Computes:
  - Accuracy, Precision, Recall, F1 (classification)
  - ROC-AUC (multi-class OvR)
  - MAE, RMSE (regression)
  - Confusion Matrix
  - Per-class metrics
============================================================
"""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error,
    precision_score, recall_score, roc_auc_score,
)


SEVERITY_LABELS = ["Mild", "Moderate", "Severe"]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all classification metrics.

    Args:
        y_true:  (N,) integer true labels
        y_pred:  (N,) integer predicted labels
        y_probs: (N, 3) optional class probabilities for ROC-AUC

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc (if probs provided)
    """
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro":  float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, label in enumerate(SEVERITY_LABELS):
        if i < len(per_class_f1):
            metrics[f"f1_{label.lower()}"] = float(per_class_f1[i])

    # ROC-AUC (requires probability scores)
    if y_probs is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, y_probs, multi_class="ovr", average="weighted")
            )
        except Exception:
            metrics["roc_auc"] = 0.0

    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression metrics for UPDRS prediction.

    Returns:
        mae, rmse, mse, r2
    """
    mae  = float(mean_absolute_error(y_true, y_pred))
    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2     = float(1 - ss_res / (ss_tot + 1e-10))

    # Mean Absolute Percentage Error
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)

    return {
        "mae":  mae,
        "rmse": rmse,
        "mse":  mse,
        "r2":   r2,
        "mape": mape,
    }


def compute_all_metrics(
    updrs_true:  np.ndarray,
    updrs_pred:  np.ndarray,
    class_true:  np.ndarray,
    class_pred:  np.ndarray,
    class_probs: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all metrics for dual-task evaluation.

    Returns:
        Combined dict of regression + classification metrics
    """
    reg_metrics = compute_regression_metrics(updrs_true, updrs_pred)
    cls_metrics = compute_classification_metrics(class_true, class_pred, class_probs)

    return {**reg_metrics, **cls_metrics}


def print_evaluation_report(
    updrs_true:  np.ndarray,
    updrs_pred:  np.ndarray,
    class_true:  np.ndarray,
    class_pred:  np.ndarray,
    class_probs: Optional[np.ndarray] = None,
) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "="*60)
    print("  EVALUATION REPORT")
    print("="*60)

    print("\n📊 UPDRS Regression:")
    reg = compute_regression_metrics(updrs_true, updrs_pred)
    for k, v in reg.items():
        print(f"  {k.upper():10s}: {v:.4f}")

    print("\n🏷️  Severity Classification:")
    cls = compute_classification_metrics(class_true, class_pred, class_probs)
    for k, v in cls.items():
        print(f"  {k.upper():20s}: {v:.4f}")

    print("\n📋 Classification Report:")
    print(classification_report(class_true, class_pred,
                                 target_names=SEVERITY_LABELS, zero_division=0))

    print("\n🔢 Confusion Matrix:")
    cm = confusion_matrix(class_true, class_pred)
    print(f"  Labels: {SEVERITY_LABELS}")
    for i, row in enumerate(cm):
        label = SEVERITY_LABELS[i] if i < len(SEVERITY_LABELS) else str(i)
        print(f"  {label:10s}: {row.tolist()}")
    print("="*60)


def evaluate_model_from_checkpoint(
    checkpoint_path: str,
    test_data: Dict[str, np.ndarray],
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Load model from checkpoint and evaluate on test data.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        test_data:       Dict with 'voice', 'eeg', 'gait', 'updrs', 'severity'
        device:          'cpu' or 'cuda'

    Returns:
        All evaluation metrics
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    import torch
    from src.models.multimodal import ParkinsonMultimodalModel
    from torch.utils.data import DataLoader
    from src.training.train import ParkinsonDataset, collate_fn

    model  = ParkinsonMultimodalModel.from_checkpoint(checkpoint_path)
    model  = model.to(device)
    model.eval()

    dataset = ParkinsonDataset(test_data)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    all_updrs_pred, all_updrs_true = [], []
    all_cls_pred,   all_cls_true   = [], []
    all_cls_probs                  = []

    import torch.nn.functional as F

    with torch.no_grad():
        for batch in loader:
            batch   = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in batch.items()}
            outputs = model(
                voice_feats=batch.get("voice"),
                eeg_feats=batch.get("eeg"),
                gait_feats=batch.get("gait"),
                modality_mask=batch.get("modality_mask"),
            )
            all_updrs_pred.extend(outputs["updrs_pred"].cpu().numpy())
            all_updrs_true.extend(batch["updrs"].cpu().numpy())
            probs = F.softmax(outputs["severity_logits"], dim=-1).cpu().numpy()
            all_cls_probs.extend(probs)
            all_cls_pred.extend(np.argmax(probs, axis=-1))
            all_cls_true.extend(batch["severity"].cpu().numpy())

    metrics = compute_all_metrics(
        np.array(all_updrs_true), np.array(all_updrs_pred),
        np.array(all_cls_true),   np.array(all_cls_pred),
        np.array(all_cls_probs),
    )

    print_evaluation_report(
        np.array(all_updrs_true), np.array(all_updrs_pred),
        np.array(all_cls_true),   np.array(all_cls_pred),
        np.array(all_cls_probs),
    )

    return metrics

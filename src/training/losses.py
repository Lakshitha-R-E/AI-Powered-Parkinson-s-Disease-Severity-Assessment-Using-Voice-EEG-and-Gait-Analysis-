"""
============================================================
Training Losses — Dual Task
AI-Powered Parkinson's Disease Severity Assessment
============================================================
Task 1: UPDRS Regression  → MSE Loss
Task 2: Severity Classification → Weighted Cross-Entropy
Combined: 0.6 × Regression Loss + 0.4 × Classification Loss
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class UPDRSRegressionLoss(nn.Module):
    """
    UPDRS Regression Loss.

    Combines:
    - MSE (primary)
    - MAE (for robustness to outliers)
    - Optionally Huber loss (smooth L1)

    Args:
        alpha:  Weight of MSE vs MAE (0=pure MAE, 1=pure MSE)
        use_huber: Use Huber loss instead of MSE+MAE blend
    """

    def __init__(self, alpha: float = 0.7, use_huber: bool = False, delta: float = 1.0):
        super().__init__()
        self.alpha     = alpha
        self.use_huber = use_huber
        self.delta     = delta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred:   (B,) predicted UPDRS scores
            target: (B,) ground truth UPDRS scores
            weight: Optional (B,) sample weights

        Returns:
            Scalar loss value
        """
        pred   = pred.squeeze()
        target = target.squeeze()

        if self.use_huber:
            loss = F.huber_loss(pred, target, delta=self.delta, reduction="none")
        else:
            mse  = F.mse_loss(pred, target, reduction="none")
            mae  = F.l1_loss(pred,  target, reduction="none")
            loss = self.alpha * mse + (1 - self.alpha) * mae

        if weight is not None:
            loss = loss * weight

        return loss.mean()


class SeverityClassificationLoss(nn.Module):
    """
    Multi-class severity classification loss with:
    - Class-weighted cross-entropy (handles class imbalance)
    - Label smoothing for regularization

    Classes: 0=Mild, 1=Moderate, 2=Severe

    Args:
        class_weights: Optional (3,) tensor of class weights
        label_smoothing: Label smoothing factor (0–0.2)
    """

    def __init__(
        self,
        class_weights:    Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.class_weights  = class_weights
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, 3) raw class logits
            labels: (B,) integer class labels {0, 1, 2}

        Returns:
            Scalar loss value
        """
        return F.cross_entropy(
            logits,
            labels.long(),
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )


class DualTaskLoss(nn.Module):
    """
    Combined dual-task loss for simultaneous regression + classification.

    Loss = λ_reg × Regression Loss + λ_cls × Classification Loss

    Default weights:
        λ_reg = 0.6  (UPDRS regression dominant)
        λ_cls = 0.4  (severity classification)

    Supports learned uncertainty weighting (Kendall et al., 2018).

    Args:
        lambda_reg:     Weight for regression loss (default 0.6)
        lambda_cls:     Weight for classification loss (default 0.4)
        class_weights:  Optional class imbalance weights
        label_smoothing: Label smoothing (0–0.2)
        use_uncertainty: Learn loss weights via uncertainty (log σ)
    """

    def __init__(
        self,
        lambda_reg:     float = 0.6,
        lambda_cls:     float = 0.4,
        class_weights:  Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
        use_uncertainty: bool = False,
    ):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_cls = lambda_cls

        self.reg_loss = UPDRSRegressionLoss(use_huber=True)
        self.cls_loss = SeverityClassificationLoss(class_weights, label_smoothing)

        # Learned uncertainty weights (log σ² for each task)
        self.use_uncertainty = use_uncertainty
        if use_uncertainty:
            self.log_sigma_reg = nn.Parameter(torch.zeros(1))
            self.log_sigma_cls = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        pred_updrs:  torch.Tensor,
        true_updrs:  torch.Tensor,
        pred_logits: torch.Tensor,
        true_class:  torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            pred_updrs:    (B,) predicted UPDRS
            true_updrs:    (B,) ground truth UPDRS
            pred_logits:   (B, 3) class logits
            true_class:    (B,) ground truth severity class
            sample_weight: Optional (B,) sample weights

        Returns:
            Dict with keys: total, regression, classification
        """
        L_reg = self.reg_loss(pred_updrs, true_updrs, sample_weight)
        L_cls = self.cls_loss(pred_logits, true_class)

        if self.use_uncertainty:
            # Uncertainty weighting: L = 1/(2σ²) * L + log σ
            sigma_reg = torch.exp(self.log_sigma_reg)
            sigma_cls = torch.exp(self.log_sigma_cls)
            L_total   = (L_reg / (2 * sigma_reg**2) + self.log_sigma_reg +
                         L_cls / (2 * sigma_cls**2) + self.log_sigma_cls)
        else:
            L_total = self.lambda_reg * L_reg + self.lambda_cls * L_cls

        return {
            "total":          L_total,
            "regression":     L_reg,
            "classification": L_cls,
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling extreme class imbalance.
    Downweights easy examples, focuses on hard ones.

    Args:
        gamma:  Focusing parameter (2.0 recommended)
        alpha:  Optional class weight tensor
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """logits: (B, C), labels: (B,)"""
        ce_loss = F.cross_entropy(logits, labels.long(), weight=self.alpha, reduction="none")
        pt      = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

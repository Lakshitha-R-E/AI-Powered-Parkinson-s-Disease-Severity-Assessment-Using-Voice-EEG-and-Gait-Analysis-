"""
============================================================
Grad-CAM Implementation
AI-Powered Parkinson's Disease Severity Assessment
============================================================
Applies Gradient-weighted Class Activation Mapping to the
CNN layers in the Voice and EEG encoders.

Produces:
  - Heatmaps over input features
  - Overlay visualizations
  - Per-class activation maps
============================================================
"""

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for 1D CNNs.

    Usage:
        gradcam = GradCAM(model.voice_encoder, target_layer_name="conv_blocks.1")
        cam     = gradcam.generate(input_tensor, class_idx=2)  # class 2 = Severe
    """

    def __init__(self, model: nn.Module, target_layer_name: str):
        """
        Args:
            model:             The encoder module (VoiceEncoder or EEGEncoder)
            target_layer_name: Dot-separated path to the target conv layer
                               e.g. 'conv_blocks.1' or 'cnn.0'
        """
        self.model       = model
        self.target_name = target_layer_name
        self.gradients   = None
        self.activations = None
        self._hooks      = []

        self._register_hooks()

    def _get_target_layer(self) -> nn.Module:
        """Navigate model attribute path to find target layer."""
        parts = self.target_name.split(".")
        module = self.model
        for part in parts:
            if part.isdigit():
                module = list(module.children())[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _register_hooks(self):
        target = self._get_target_layer()

        def save_activation(module, input, output):
            self.activations = output.detach()

        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(target.register_forward_hook(save_activation))
        self._hooks.append(target.register_backward_hook(save_gradient))

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int,
        use_relu: bool = True,
    ) -> np.ndarray:
        """
        Generate Grad-CAM activation map.

        Args:
            input_tensor: (1, input_dim) model input
            class_idx:    Target class index (0=Mild, 1=Moderate, 2=Severe)
            use_relu:     Apply ReLU to final CAM (keep only positive contributions)

        Returns:
            cam: 1-D normalized activation map of shape (input_dim,)
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)                  # (1, embed_dim)

        # Backward for target class
        self.model.zero_grad()
        # For regression use output[0] directly, for class use a proxy
        score = output[0, class_idx % output.shape[-1]] if output.dim() == 2 else output[0]
        score.backward(retain_graph=True)

        # Compute CAM
        if self.gradients is None or self.activations is None:
            return np.zeros(input_tensor.shape[-1])

        # Global average pool gradients → weights: (C,)
        weights = self.gradients.mean(dim=-1, keepdim=True)  # (1, C, 1) or (C, 1)

        # Weighted sum of activations: (C, L) → (L,)
        if self.activations.dim() == 3:
            acts = self.activations[0]                     # (C, L)
            cam  = (weights[0] * acts).sum(dim=0)          # (L,)
        else:
            acts = self.activations[0]                     # (C,)
            cam  = (weights[0] * acts).sum(dim=0, keepdim=True)

        # Apply ReLU
        if use_relu:
            cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        # Resize to input dimension
        input_dim = input_tensor.shape[-1]
        cam_resized = np.interp(
            np.linspace(0, len(cam) - 1, input_dim),
            np.arange(len(cam)),
            cam
        )
        return cam_resized

    def generate_all_classes(
        self,
        input_tensor: torch.Tensor,
        n_classes: int = 3,
    ) -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM for all severity classes.

        Returns:
            Dict mapping class_name → cam array
        """
        class_names = ["Mild", "Moderate", "Severe"]
        cams        = {}
        for i in range(n_classes):
            try:
                cam = self.generate(input_tensor.clone(), class_idx=i)
                cams[class_names[i]] = cam
            except Exception as e:
                print(f"[WARN] Grad-CAM failed for class {i}: {e}")
                cams[class_names[i]] = np.zeros(input_tensor.shape[-1])
        return cams

    def plot_cam(
        self,
        cam:           np.ndarray,
        input_data:    Optional[np.ndarray] = None,
        feature_names: Optional[List[str]]  = None,
        title:         str = "Grad-CAM Activation",
        save_path:     Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot Grad-CAM activation with optional overlay of input signal.

        Args:
            cam:           1-D CAM array
            input_data:    Optional input signal for overlay
            feature_names: Optional feature names for x-axis
            title:         Plot title
            save_path:     Optional save path
        """
        fig, ax = plt.subplots(figsize=(12, 4))

        x = np.arange(len(cam))

        # Fill under CAM curve
        ax.fill_between(x, cam, alpha=0.4, color="#e74c3c", label="Grad-CAM")
        ax.plot(x, cam, color="#c0392b", linewidth=2)

        # Overlay input signal (normalized)
        if input_data is not None:
            sig = (input_data - input_data.min()) / (input_data.max() - input_data.min() + 1e-8)
            sig = np.interp(np.linspace(0, len(sig)-1, len(cam)),
                            np.arange(len(sig)), sig)
            ax.plot(x, sig, color="#3498db", linewidth=1.5, alpha=0.7, label="Input signal")

        if feature_names and len(feature_names) == len(cam):
            step = max(1, len(feature_names) // 20)
            ax.set_xticks(x[::step])
            ax.set_xticklabels(feature_names[::step], rotation=45, ha="right", fontsize=8)
        else:
            ax.set_xlabel("Feature Index", fontsize=11)

        ax.set_ylabel("Activation", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_all_classes(
        self,
        cams: Dict[str, np.ndarray],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot Grad-CAM for all 3 severity classes in one figure."""
        class_colors = {
            "Mild":     "#27ae60",
            "Moderate": "#f39c12",
            "Severe":   "#e74c3c",
        }

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        for ax, (cls_name, cam) in zip(axes, cams.items()):
            color = class_colors.get(cls_name, "#3498db")
            ax.fill_between(np.arange(len(cam)), cam, alpha=0.4, color=color)
            ax.plot(cam, color=color, linewidth=2)
            ax.set_ylabel(cls_name, fontsize=11, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.spines[["top", "right"]].set_visible(False)

        axes[-1].set_xlabel("Feature Index", fontsize=11)
        fig.suptitle("Grad-CAM Activations by Severity Class", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

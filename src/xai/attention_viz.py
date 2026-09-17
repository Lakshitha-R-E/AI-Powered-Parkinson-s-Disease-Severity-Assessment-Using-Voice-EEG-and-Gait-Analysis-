"""
============================================================
Attention Visualization
AI-Powered Parkinson's Disease Severity Assessment
============================================================
Visualizes:
  - EEG Encoder self-attention weights (CLS attention)
  - Fusion Transformer cross-modal attention
  - Gait BiLSTM attention weights over time
  - Attention heatmaps (channel × channel, time × channel)
============================================================
"""

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn


class AttentionVisualizer:
    """
    Visualizes attention weights from the multimodal model.
    """

    MODALITY_LABELS = ["Voice", "EEG", "Gait"]
    SEVERITY_LABELS = ["Mild", "Moderate", "Severe"]

    def plot_eeg_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        layer_idx:         int = -1,
        save_path:         Optional[str] = None,
        title:             str = "EEG Self-Attention Heatmap",
    ) -> plt.Figure:
        """
        Plot attention heatmap for EEG encoder.

        Args:
            attention_weights: Tensor of shape (B, n_heads, seq_len, seq_len)
                               or (n_heads, seq_len, seq_len)
            layer_idx:         Which attention layer to visualize (-1 = last)
        """
        if isinstance(attention_weights, torch.Tensor):
            attn = attention_weights.detach().cpu().numpy()
        else:
            attn = np.array(attention_weights)

        # Use first sample if batched
        if attn.ndim == 4:
            attn = attn[0]                                 # (n_heads, seq, seq)

        # Average over heads
        attn_avg = attn.mean(axis=0)                       # (seq_len, seq_len)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Average attention
        im = axes[0].imshow(attn_avg, cmap="viridis", vmin=0, vmax=attn_avg.max())
        axes[0].set_title(f"{title}\n(Mean over heads)", fontweight="bold")
        axes[0].set_xlabel("Key position")
        axes[0].set_ylabel("Query position")
        plt.colorbar(im, ax=axes[0], fraction=0.046)

        # Panel 2: Per-head heatmap (first 4 heads as grid)
        n_heads = min(4, attn.shape[0])
        attn_heads = attn[:n_heads]
        grid = np.concatenate(
            [np.concatenate(attn_heads[i * 2:(i + 1) * 2], axis=1)
             for i in range(n_heads // 2 + 1) if i * 2 < n_heads],
            axis=0,
        )
        im2 = axes[1].imshow(grid, cmap="magma", aspect="auto")
        axes[1].set_title("Per-head Attention (first 4)", fontweight="bold")
        plt.colorbar(im2, ax=axes[1], fraction=0.046)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_fusion_attention(
        self,
        fusion_attention: torch.Tensor,
        save_path:        Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot cross-modal fusion attention as 3×3 matrix.
        Rows = query modality, Columns = key modality.

        Args:
            fusion_attention: (B, n_heads, 3, 3) or (n_heads, 3, 3)
        """
        if isinstance(fusion_attention, torch.Tensor):
            attn = fusion_attention.detach().cpu().numpy()
        else:
            attn = np.array(fusion_attention)

        if attn.ndim == 4:
            attn = attn[0]                                  # (n_heads, 3, 3)

        attn_avg = attn.mean(axis=0)                        # (3, 3)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(attn_avg, cmap="YlOrRd", vmin=0, vmax=1)

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(self.MODALITY_LABELS, fontsize=11)
        ax.set_yticklabels(self.MODALITY_LABELS, fontsize=11)
        ax.set_xlabel("Key Modality", fontsize=11)
        ax.set_ylabel("Query Modality", fontsize=11)
        ax.set_title("Cross-Modal Fusion Attention", fontsize=13, fontweight="bold")

        # Annotate cells
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{attn_avg[i, j]:.2f}",
                        ha="center", va="center", fontsize=12,
                        color="white" if attn_avg[i, j] > 0.5 else "black",
                        fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_gait_attention(
        self,
        attention_weights: torch.Tensor,
        time_labels:       Optional[List[str]] = None,
        save_path:         Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot BiLSTM attention weights over gait sequence tokens.

        Args:
            attention_weights: (B, seq_len) or (seq_len,)
        """
        if isinstance(attention_weights, torch.Tensor):
            attn = attention_weights.detach().cpu().numpy()
        else:
            attn = np.array(attention_weights)

        if attn.ndim == 2:
            attn = attn[0]                                  # (seq_len,)

        seq_len = len(attn)
        labels  = time_labels or [f"Token {i+1}" for i in range(seq_len)]

        fig, ax = plt.subplots(figsize=(max(8, seq_len * 0.8), 3))

        colors  = plt.cm.RdYlGn(attn)
        bars    = ax.bar(range(seq_len), attn, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Attention Weight", fontsize=11)
        ax.set_title("Gait BiLSTM Attention over Sequence Tokens",
                     fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(attn) * 1.2)
        ax.spines[["top", "right"]].set_visible(False)

        # Add value labels
        for bar, val in zip(bars, attn):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_modality_contribution(
        self,
        modality_scores: Dict[str, float],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Radar chart showing relative contribution of each modality.

        Args:
            modality_scores: Dict mapping modality → importance score
        """
        labels = list(modality_scores.keys())
        values = list(modality_scores.values())
        n      = len(labels)

        # Normalize
        total  = sum(values) + 1e-8
        values_norm = [v / total for v in values]
        values_norm += values_norm[:1]     # Close the polygon

        angles = [i * 2 * np.pi / n for i in range(n)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values_norm, "o-", linewidth=2, color="#3498db")
        ax.fill(angles, values_norm, alpha=0.25, color="#3498db")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_title("Modality Contribution", fontsize=13, fontweight="bold", pad=20)
        ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def generate_full_report(
        self,
        model_outputs: Dict,
        save_dir: str,
    ) -> Dict[str, plt.Figure]:
        """
        Generate all attention visualizations from a model output dict.
        """
        from pathlib import Path
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        figures = {}

        # Fusion attention
        if "fusion_attention" in model_outputs:
            fig = self.plot_fusion_attention(
                model_outputs["fusion_attention"],
                save_path=str(save_dir / "fusion_attention.png"),
            )
            figures["fusion_attention"] = fig

        # Gait attention
        if "gait_attention" in model_outputs:
            fig = self.plot_gait_attention(
                model_outputs["gait_attention"],
                save_path=str(save_dir / "gait_attention.png"),
            )
            figures["gait_attention"] = fig

        # Modality contribution
        if "modality_mask" in model_outputs:
            mask = model_outputs["modality_mask"]
            if isinstance(mask, torch.Tensor):
                mask = mask[0].cpu().numpy()
            scores = {
                "Voice": float(mask[0]),
                "EEG":   float(mask[1]),
                "Gait":  float(mask[2]),
            }
            fig = self.plot_modality_contribution(
                scores, save_path=str(save_dir / "modality_contribution.png")
            )
            figures["modality_contribution"] = fig

        return figures

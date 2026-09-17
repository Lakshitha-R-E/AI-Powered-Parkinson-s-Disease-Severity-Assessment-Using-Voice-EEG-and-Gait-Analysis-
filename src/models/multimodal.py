"""
============================================================
Full Multimodal Parkinson's Severity Assessment Model
AI-Powered Parkinson's Disease Severity Assessment
============================================================
Combines:
  - VoiceEncoder (CNN)
  - EEGEncoder (CNN + Attention)
  - GaitEncoder (BiLSTM)
  - MultimodalFusion (Transformer cross-attention)
  - Dual prediction heads:
      • UPDRS Regression head
      • Severity Classification head (Mild/Moderate/Severe)

Safe Fallback:
  - Supports missing modalities via zero-filling + mask
  - Dynamic fusion weight adjustment
============================================================
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .voice_encoder import VoiceEncoder
from .eeg_encoder   import EEGEncoder
from .gait_encoder  import GaitEncoder
from .fusion        import MultimodalFusion


# ─────────────────────────────────────────────
# Severity Classes
# ─────────────────────────────────────────────
SEVERITY_CLASSES = ["Mild", "Moderate", "Severe"]
N_SEVERITY_CLASSES = 3


# ─────────────────────────────────────────────
# Prediction Heads
# ─────────────────────────────────────────────

class UPDRSRegressionHead(nn.Module):
    """
    Regression head for UPDRS score prediction.
    Output range: [0, 108] (total UPDRS scale)
    """

    def __init__(self, d_model: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 4, 1),
        )
        # UPDRS score bounded: [0, 108]
        self.scale = nn.Parameter(torch.tensor(108.0), requires_grad=False)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused: (B, d_model)
        Returns:
            updrs: (B,) in range [0, 108]
        """
        raw = self.head(fused).squeeze(-1)               # (B,)
        return torch.sigmoid(raw) * self.scale           # [0, 108]


class SeverityClassificationHead(nn.Module):
    """
    Classification head for severity class:
    0=Mild, 1=Moderate, 2=Severe
    """

    def __init__(self, d_model: int, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 4, n_classes),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused: (B, d_model)
        Returns:
            logits: (B, n_classes)
        """
        return self.head(fused)


# ─────────────────────────────────────────────
# Complete Multimodal Model
# ─────────────────────────────────────────────

class ParkinsonMultimodalModel(nn.Module):
    """
    Complete Parkinson's Disease Severity Assessment Model.

    Supports:
    - All 3 modalities (voice + EEG + gait)
    - Any subset (1 or 2 modalities — graceful degradation)
    - Modality dropout during training for robustness

    Args:
        voice_input_dim:      Dimensionality of voice features
        eeg_input_dim:        Dimensionality of EEG features
        gait_input_dim:       Dimensionality of gait features
        embed_dim:            Per-encoder embedding dimension
        fusion_d_model:       Fusion transformer dimension
        fusion_n_heads:       Fusion attention heads
        fusion_n_layers:      Fusion transformer depth
        n_severity_classes:   Number of severity classes (3)
        dropout:              Dropout rate
        modality_dropout_p:   Probability of randomly dropping a modality during training

    Forward:
        voice_feats:    (B, voice_input_dim) or None
        eeg_feats:      (B, eeg_input_dim) or None
        gait_feats:     (B, gait_input_dim) or None
        modality_mask:  (B, 3) binary mask [voice, eeg, gait] or None

    Returns:
        Dict with:
            updrs_pred:     (B,) UPDRS regression predictions
            severity_logits:(B, 3) severity class logits
            voice_emb:      (B, embed_dim)
            eeg_emb:        (B, embed_dim)
            gait_emb:       (B, embed_dim)
            fused_emb:      (B, fusion_d_model)
    """

    def __init__(
        self,
        voice_input_dim:    int   = 200,
        eeg_input_dim:      int   = 150,
        gait_input_dim:     int   = 40,
        embed_dim:          int   = 256,
        fusion_d_model:     int   = 512,
        fusion_n_heads:     int   = 8,
        fusion_n_layers:    int   = 4,
        n_severity_classes: int   = 3,
        dropout:            float = 0.2,
        modality_dropout_p: float = 0.15,
    ):
        super().__init__()
        self.modality_dropout_p = modality_dropout_p
        self.embed_dim          = embed_dim
        self.fusion_d_model     = fusion_d_model
        self.voice_input_dim    = voice_input_dim
        self.eeg_input_dim      = eeg_input_dim
        self.gait_input_dim     = gait_input_dim

        # ── Modality Encoders ─────────────────────────────
        self.voice_encoder = VoiceEncoder(
            input_dim=voice_input_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.eeg_encoder = EEGEncoder(
            input_dim=eeg_input_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.gait_encoder = GaitEncoder(
            input_dim=gait_input_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        # ── Multimodal Fusion ─────────────────────────────
        self.fusion = MultimodalFusion(
            voice_dim=embed_dim,
            eeg_dim=embed_dim,
            gait_dim=embed_dim,
            d_model=fusion_d_model,
            n_heads=fusion_n_heads,
            n_layers=fusion_n_layers,
            dropout=dropout,
        )

        # ── Prediction Heads ──────────────────────────────
        self.updrs_head    = UPDRSRegressionHead(fusion_d_model, dropout)
        self.severity_head = SeverityClassificationHead(
            fusion_d_model, n_severity_classes, dropout
        )

        # Parameter count summary
        total = sum(p.numel() for p in self.parameters())
        print(f"[MODEL] ParkinsonMultimodalModel — {total/1e6:.2f}M parameters")

    def _apply_modality_dropout(
        self,
        voice_feats: Optional[torch.Tensor],
        eeg_feats:   Optional[torch.Tensor],
        gait_feats:  Optional[torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        During training, randomly drop modalities to improve robustness.
        At least one modality is always preserved.
        """
        if not self.training or self.modality_dropout_p == 0.0:
            return voice_feats, eeg_feats, gait_feats, modality_mask

        B = modality_mask.shape[0]
        for i in range(B):
            # Count available modalities
            available = modality_mask[i].nonzero().squeeze(-1)
            if len(available) <= 1:
                continue  # Don't drop if only one left

            for mod_idx in range(3):
                if (modality_mask[i, mod_idx] == 1 and
                        torch.rand(1).item() < self.modality_dropout_p):
                    modality_mask[i, mod_idx] = 0.0

            # Ensure at least one modality
            if modality_mask[i].sum() == 0:
                modality_mask[i, torch.randint(3, (1,)).item()] = 1.0

        return voice_feats, eeg_feats, gait_feats, modality_mask

    def encode_modalities(
        self,
        voice_feats: Optional[torch.Tensor],
        eeg_feats:   Optional[torch.Tensor],
        gait_feats:  Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode each modality independently.
        Missing modalities are encoded as zero vectors.
        """
        device = next(self.parameters()).device
        B = next(
            (x.shape[0] for x in [voice_feats, eeg_feats, gait_feats] if x is not None),
            1
        )

        if voice_feats is not None:
            voice_emb = self.voice_encoder(voice_feats)
        else:
            voice_emb = torch.zeros(B, self.embed_dim, device=device)

        if eeg_feats is not None:
            eeg_emb = self.eeg_encoder(eeg_feats)
        else:
            eeg_emb = torch.zeros(B, self.embed_dim, device=device)

        if gait_feats is not None:
            gait_emb = self.gait_encoder(gait_feats)
        else:
            gait_emb = torch.zeros(B, self.embed_dim, device=device)

        return voice_emb, eeg_emb, gait_emb

    def forward(
        self,
        voice_feats:    Optional[torch.Tensor] = None,
        eeg_feats:      Optional[torch.Tensor] = None,
        gait_feats:     Optional[torch.Tensor] = None,
        modality_mask:  Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        """
        device = next(self.parameters()).device
        B = next(
            (x.shape[0] for x in [voice_feats, eeg_feats, gait_feats] if x is not None),
            1
        )

        # Build modality mask from available inputs
        if modality_mask is None:
            modality_mask = torch.tensor(
                [
                    [float(voice_feats is not None),
                     float(eeg_feats is not None),
                     float(gait_feats is not None)]
                ],
                device=device,
            ).expand(B, -1).clone()

        # Apply modality dropout during training
        voice_feats, eeg_feats, gait_feats, modality_mask = self._apply_modality_dropout(
            voice_feats, eeg_feats, gait_feats, modality_mask
        )

        # Encode modalities
        voice_emb, eeg_emb, gait_emb = self.encode_modalities(
            voice_feats, eeg_feats, gait_feats
        )

        # Zero out masked embeddings
        mask = modality_mask.unsqueeze(-1)               # (B, 3, 1)
        voice_emb = voice_emb * mask[:, 0, :]
        eeg_emb   = eeg_emb   * mask[:, 1, :]
        gait_emb  = gait_emb  * mask[:, 2, :]

        # Multimodal fusion
        fused_emb = self.fusion(
            voice_emb=voice_emb,
            eeg_emb=eeg_emb,
            gait_emb=gait_emb,
            modality_mask=modality_mask,
        )                                                 # (B, fusion_d_model)

        # Predictions
        updrs_pred      = self.updrs_head(fused_emb)     # (B,)
        severity_logits = self.severity_head(fused_emb)  # (B, 3)

        return {
            "updrs_pred":      updrs_pred,
            "severity_logits": severity_logits,
            "voice_emb":       voice_emb,
            "eeg_emb":         eeg_emb,
            "gait_emb":        gait_emb,
            "fused_emb":       fused_emb,
        }

    def predict(
        self,
        voice_feats:   Optional[torch.Tensor] = None,
        eeg_feats:     Optional[torch.Tensor] = None,
        gait_feats:    Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """
        Inference-mode prediction with human-readable outputs.

        Returns:
            Dict with updrs_score, severity_class, severity_label,
            class_probabilities, confidence
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(voice_feats, eeg_feats, gait_feats, modality_mask)

            updrs          = outputs["updrs_pred"].cpu().numpy()
            severity_probs = F.softmax(outputs["severity_logits"], dim=-1).cpu().numpy()
            severity_class = severity_probs.argmax(axis=-1)
            confidence     = severity_probs.max(axis=-1)

            return {
                "updrs_score":         updrs,
                "severity_class":      severity_class,
                "severity_label":      [SEVERITY_CLASSES[c] for c in severity_class],
                "class_probabilities": severity_probs,
                "confidence":          confidence,
                "fused_embedding":     outputs["fused_emb"].cpu().numpy(),
            }

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, **kwargs) -> "ParkinsonMultimodalModel":
        """Load model from checkpoint file."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_args = checkpoint.get("model_args", {})
        model_args.update(kwargs)
        model = cls(**model_args)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[MODEL] Loaded checkpoint from {checkpoint_path}")
        return model

"""
============================================================
Transformer-Based Multimodal Fusion Network
AI-Powered Parkinson's Disease Severity Assessment
============================================================
Architecture:
  Input: Voice Embedding + EEG Embedding + Gait Embedding
  → Modality-specific linear projections (→ common d_model)
  → Learnable modality tokens + positional encoding
  → Multi-head Self-Attention (intra-modal)
  → Cross-modal Attention (inter-modal)
  → Residual connections + Layer Normalization
  → Output: Unified multimodal representation (d_model)
============================================================
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Modality Projection
# ─────────────────────────────────────────────

class ModalityProjection(nn.Module):
    """
    Projects each modality embedding to a common d_model.
    Also applies learnable modality-type embedding.
    """

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_dim) → (B, d_model)"""
        return self.proj(x)


# ─────────────────────────────────────────────
# Cross-Modal Attention
# ─────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """
    Cross-attention between a query modality and key/value modalities.
    Allows modalities to attend to each other's representations.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query:     (B, 1, d_model) — query modality token
            key_value: (B, N, d_model) — context modalities

        Returns:
            attended: (B, 1, d_model)
            weights:  (B, 1, N)
        """
        attn_out, weights = self.attn(query, key_value, key_value)
        out = self.norm(query + self.dropout(attn_out))
        return out, weights


# ─────────────────────────────────────────────
# Fusion Transformer Block
# ─────────────────────────────────────────────

class FusionTransformerBlock(nn.Module):
    """
    Single fusion transformer block combining:
    1. Self-attention across all modality tokens
    2. Cross-attention: each modality attends to others
    3. Feed-forward network
    All with residual connections + layer norm.
    """

    def __init__(self, d_model: int, n_heads: int = 8, ff_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        ff_dim = d_model * ff_ratio

        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1     = nn.LayerNorm(d_model)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

        # Cross-modal attention (one per modality pair direction)
        self.cross_attn_v2e  = CrossModalAttention(d_model, n_heads, dropout)  # voice ← EEG+Gait
        self.cross_attn_e2v  = CrossModalAttention(d_model, n_heads, dropout)  # EEG ← Voice+Gait
        self.cross_attn_g2ve = CrossModalAttention(d_model, n_heads, dropout)  # Gait ← Voice+EEG

        self.norm_cross = nn.LayerNorm(d_model)

        # Store attention weights for visualization
        self.self_attn_weights  = None
        self.cross_attn_weights = []

    def forward(
        self,
        tokens: torch.Tensor,
        mask:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (B, 3, d_model) — [voice, eeg, gait] tokens
            mask:   Optional modality mask (B, 3) — 0 = missing

        Returns:
            tokens: (B, 3, d_model) — updated tokens
        """
        B = tokens.shape[0]

        # ── Self-Attention across modalities ──────────────
        attn_out, self.self_attn_weights = self.self_attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + self.drop(attn_out))

        # ── Cross-Modal Attention ─────────────────────────
        v_tok = tokens[:, 0:1, :]   # (B, 1, d_model)
        e_tok = tokens[:, 1:2, :]
        g_tok = tokens[:, 2:3, :]

        # Voice attends to EEG + Gait
        eg_ctx = torch.cat([e_tok, g_tok], dim=1)       # (B, 2, d_model)
        vg_ctx = torch.cat([v_tok, g_tok], dim=1)
        ve_ctx = torch.cat([v_tok, e_tok], dim=1)

        v_new, w_v = self.cross_attn_v2e(v_tok, eg_ctx)
        e_new, w_e = self.cross_attn_e2v(e_tok, vg_ctx)
        g_new, w_g = self.cross_attn_g2ve(g_tok, ve_ctx)

        self.cross_attn_weights = [w_v, w_e, w_g]
        tokens = self.norm_cross(
            torch.cat([v_new, e_new, g_new], dim=1)
        )                                                # (B, 3, d_model)

        # ── Feed-Forward ──────────────────────────────────
        ff_out = self.ff(tokens)
        tokens = self.norm2(tokens + ff_out)

        return tokens


# ─────────────────────────────────────────────
# Main Fusion Network
# ─────────────────────────────────────────────

class MultimodalFusion(nn.Module):
    """
    Transformer-Based Multimodal Fusion Network.

    Fuses voice, EEG, and gait embeddings using:
    - Per-modality projections to common d_model
    - Learnable modality type embeddings
    - Stacked fusion transformer blocks (self + cross attention)
    - Modality dropout for robustness

    Args:
        voice_dim:  Voice embedding dimension
        eeg_dim:    EEG embedding dimension
        gait_dim:   Gait embedding dimension
        d_model:    Common model dimension
        n_heads:    Attention heads
        n_layers:   Fusion transformer layers
        dropout:    Dropout rate

    Input:
        voice_emb: (B, voice_dim) or None
        eeg_emb:   (B, eeg_dim) or None
        gait_emb:  (B, gait_dim) or None
        modality_mask: (B, 3) binary mask [voice, eeg, gait]

    Output:
        fused: (B, d_model)
    """

    def __init__(
        self,
        voice_dim: int  = 256,
        eeg_dim:   int  = 256,
        gait_dim:  int  = 256,
        d_model:   int  = 512,
        n_heads:   int  = 8,
        n_layers:  int  = 4,
        dropout:   float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Modality projections
        self.voice_proj = ModalityProjection(voice_dim, d_model, dropout)
        self.eeg_proj   = ModalityProjection(eeg_dim,   d_model, dropout)
        self.gait_proj  = ModalityProjection(gait_dim,  d_model, dropout)

        # Learnable modality type embeddings [voice, eeg, gait]
        self.modality_embeddings = nn.Embedding(3, d_model)

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Fusion transformer stack
        self.fusion_blocks = nn.ModuleList([
            FusionTransformerBlock(d_model, n_heads, ff_ratio=4, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.output_norm = nn.LayerNorm(d_model)

        # Modality gate (soft weighting based on available modalities)
        self.modality_gate = nn.Sequential(
            nn.Linear(3, d_model),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_fusion_attention(self) -> List[torch.Tensor]:
        """Return self-attention weights from all fusion blocks."""
        return [blk.self_attn_weights for blk in self.fusion_blocks
                if blk.self_attn_weights is not None]

    def forward(
        self,
        voice_emb:     Optional[torch.Tensor] = None,
        eeg_emb:       Optional[torch.Tensor] = None,
        gait_emb:      Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            voice_emb:     (B, voice_dim) or None
            eeg_emb:       (B, eeg_dim) or None
            gait_emb:      (B, gait_dim) or None
            modality_mask: (B, 3) — 1=available, 0=missing

        Returns:
            fused: (B, d_model)
        """
        B = next(
            (x.shape[0] for x in [voice_emb, eeg_emb, gait_emb] if x is not None),
            1
        )
        device = next(self.parameters()).device

        if modality_mask is None:
            modality_mask = torch.ones(B, 3, device=device)

        # ── Project each modality ─────────────────────────
        # Replace missing modalities with zeros
        if voice_emb is None:
            voice_emb = torch.zeros(B, self.voice_proj.proj[0].in_features, device=device)
        if eeg_emb is None:
            eeg_emb   = torch.zeros(B, self.eeg_proj.proj[0].in_features,   device=device)
        if gait_emb is None:
            gait_emb  = torch.zeros(B, self.gait_proj.proj[0].in_features,  device=device)

        v_tok = self.voice_proj(voice_emb)               # (B, d_model)
        e_tok = self.eeg_proj(eeg_emb)
        g_tok = self.gait_proj(gait_emb)

        # ── Add modality type embeddings ──────────────────
        mod_ids = torch.arange(3, device=device)         # [0, 1, 2]
        mod_emb = self.modality_embeddings(mod_ids)      # (3, d_model)

        v_tok = v_tok + mod_emb[0]
        e_tok = e_tok + mod_emb[1]
        g_tok = g_tok + mod_emb[2]

        # ── Apply modality mask (zero out missing) ────────
        mask = modality_mask.unsqueeze(-1)               # (B, 3, 1)
        tokens = torch.stack([v_tok, e_tok, g_tok], dim=1) * mask  # (B, 3, d_model)

        # ── Soft modality gating ──────────────────────────
        gate = self.modality_gate(modality_mask)         # (B, d_model)

        # ── Fusion transformer blocks ─────────────────────
        for block in self.fusion_blocks:
            tokens = block(tokens)                       # (B, 3, d_model)

        # ── Aggregate: weighted mean over modality tokens ─
        weights = modality_mask / (modality_mask.sum(dim=-1, keepdim=True) + 1e-8)
        fused   = (tokens * weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)

        # Apply gating
        fused   = fused * gate                           # (B, d_model)

        return self.output_norm(fused)                   # (B, d_model)

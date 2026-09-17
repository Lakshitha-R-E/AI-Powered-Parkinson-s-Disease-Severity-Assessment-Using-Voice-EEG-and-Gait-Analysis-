"""
============================================================
EEG Encoder — CNN + Multi-head Attention
AI-Powered Parkinson's Disease Severity Assessment
============================================================
Architecture:
  Input: EEG feature vector (variable dim)
  → Linear projection → reshape to (B, C, L)
  → CNN feature extraction
  → Multi-head Self-Attention (temporal)
  → Global pooling
  → Output: EEG embedding (256-d)
============================================================
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    Efficient depthwise separable convolution for EEG processing.
    Depthwise + Pointwise convolution reduces parameter count.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dw = nn.Conv1d(
            in_ch, in_ch, kernel_size,
            padding=kernel_size // 2, groups=in_ch, bias=False
        )
        self.pw    = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn    = nn.BatchNorm1d(out_ch)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return self.drop(x)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for attention mechanism.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position  = torch.arange(max_len).unsqueeze(1)
        div_term  = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe        = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (seq_len, B, d_model)"""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class EEGAttentionBlock(nn.Module):
    """
    Transformer encoder block adapted for EEG feature sequences.
    """

    def __init__(self, d_model: int, n_heads: int = 8, ff_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)"""
        # Self-attention
        attn_out, self.attn_weights = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop(ff_out))
        return x


class EEGEncoder(nn.Module):
    """
    CNN + Multi-head Attention EEG Encoder.

    Architecture:
        1. Linear projection of input features → sequence of tokens
        2. Depthwise separable CNN blocks for local feature extraction
        3. Positional encoding
        4. Multi-head self-attention stack for global context
        5. CLS-token output → embedding

    Args:
        input_dim:  EEG feature dimension
        embed_dim:  Output embedding dimension
        n_heads:    Number of attention heads
        n_layers:   Number of attention layers
        dropout:    Dropout rate

    Input:  (B, input_dim)
    Output: (B, embed_dim)
    """

    def __init__(
        self,
        input_dim: int = 150,
        embed_dim: int = 256,
        n_heads:   int = 8,
        n_layers:  int = 3,
        dropout:   float = 0.2,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        seq_len = 16  # Number of tokens after reshape

        # Project input to (seq_len * d_model)
        d_model = embed_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, seq_len * d_model // 4),
            nn.GELU(),
            nn.Linear(seq_len * d_model // 4, seq_len * d_model),
        )
        self.seq_len    = seq_len
        self.d_model    = d_model

        # CNN blocks operating on the sequence (B, d_model, seq_len)
        self.cnn = nn.Sequential(
            DepthwiseSeparableConv(d_model, d_model, kernel_size=3, dropout=dropout),
            DepthwiseSeparableConv(d_model, d_model, kernel_size=3, dropout=dropout),
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Attention blocks
        self.attn_blocks = nn.ModuleList([
            EEGAttentionBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        self.embed_dim = embed_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_attention_weights(self) -> list:
        """Return attention weights from all layers (for visualization)."""
        return [blk.attn_weights for blk in self.attn_blocks if hasattr(blk, "attn_weights")]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)
        Returns:
            embedding: (B, embed_dim)
        """
        B = x.shape[0]

        # Project and reshape → (B, seq_len, d_model)
        h = self.input_proj(x)                         # (B, seq_len * d_model)
        h = h.view(B, self.seq_len, self.d_model)      # (B, seq_len, d_model)

        # CNN: transpose to (B, d_model, seq_len), apply, transpose back
        h = self.cnn(h.transpose(1, 2)).transpose(1, 2)  # (B, seq_len, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)        # (B, 1, d_model)
        h   = torch.cat([cls, h], dim=1)               # (B, 1+seq_len, d_model)

        # Self-attention layers
        for blk in self.attn_blocks:
            h = blk(h)                                 # (B, 1+seq_len, d_model)

        # Use CLS token as embedding
        cls_out   = h[:, 0, :]                         # (B, d_model)
        embedding = self.head(cls_out)                 # (B, embed_dim)
        return embedding

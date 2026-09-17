"""
============================================================
Voice Encoder — CNN-based
AI-Powered Parkinson's Disease Severity Assessment
============================================================
Architecture:
  Input: Voice feature vector (variable dim)
  → Linear projection → (B, C, L)
  → Conv1D blocks with residual connections
  → Global Average Pooling
  → Output: Voice embedding (256-d)
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    """
    1D Convolutional block with:
    - Conv1d
    - BatchNorm
    - GELU activation
    - Residual connection (with 1x1 projection if dims differ)
    - Dropout
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int = 3,
        stride:       int = 1,
        dropout:      float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        # Residual projection if channels or stride differ
        self.residual = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.residual(x))


class VoiceEncoder(nn.Module):
    """
    CNN-based Voice Encoder for Parkinson's voice feature embedding.

    Args:
        input_dim:     Dimensionality of input voice feature vector
        embed_dim:     Output embedding dimension (default 256)
        channels:      List of channel sizes for each conv stage
        dropout:       Dropout rate

    Input:
        x: (batch_size, input_dim)
    Output:
        embedding: (batch_size, embed_dim)
    """

    def __init__(
        self,
        input_dim: int = 200,
        embed_dim: int = 256,
        channels:  list = None,
        dropout:   float = 0.2,
    ):
        super().__init__()
        channels = channels or [64, 128, 256, 256]

        # Project input to first channel and expand as sequence
        self.input_proj = nn.Linear(input_dim, channels[0] * 8)

        # Build convolutional stages
        conv_layers = []
        for i in range(len(channels) - 1):
            conv_layers.append(
                ConvBlock1D(channels[i], channels[i + 1], kernel_size=3, dropout=dropout)
            )
        self.conv_blocks = nn.Sequential(*conv_layers)

        # Global average + max pooling, then project to embed_dim
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(channels[-1], embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        self.embed_dim = embed_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) — batch of voice feature vectors
        Returns:
            embedding: (B, embed_dim)
        """
        B = x.shape[0]

        # Project & reshape: (B, C*L) → (B, C, L)
        h = self.input_proj(x)                          # (B, channels[0]*8)
        h = h.view(B, -1, 8)                            # (B, channels[0], 8)

        # Convolutional feature extraction
        h = self.conv_blocks(h)                         # (B, channels[-1], L)

        # Global average pooling
        h = self.pool(h).squeeze(-1)                    # (B, channels[-1])

        # Project to embedding
        embedding = self.head(h)                        # (B, embed_dim)
        return embedding

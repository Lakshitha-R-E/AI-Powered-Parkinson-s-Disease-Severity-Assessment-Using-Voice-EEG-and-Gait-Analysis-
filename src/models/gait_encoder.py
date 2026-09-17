"""
============================================================
Gait Encoder — Bidirectional LSTM
AI-Powered Parkinson's Disease Severity Assessment
============================================================
Architecture:
  Input: Gait feature vector (variable dim)
  → Linear projection → reshape as sequence
  → Bidirectional LSTM stack
  → Attention pooling over time
  → Output: Gait embedding (256-d)
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAttentionPooling(nn.Module):
    """
    Attention-weighted pooling over LSTM output sequence.
    Learns which timesteps are most relevant for classification.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, lstm_out: torch.Tensor) -> tuple:
        """
        Args:
            lstm_out: (B, seq_len, hidden_dim)
        Returns:
            context:  (B, hidden_dim) weighted sum
            weights:  (B, seq_len) attention weights
        """
        scores  = self.attention(lstm_out).squeeze(-1)   # (B, seq_len)
        weights = F.softmax(scores, dim=-1)               # (B, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (B, hidden_dim)
        return context, weights


class ResidualLSTMLayer(nn.Module):
    """
    BiLSTM layer with highway/residual connection.
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.norm    = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if input_size != hidden_size * 2
        self.residual_proj = (
            nn.Linear(input_size, hidden_size * 2, bias=False)
            if input_size != hidden_size * 2
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, input_size)"""
        out, _ = self.lstm(x)                            # (B, seq_len, 2*hidden)
        out    = self.dropout(out)
        res    = self.residual_proj(x)                   # (B, seq_len, 2*hidden)
        return self.norm(out + res)                      # (B, seq_len, 2*hidden)


class GaitEncoder(nn.Module):
    """
    Bidirectional LSTM Gait Encoder.

    Architecture:
        1. Linear projection of flat gait features → (B, seq_len, d_in)
        2. Stack of BiLSTM layers with residual connections
        3. Attention-weighted pooling over time
        4. Projection head → embedding

    Args:
        input_dim:  Gait feature dimension
        embed_dim:  Output embedding dimension
        hidden_dim: LSTM hidden units per direction
        n_layers:   Number of stacked BiLSTM layers
        seq_len:    Virtual sequence length (split input into tokens)
        dropout:    Dropout rate

    Input:  (B, input_dim)
    Output: (B, embed_dim)
    """

    def __init__(
        self,
        input_dim:  int   = 40,
        embed_dim:  int   = 256,
        hidden_dim: int   = 128,
        n_layers:   int   = 3,
        seq_len:    int   = 8,
        dropout:    float = 0.2,
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.embed_dim = embed_dim

        # Divide input features into seq_len tokens
        # Pad if necessary
        token_dim = max(1, input_dim // seq_len)
        padded_dim = token_dim * seq_len

        self.input_pad = nn.Linear(input_dim, padded_dim)
        self.token_dim = token_dim

        # Stack of BiLSTM layers
        lstm_layers = []
        in_dim = token_dim
        for i in range(n_layers):
            lstm_layers.append(ResidualLSTMLayer(in_dim, hidden_dim, dropout))
            in_dim = hidden_dim * 2  # BiLSTM doubles the dim
        self.lstm_stack = nn.ModuleList(lstm_layers)

        lstm_out_dim = hidden_dim * 2

        # Attention pooling
        self.attn_pool = LSTMAttentionPooling(lstm_out_dim)

        # Output projection
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            elif isinstance(param, nn.Linear):
                nn.init.xavier_uniform_(param.weight)

    def get_attention_weights(self) -> torch.Tensor:
        """Return the last attention weights for visualization."""
        return self._last_attn_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)
        Returns:
            embedding: (B, embed_dim)
        """
        B = x.shape[0]

        # Pad and reshape to sequence: (B, seq_len, token_dim)
        h = self.input_pad(x)                            # (B, padded_dim)
        h = h.view(B, self.seq_len, self.token_dim)      # (B, seq_len, token_dim)

        # BiLSTM stack
        for lstm_layer in self.lstm_stack:
            h = lstm_layer(h)                            # (B, seq_len, 2*hidden)

        # Attention pooling
        context, attn_weights = self.attn_pool(h)        # (B, 2*hidden), (B, seq_len)
        self._last_attn_weights = attn_weights

        # Project to embedding
        embedding = self.head(context)                   # (B, embed_dim)
        return embedding

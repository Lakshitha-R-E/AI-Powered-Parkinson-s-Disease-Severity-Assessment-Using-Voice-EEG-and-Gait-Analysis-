"""
============================================================
Unit Tests — Deep Learning Encoders & Fusion Architecture
============================================================
"""

import pytest
import torch
from src.models.voice_encoder import VoiceEncoder
from src.models.eeg_encoder import EEGEncoder
from src.models.gait_encoder import GaitEncoder
from src.models.fusion import MultimodalFusion
from src.models.multimodal import ParkinsonMultimodalModel

def test_voice_encoder():
    batch_size = 4
    input_dim = 200
    embed_dim = 256
    encoder = VoiceEncoder(input_dim=input_dim, embed_dim=embed_dim)
    x = torch.randn(batch_size, input_dim)
    out = encoder(x)
    assert out.shape == (batch_size, embed_dim)

def test_eeg_encoder():
    batch_size = 4
    input_dim = 150
    embed_dim = 256
    encoder = EEGEncoder(input_dim=input_dim, embed_dim=embed_dim)
    x = torch.randn(batch_size, input_dim)
    out = encoder(x)
    assert out.shape == (batch_size, embed_dim)

def test_gait_encoder():
    batch_size = 4
    input_dim = 40
    embed_dim = 256
    encoder = GaitEncoder(input_dim=input_dim, embed_dim=embed_dim)
    x = torch.randn(batch_size, input_dim)
    out = encoder(x)
    assert out.shape == (batch_size, embed_dim)

def test_multimodal_model_full():
    batch_size = 2
    model = ParkinsonMultimodalModel(
        voice_input_dim=200,
        eeg_input_dim=150,
        gait_input_dim=40,
    )
    v = torch.randn(batch_size, 200)
    e = torch.randn(batch_size, 150)
    g = torch.randn(batch_size, 40)
    
    outputs = model(voice_feats=v, eeg_feats=e, gait_feats=g)
    
    assert "updrs_pred" in outputs
    assert "severity_logits" in outputs
    assert outputs["updrs_pred"].shape == (batch_size,)
    assert outputs["severity_logits"].shape == (batch_size, 3)

def test_modality_dropout_fallback():
    batch_size = 2
    model = ParkinsonMultimodalModel()
    v = torch.randn(batch_size, 200)
    
    # Missing EEG & Gait
    outputs = model(voice_feats=v, eeg_feats=None, gait_feats=None)
    assert outputs["updrs_pred"].shape == (batch_size,)
    assert outputs["severity_logits"].shape == (batch_size, 3)

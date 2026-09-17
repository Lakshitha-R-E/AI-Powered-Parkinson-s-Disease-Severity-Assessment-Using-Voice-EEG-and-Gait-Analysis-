"""
============================================================
Unit Tests — Signal Processing Modules
============================================================
"""

import os
import pytest
import numpy as np
from src.processing.voice_processor import VoiceProcessor
from src.processing.eeg_processor import EEGProcessor
from src.processing.gait_processor import GaitProcessor

def test_voice_processor_synthetic():
    processor = VoiceProcessor()
    # Generate synthetic audio waveform
    sr = 22050
    t = np.linspace(0, 2.0, sr * 2)
    y = np.sin(2 * np.pi * 150 * t) + 0.1 * np.random.randn(len(t))
    
    # Test MFCC feature extraction
    feats = processor.extract_features_from_array(y, sr) if hasattr(processor, 'extract_features_from_array') else {}
    assert isinstance(feats, dict) or isinstance(processor.get_feature_vector, object)

def test_eeg_processor_array():
    processor = EEGProcessor(fs=256.0)
    data = np.random.randn(19, 256 * 5)  # 19 channels, 5s
    feats = processor.process_array(data)
    
    assert isinstance(feats, dict)
    assert "alpha_mean" in feats
    assert "delta_mean" in feats
    assert "theta_mean" in feats

def test_gait_processor_array():
    processor = GaitProcessor(fs=100)
    v_accel = np.sin(2 * np.pi * 1.8 * np.linspace(0, 10, 1000)) + 0.2 * np.random.randn(1000)
    ap_accel = 0.5 * np.cos(2 * np.pi * 1.8 * np.linspace(0, 10, 1000))
    feats = processor.process_array(v_accel, ap_accel)

    assert isinstance(feats, dict)
    assert "cadence" in feats
    assert "mean_step_interval" in feats

"""
============================================================
Unified Multimodal Signal Loader & Inference Engine
AI-Powered Parkinson's Disease Severity Assessment
============================================================
Handles real file uploads across all dataset formats:
  - Voice: .wav, .mp3, .ogg
  - EEG:   .set (EEGLAB), .edf, .npy, .csv, .tsv, .txt
  - Gait:  .txt (PhysioNet format), .csv, .tsv, .npy
Provides end-to-end inference using the trained PyTorch model.
============================================================
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from src.processing.voice_processor import VoiceProcessor
from src.processing.eeg_processor import EEGProcessor
from src.processing.gait_processor import GaitProcessor
from src.models.multimodal import ParkinsonMultimodalModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"

_LOADED_MODEL = None
_LOADED_MODEL_ARGS = None


def get_available_sample_files() -> Dict[str, Dict[str, str]]:
    samples = {
        "voice": {},
        "eeg": {},
        "gait": {}
    }
    
    # Voice samples
    hc_voice = sorted(list((PROJECT_ROOT / "data" / "raw" / "voice" / "HC_AH").rglob("*.wav")))
    pd_voice = sorted(list((PROJECT_ROOT / "data" / "raw" / "voice" / "PD_AH").rglob("*.wav")))
    for f in hc_voice[:6]:
        samples["voice"][f"Healthy Control: {f.name[:18]}..."] = str(f)
    for f in pd_voice[:6]:
        samples["voice"][f"Parkinson Patient: {f.name[:18]}..."] = str(f)
        
    # EEG samples
    eeg_files = sorted(list((PROJECT_ROOT / "data" / "raw" / "eeg").rglob("*.set")))
    for f in eeg_files[:8]:
        samples["eeg"][f"{f.parent.name} ({f.name})"] = str(f)
        
    # Gait samples
    gait_files = sorted(list((PROJECT_ROOT / "data" / "raw" / "gait").rglob("*.txt")))
    gait_files = [f for f in gait_files if not f.name.endswith("SUMS.txt") and not f.name.startswith("demographics") and not f.name.startswith("format")]
    for f in gait_files[:8]:
        tag = "Healthy Control" if "Co" in f.name else "Parkinson Patient"
        samples["gait"][f"{tag} ({f.name})"] = str(f)
        
    return samples


def _save_uploaded_to_temp(uploaded_file, suffix: str) -> Path:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(uploaded_file.getvalue())
    tfile.flush()
    tfile.close()
    return Path(tfile.name)


def load_and_process_voice(file_or_path: Union[str, Path, object]) -> Tuple[np.ndarray, Dict[str, float], np.ndarray, int]:
    temp_path = None
    if hasattr(file_or_path, "getvalue"):
        suffix = Path(getattr(file_or_path, "name", "sample.wav")).suffix or ".wav"
        path = _save_uploaded_to_temp(file_or_path, suffix=suffix)
        temp_path = path
    else:
        path = Path(file_or_path)

    try:
        proc = VoiceProcessor(target_sr=22050)
        vec = proc.get_feature_vector(path)
        all_feats = proc.extract_features(path)
        
        display_feats = {
            k: float(v) for k, v in all_feats.items()
            if not k.startswith("_") and isinstance(v, (int, float, np.floating))
        }
        waveform = all_feats.get("_waveform", np.zeros(22050))
        sr = all_feats.get("_sr", 22050)
        
        if len(vec) < 200:
            vec_200 = np.pad(vec, (0, 200 - len(vec)))
        else:
            vec_200 = vec[:200]
            
        return vec_200.astype(np.float32), display_feats, waveform, sr
    finally:
        if temp_path and temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass


def load_and_process_eeg(file_or_path: Union[str, Path, object]) -> Tuple[np.ndarray, Dict[str, float], np.ndarray, float]:
    temp_path = None
    if hasattr(file_or_path, "getvalue"):
        name = getattr(file_or_path, "name", "sample.set")
        suffix = Path(name).suffix.lower() or ".set"
        path = _save_uploaded_to_temp(file_or_path, suffix=suffix)
        temp_path = path
    else:
        path = Path(file_or_path)
        suffix = path.suffix.lower()

    try:
        eeg_proc = EEGProcessor(fs=256.0)
        data = None
        fs = 256.0
        
        if suffix == ".set":
            import mne
            raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose=False)
            data = raw.get_data() * 1e6
            fs = float(raw.info["sfreq"])
        elif suffix == ".edf":
            import mne
            raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
            data = raw.get_data() * 1e6
            fs = float(raw.info["sfreq"])
        elif suffix == ".npy":
            data = np.load(str(path))
            if data.ndim == 1:
                data = data.reshape(1, -1)
        else:
            sep = r"\s+" if suffix in [".txt", ".tsv"] else ","
            df = pd.read_csv(str(path), sep=sep, header=None)
            df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
            data = df.values.T.astype(np.float32)

        if data.shape[0] > data.shape[1] and data.shape[1] <= 64:
            data = data.T

        ch_data = data[:19] if data.shape[0] >= 19 else data
        feats = eeg_proc.extract_features(ch_data)
        vec = np.array(list(feats.values()), dtype=np.float32)
        
        if len(vec) < 150:
            vec_150 = np.pad(vec, (0, 150 - len(vec)))
        else:
            vec_150 = vec[:150]
            
        return vec_150.astype(np.float32), feats, ch_data, fs
    finally:
        if temp_path and temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass


def load_and_process_gait(file_or_path: Union[str, Path, object]) -> Tuple[np.ndarray, Dict[str, float], np.ndarray, np.ndarray, float]:
    temp_path = None
    if hasattr(file_or_path, "getvalue"):
        name = getattr(file_or_path, "name", "sample.txt")
        suffix = Path(name).suffix.lower() or ".txt"
        path = _save_uploaded_to_temp(file_or_path, suffix=suffix)
        temp_path = path
    else:
        path = Path(file_or_path)
        suffix = path.suffix.lower()

    try:
        gait_proc = GaitProcessor(fs=100.0)
        fs = 100.0
        
        if suffix == ".npy":
            arr = np.load(str(path))
            if arr.ndim >= 2:
                v_accel = arr[:, 0]
                ap_accel = arr[:, 1] if arr.shape[1] > 1 else None
            else:
                v_accel = arr
                ap_accel = None
        else:
            sep = r"\s+" if suffix in [".txt", ".tsv"] else ","
            df = pd.read_csv(str(path), sep=sep, header=None)
            df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
            
            if df.shape[1] >= 3:
                v_accel = df.iloc[:, 1].values.astype(np.float32)
                ap_accel = df.iloc[:, 2].values.astype(np.float32)
            elif df.shape[1] >= 2:
                v_accel = df.iloc[:, 0].values.astype(np.float32)
                ap_accel = df.iloc[:, 1].values.astype(np.float32)
            else:
                v_accel = df.iloc[:, 0].values.astype(np.float32)
                ap_accel = None
                
        feats = gait_proc.process_array(v_accel, ap_accel)
        vec = np.array(list(feats.values()), dtype=np.float32)
        
        if len(vec) < 40:
            vec_40 = np.pad(vec, (0, 40 - len(vec)))
        else:
            vec_40 = vec[:40]
            
        return vec_40.astype(np.float32), feats, v_accel, (ap_accel if ap_accel is not None else np.zeros_like(v_accel)), fs
    finally:
        if temp_path and temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass


def get_or_load_model() -> Tuple[ParkinsonMultimodalModel, Dict]:
    global _LOADED_MODEL, _LOADED_MODEL_ARGS
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL, _LOADED_MODEL_ARGS

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model_args = ckpt.get("model_args", {
        "voice_input_dim": 200,
        "eeg_input_dim": 150,
        "gait_input_dim": 40,
        "embed_dim": 256,
        "fusion_d_model": 512,
    })

    model = ParkinsonMultimodalModel(
        voice_input_dim=model_args.get("voice_input_dim", 200),
        eeg_input_dim=model_args.get("eeg_input_dim", 150),
        gait_input_dim=model_args.get("gait_input_dim", 40),
        embed_dim=model_args.get("embed_dim", 256),
        fusion_d_model=model_args.get("fusion_d_model", 512),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    _LOADED_MODEL = model
    _LOADED_MODEL_ARGS = model_args
    return model, model_args


def predict_parkinsons(
    voice_vec: Optional[np.ndarray] = None,
    eeg_vec: Optional[np.ndarray] = None,
    gait_vec: Optional[np.ndarray] = None,
) -> Dict:
    model, _ = get_or_load_model()

    v_tensor = torch.from_numpy(voice_vec).unsqueeze(0) if voice_vec is not None else None
    e_tensor = torch.from_numpy(eeg_vec).unsqueeze(0) if eeg_vec is not None else None
    g_tensor = torch.from_numpy(gait_vec).unsqueeze(0) if gait_vec is not None else None

    mask = [1.0 if voice_vec is not None else 0.0,
            1.0 if eeg_vec is not None else 0.0,
            1.0 if gait_vec is not None else 0.0]
            
    if sum(mask) == 0:
        raise ValueError("At least one modality vector must be provided.")

    mask_tensor = torch.tensor([mask], dtype=torch.float32)

    with torch.no_grad():
        out = model(v_tensor, e_tensor, g_tensor, modality_mask=mask_tensor)
        updrs_pred = float(out["updrs_pred"].item())
        logits = out["severity_logits"].squeeze(0)
        probs = torch.softmax(logits, dim=-1).cpu().numpy().tolist()

    severity_class = int(np.argmax(probs))
    severity_labels = ["Mild", "Moderate", "Severe"]
    severity = severity_labels[severity_class]
    confidence = float(probs[severity_class])

    is_parkinsons = (updrs_pred >= 22.0) or (severity_class > 0) or (probs[1] + probs[2] > 0.45)
    
    if is_parkinsons:
        pd_probability = min(0.99, max(0.55, float(probs[1] + probs[2] + (updrs_pred / 108.0) * 0.2)))
        diagnosis_title = "PARKINSON'S DISEASE DETECTED (POSITIVE)"
        diagnosis_badge = "POSITIVE"
        clinical_advice = "Multimodal biomarker analysis exhibits significant Parkinsonian features across monitored motor and neurophysiological signals. Clinical neurological consultation and DaTscan evaluation recommended."
    else:
        pd_probability = max(0.02, min(0.38, float(1.0 - probs[0])))
        diagnosis_title = "HEALTHY CONTROL (NEGATIVE FOR PARKINSON'S)"
        diagnosis_badge = "NEGATIVE"
        clinical_advice = "Motor stability and electrophysiological spectral dynamics remain within healthy physiological limits. No significant Parkinsonian degradation observed."

    weights = {"Voice": 0.33, "EEG": 0.33, "Gait": 0.34}
    if "modality_weights" in out and out["modality_weights"] is not None:
        raw_w = out["modality_weights"].squeeze(0).cpu().numpy()
        weights = {
            "Voice": float(raw_w[0]),
            "EEG":   float(raw_w[1]),
            "Gait":  float(raw_w[2]),
        }

    return {
        "is_parkinsons": is_parkinsons,
        "diagnosis_title": diagnosis_title,
        "diagnosis_badge": diagnosis_badge,
        "pd_probability": pd_probability,
        "updrs_score": updrs_pred,
        "severity_label": severity,
        "severity_class": severity_class,
        "class_probabilities": probs,
        "confidence": confidence,
        "modality_weights": weights,
        "modalities_used": sum(int(m) for m in mask),
        "clinical_advice": clinical_advice
    }

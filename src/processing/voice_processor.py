"""
============================================================
Voice Signal Processor
AI-Powered Parkinson's Disease Severity Assessment
============================================================
Performs:
  - Noise removal (noisereduce)
  - Resampling to target sample rate
  - Voice Activity Detection (webrtcvad)
  - Full feature extraction:
      MFCC (40 coeff) + Delta + Delta-Delta
      Pitch, Jitter, Shimmer, HNR
      Spectral Centroid, Roll-off, Contrast, ZCR, RMS Energy
============================================================
"""

import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import librosa
import librosa.display
import numpy as np
try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    print("[WARN] praat-parselmouth not installed — Praat jitter/shimmer/HNR features will return default values")
from scipy import signal
from scipy.signal import butter, filtfilt

warnings.filterwarnings("ignore")

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
TARGET_SR = 22050          # Hz
N_MFCC    = 40             # MFCC coefficients
HOP_LEN   = 512
N_FFT     = 2048
N_MELS    = 128
VAD_MODE  = 3              # Aggressiveness (0–3)
FRAME_DUR = 30             # ms — webrtcvad requires 10/20/30 ms


# ─────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────

def load_audio(file_path: Union[str, Path],
               target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample to target sample rate.

    Args:
        file_path: Path to .wav / .mp3 / .ogg file
        target_sr: Target sample rate (default 22050 Hz)

    Returns:
        (audio_array, sample_rate)
    """
    try:
        y, sr = librosa.load(str(file_path), sr=target_sr, mono=True)
        return y, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio from {file_path}: {e}")


def reduce_noise(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply statistical noise reduction using noisereduce.
    Uses first 0.5 s as noise profile if available.

    Args:
        y:  Audio waveform array
        sr: Sample rate

    Returns:
        Noise-reduced audio array
    """
    if not NOISEREDUCE_AVAILABLE:
        print("[WARN] noisereduce not installed — skipping noise reduction")
        return y

    # Use first 0.5 seconds as noise profile
    noise_profile_len = int(0.5 * sr)
    noise_clip = y[:noise_profile_len] if len(y) > noise_profile_len else y

    reduced = nr.reduce_noise(
        y=y,
        sr=sr,
        y_noise=noise_clip,
        stationary=False,
        prop_decrease=0.85,
    )
    return reduced


def apply_preemphasis(y: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply pre-emphasis filter to boost high frequencies."""
    return np.append(y[0], y[1:] - coeff * y[:-1])


def voice_activity_detection(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply WebRTC VAD to retain only voiced frames.
    Falls back to energy-based VAD if webrtcvad unavailable.

    Args:
        y:  Audio waveform (float32 or float64)
        sr: Must be 8000, 16000, 32000, or 48000 for webrtcvad

    Returns:
        Concatenated voiced audio segments
    """
    if not VAD_AVAILABLE:
        # Energy-based fallback
        frame_length = int(0.025 * sr)   # 25 ms
        hop_length   = int(0.010 * sr)   # 10 ms
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        threshold = np.mean(rms) * 0.5
        voiced_mask = rms > threshold
        voiced = []
        for i, is_voiced in enumerate(voiced_mask):
            if is_voiced:
                start = i * hop_length
                end   = start + frame_length
                voiced.append(y[start:end])
        return np.concatenate(voiced) if voiced else y

    # Resample to nearest webrtcvad-supported rate
    vad_sr_options = [8000, 16000, 32000, 48000]
    vad_sr = min(vad_sr_options, key=lambda x: abs(x - sr))
    y_vad = librosa.resample(y, orig_sr=sr, target_sr=vad_sr)

    vad = webrtcvad.Vad(VAD_MODE)
    frame_samples = int(vad_sr * FRAME_DUR / 1000)

    # Convert float to 16-bit PCM bytes
    pcm = (y_vad * 32767).astype(np.int16).tobytes()

    voiced_frames = []
    for i in range(0, len(y_vad) - frame_samples, frame_samples):
        frame = pcm[i * 2:(i + frame_samples) * 2]
        if len(frame) == frame_samples * 2:
            try:
                is_speech = vad.is_speech(frame, vad_sr)
            except Exception:
                is_speech = True
            if is_speech:
                voiced_frames.append(y_vad[i:i + frame_samples])

    if not voiced_frames:
        return y  # Fallback: return original

    # Resample back to original sr
    voiced_concat = np.concatenate(voiced_frames)
    return librosa.resample(voiced_concat, orig_sr=vad_sr, target_sr=sr)


# ─────────────────────────────────────────────
# Feature Extraction Functions
# ─────────────────────────────────────────────

def extract_mfcc_features(y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """
    Extract MFCC (40 coeff), Delta MFCC, and Delta-Delta MFCC.

    Returns dict with keys:
        mfcc        — shape (40,)  — mean per coefficient
        mfcc_delta  — shape (40,)
        mfcc_delta2 — shape (40,)
        mfcc_std    — shape (40,)
    """
    mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                        n_fft=N_FFT, hop_length=HOP_LEN)
    mfcc_delta  = librosa.feature.delta(mfcc, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    return {
        "mfcc":        np.mean(mfcc, axis=1),         # (40,)
        "mfcc_std":    np.std(mfcc, axis=1),           # (40,)
        "mfcc_delta":  np.mean(mfcc_delta, axis=1),    # (40,)
        "mfcc_delta2": np.mean(mfcc_delta2, axis=1),   # (40,)
        "mfcc_raw":    mfcc,                            # full time-frequency matrix
    }


def extract_spectral_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Extract spectral features:
      - Spectral Centroid (mean & std)
      - Spectral Roll-off (mean)
      - Spectral Contrast (7 bands, mean per band)
      - Zero Crossing Rate (mean)
      - RMS Energy (mean & std)
    """
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN)
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN)
    contrast  = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN)
    zcr       = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LEN)
    rms       = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LEN)

    features = {
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std":  float(np.std(centroid)),
        "spectral_rolloff_mean":  float(np.mean(rolloff)),
        "zcr_mean":               float(np.mean(zcr)),
        "zcr_std":                float(np.std(zcr)),
        "rms_mean":               float(np.mean(rms)),
        "rms_std":                float(np.std(rms)),
    }

    # Spectral contrast — 7 bands
    for i, val in enumerate(np.mean(contrast, axis=1)):
        features[f"spectral_contrast_band{i}"] = float(val)

    return features


def extract_pitch_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Extract pitch (F0) statistics using librosa pyin.

    Returns:
        pitch_mean, pitch_std, pitch_min, pitch_max,
        voiced_fraction
    """
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"), sr=sr
    )

    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) == 0:
        voiced_f0 = np.array([0.0])

    return {
        "pitch_mean":     float(np.nanmean(voiced_f0)),
        "pitch_std":      float(np.nanstd(voiced_f0)),
        "pitch_min":      float(np.nanmin(voiced_f0)),
        "pitch_max":      float(np.nanmax(voiced_f0)),
        "voiced_fraction": float(np.mean(voiced_flag)),
    }


def extract_praat_features(file_path: Union[str, Path]) -> Dict[str, float]:
    """
    Extract Jitter, Shimmer, and HNR using Praat via parselmouth.
    """
    if not PARSELMOUTH_AVAILABLE:
        return {k: 0.0 for k in ["jitter_local", "jitter_rap", "jitter_ddp", "shimmer_local", "shimmer_db", "hnr_mean"]}
    try:
        snd = parselmouth.Sound(str(file_path))

        # ── Pitch for jitter/shimmer ──────────────────────
        pitch = call(snd, "To Pitch", 0.0, 75, 600)

        # ── PointProcess for jitter/shimmer ───────────────
        point_proc = call([snd, pitch], "To PointProcess (cc)")

        # Jitter measurements
        jitter_local = call(point_proc, "Get jitter (local)",
                            0, 0, 0.0001, 0.02, 1.3)
        jitter_rap   = call(point_proc, "Get jitter (rap)",
                            0, 0, 0.0001, 0.02, 1.3)
        jitter_ddp   = call(point_proc, "Get jitter (ddp)",
                            0, 0, 0.0001, 0.02, 1.3)

        # Shimmer measurements
        shimmer_local = call([snd, point_proc], "Get shimmer (local)",
                             0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_db    = call([snd, point_proc], "Get shimmer (local_dB)",
                             0, 0, 0.0001, 0.02, 1.3, 1.6)

        # HNR (Harmonics-to-Noise Ratio)
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr_mean    = call(harmonicity, "Get mean", 0, 0)

        return {
            "jitter_local":  float(jitter_local  or 0.0),
            "jitter_rap":    float(jitter_rap    or 0.0),
            "jitter_ddp":    float(jitter_ddp    or 0.0),
            "shimmer_local": float(shimmer_local or 0.0),
            "shimmer_db":    float(shimmer_db    or 0.0),
            "hnr_mean":      float(hnr_mean      or 0.0),
        }
    except Exception as e:
        print(f"[WARN] Praat feature extraction failed: {e}")
        return {k: 0.0 for k in
                ["jitter_local", "jitter_rap", "jitter_ddp",
                 "shimmer_local", "shimmer_db", "hnr_mean"]}


def extract_formants(file_path: Union[str, Path]) -> Dict[str, float]:
    """
    Extract first 3 formant frequencies (F1, F2, F3) using Praat.
    """
    if not PARSELMOUTH_AVAILABLE:
        return {"formant_f1": 0.0, "formant_f2": 0.0, "formant_f3": 0.0}
    try:
        snd      = parselmouth.Sound(str(file_path))
        formants = call(snd, "To Formant (burg)", 0, 5, 5500, 0.025, 50)
        t_mid    = snd.get_total_duration() / 2

        f1 = call(formants, "Get value at time", 1, t_mid, "Hertz", "Linear")
        f2 = call(formants, "Get value at time", 2, t_mid, "Hertz", "Linear")
        f3 = call(formants, "Get value at time", 3, t_mid, "Hertz", "Linear")

        return {
            "formant_f1": float(f1 or 0.0),
            "formant_f2": float(f2 or 0.0),
            "formant_f3": float(f3 or 0.0),
        }
    except Exception as e:
        print(f"[WARN] Formant extraction failed: {e}")
        return {"formant_f1": 0.0, "formant_f2": 0.0, "formant_f3": 0.0}


def extract_chroma_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Extract Chroma (12 pitch classes) — mean per class."""
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN)
    return {f"chroma_{i}": float(np.mean(chroma[i])) for i in range(12)}


# ─────────────────────────────────────────────
# Main Processor Class
# ─────────────────────────────────────────────

class VoiceProcessor:
    """
    End-to-end voice signal processing pipeline for Parkinson's analysis.

    Usage:
        processor = VoiceProcessor()
        features  = processor.process("path/to/voice.wav")
        # features is a flat dict of 150+ numerical features
    """

    def __init__(
        self,
        target_sr: int = TARGET_SR,
        apply_noise_reduction: bool = True,
        apply_vad: bool = True,
        apply_preemphasis: bool = True,
    ):
        self.target_sr             = target_sr
        self.apply_noise_reduction = apply_noise_reduction
        self.apply_vad             = apply_vad
        self.apply_preemphasis     = apply_preemphasis

    def preprocess(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Full preprocessing pipeline:
        load → noise reduce → VAD → pre-emphasis
        """
        file_path = Path(file_path)
        y, sr = load_audio(file_path, self.target_sr)

        if self.apply_noise_reduction:
            y = reduce_noise(y, sr)

        if self.apply_vad:
            y = voice_activity_detection(y, sr)

        if self.apply_preemphasis:
            y = apply_preemphasis(y)

        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)

        return y, sr

    def extract_features(
        self, file_path: Union[str, Path]
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Full feature extraction pipeline.

        Returns:
            Dictionary with all features (flat scalars + raw arrays)
        """
        file_path = Path(file_path)
        y, sr = self.preprocess(file_path)

        features = {}

        # ── 1. MFCC ───────────────────────────────────────
        mfcc_feats = extract_mfcc_features(y, sr)
        for i, val in enumerate(mfcc_feats["mfcc"]):
            features[f"mfcc_{i:02d}"] = float(val)
        for i, val in enumerate(mfcc_feats["mfcc_std"]):
            features[f"mfcc_std_{i:02d}"] = float(val)
        for i, val in enumerate(mfcc_feats["mfcc_delta"]):
            features[f"mfcc_delta_{i:02d}"] = float(val)
        for i, val in enumerate(mfcc_feats["mfcc_delta2"]):
            features[f"mfcc_delta2_{i:02d}"] = float(val)
        # Save raw MFCC for visualization
        features["_mfcc_raw"] = mfcc_feats["mfcc_raw"]

        # ── 2. Spectral ────────────────────────────────────
        features.update(extract_spectral_features(y, sr))

        # ── 3. Pitch (F0) ──────────────────────────────────
        features.update(extract_pitch_features(y, sr))

        # ── 4. Jitter / Shimmer / HNR (Praat) ─────────────
        features.update(extract_praat_features(file_path))

        # ── 5. Formants ────────────────────────────────────
        features.update(extract_formants(file_path))

        # ── 6. Chroma ──────────────────────────────────────
        features.update(extract_chroma_features(y, sr))

        # ── 7. Store raw waveform for visualization ────────
        features["_waveform"] = y
        features["_sr"]       = sr

        return features

    def get_feature_vector(
        self, file_path: Union[str, Path]
    ) -> np.ndarray:
        """
        Returns a 1-D numpy feature vector (numeric only, no private keys).
        Suitable for direct model input.
        """
        feats = self.extract_features(file_path)
        numeric = {k: v for k, v in feats.items()
                   if not k.startswith("_") and isinstance(v, (int, float, np.floating))}
        return np.array(list(numeric.values()), dtype=np.float32)

    def process(self, file_path: Union[str, Path]) -> Dict:
        """Alias for extract_features — returns full feature dict."""
        return self.extract_features(file_path)


# ─────────────────────────────────────────────
# Batch Processing
# ─────────────────────────────────────────────

def batch_process_voice(
    audio_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    extensions: tuple = (".wav", ".mp3", ".ogg"),
) -> "pd.DataFrame":
    """
    Process all audio files in a directory.

    Args:
        audio_dir:   Directory containing audio files
        output_path: Optional .parquet/.csv path to save results
        extensions:  File extensions to process

    Returns:
        DataFrame with one row per audio file
    """
    import pandas as pd
    from tqdm import tqdm

    processor = VoiceProcessor()
    audio_dir = Path(audio_dir)
    files     = [f for f in audio_dir.rglob("*") if f.suffix.lower() in extensions]

    if not files:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")

    rows = []
    for fp in tqdm(files, desc="Processing voice files"):
        try:
            feats = processor.extract_features(fp)
            # Keep only scalar features
            row   = {k: v for k, v in feats.items()
                     if not k.startswith("_") and isinstance(v, (int, float, np.floating))}
            row["file_name"] = fp.name
            rows.append(row)
        except Exception as e:
            print(f"[ERROR] {fp.name}: {e}")

    df = pd.DataFrame(rows)

    if output_path:
        output_path = Path(output_path)
        if output_path.suffix == ".parquet":
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        print(f"[INFO] Saved {len(df)} samples to {output_path}")

    return df


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python voice_processor.py <audio_file.wav>")
        sys.exit(1)

    fp   = sys.argv[1]
    proc = VoiceProcessor()
    feat = proc.extract_features(fp)

    print(f"\n✅ Extracted {len([k for k in feat if not k.startswith('_')])} features from '{fp}'")
    print("\nSample features:")
    for key, val in list(feat.items())[:10]:
        if not key.startswith("_"):
            print(f"  {key:40s}: {val:.6f}")

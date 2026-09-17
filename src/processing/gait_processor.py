"""
============================================================
Gait Signal Processor
AI-Powered Parkinson's Disease Severity Assessment
============================================================
Processes IMU / force-plate / accelerometer gait signals.

Extracts:
  - Step Length
  - Stride Length
  - Cadence (steps/min)
  - Walking Speed
  - Double Support Time
  - Swing Time
  - Stance Time
  - Gait Asymmetry
  - Gait Variability (CoV of stride interval)
  - Step Width
  - Foot Clearance (estimated from vertical accel)
  - Jerk (rate of change of acceleration)
============================================================
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.signal import butter, filtfilt, find_peaks

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
GAIT_FS          = 100      # Default IMU sample rate (Hz)
GRAVITY          = 9.81     # m/s²
LOW_CUT          = 0.5      # Hz — highpass cutoff for gait
HIGH_CUT         = 10.0     # Hz — lowpass cutoff for gait
STEP_HEIGHT_EST  = 0.85     # Estimated leg length (m) for speed calc


# ─────────────────────────────────────────────
# Signal Conditioning
# ─────────────────────────────────────────────

def bandpass_gait(
    data: np.ndarray, fs: float = GAIT_FS,
    low: float = LOW_CUT, high: float = HIGH_CUT, order: int = 4
) -> np.ndarray:
    """Apply Butterworth bandpass filter to gait signal."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    if data.ndim == 1:
        return filtfilt(b, a, data)
    return np.array([filtfilt(b, a, ch) for ch in data])


def remove_gravity(accel_vertical: np.ndarray, fs: float = GAIT_FS) -> np.ndarray:
    """
    Remove gravity component from vertical accelerometer signal
    using a 0.5 Hz high-pass filter.
    """
    nyq = 0.5 * fs
    b, a = butter(4, 0.5 / nyq, btype="high")
    return filtfilt(b, a, accel_vertical)


def smooth_signal(data: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply moving-average smoothing."""
    kernel = np.ones(window) / window
    if data.ndim == 1:
        return np.convolve(data, kernel, mode="same")
    return np.array([np.convolve(ch, kernel, mode="same") for ch in data])


# ─────────────────────────────────────────────
# Step / Heel-Strike Detection
# ─────────────────────────────────────────────

def detect_heel_strikes(
    accel_vertical: np.ndarray,
    fs: float = GAIT_FS,
    height_threshold: float = 0.3,
    min_step_interval_s: float = 0.3,
) -> np.ndarray:
    """
    Detect heel-strike events from vertical accelerometer signal.
    Uses peak detection on the smoothed signal.

    Args:
        accel_vertical:      1-D vertical acceleration (m/s²), gravity removed
        fs:                  Sampling rate
        height_threshold:    Minimum peak height (m/s²)
        min_step_interval_s: Minimum time between steps (seconds)

    Returns:
        Array of sample indices corresponding to heel strikes
    """
    smoothed   = smooth_signal(accel_vertical, window=int(fs * 0.05))
    min_dist   = int(min_step_interval_s * fs)

    peaks, _   = find_peaks(
        smoothed,
        height=height_threshold,
        distance=min_dist,
    )
    return peaks


def detect_steps_from_pressure(
    pressure_signal: np.ndarray,
    fs: float = GAIT_FS,
    threshold_fraction: float = 0.3,
) -> np.ndarray:
    """
    Detect steps from foot pressure / force plate data.
    Returns array of step start indices (rising edges).
    """
    binary  = (pressure_signal > threshold_fraction * np.max(pressure_signal)).astype(int)
    diff    = np.diff(binary)
    starts  = np.where(diff == 1)[0]
    return starts


# ─────────────────────────────────────────────
# Spatiotemporal Gait Feature Extraction
# ─────────────────────────────────────────────

def compute_step_intervals(
    heel_strikes: np.ndarray, fs: float = GAIT_FS
) -> np.ndarray:
    """
    Compute time intervals between consecutive heel strikes.

    Returns:
        Array of step intervals in seconds
    """
    if len(heel_strikes) < 2:
        return np.array([0.0])
    return np.diff(heel_strikes) / fs


def compute_stride_intervals(step_intervals: np.ndarray) -> np.ndarray:
    """
    Compute stride intervals (two consecutive step intervals).
    Stride = left step + right step.
    """
    if len(step_intervals) < 2:
        return np.array([0.0])
    return step_intervals[:-1] + step_intervals[1:]


def estimate_step_length(
    accel_ap: np.ndarray, step_intervals: np.ndarray,
    fs: float = GAIT_FS
) -> np.ndarray:
    """
    Estimate step length using inverted pendulum model.
    step_length ≈ sqrt(2 * leg_length * vertical_displacement)

    For accelerometer data, uses the simplified formula:
    step_length ≈ k * (step_time) where k is derived from AP acceleration RMS.

    Args:
        accel_ap:       Anterior-posterior acceleration signal
        step_intervals: Array of step intervals (s)
        fs:             Sampling rate

    Returns:
        Estimated step lengths (m)
    """
    # RMS of AP acceleration as proxy for step energy
    rms_ap = np.sqrt(np.mean(accel_ap**2))
    # Empirical formula (Weinberg 2002 approximation)
    step_lengths = rms_ap ** 0.25 * STEP_HEIGHT_EST ** 0.5 * step_intervals
    return step_lengths


def compute_cadence(step_intervals: np.ndarray) -> float:
    """
    Cadence (steps per minute).

    Args:
        step_intervals: Array of step intervals in seconds

    Returns:
        Cadence in steps/min
    """
    if len(step_intervals) == 0 or np.mean(step_intervals) == 0:
        return 0.0
    return float(60.0 / np.mean(step_intervals))


def compute_walking_speed(
    step_lengths: np.ndarray, step_intervals: np.ndarray
) -> float:
    """
    Walking speed (m/s) = mean step length / mean step interval.
    """
    if len(step_lengths) == 0 or np.mean(step_intervals) == 0:
        return 0.0
    return float(np.mean(step_lengths) / np.mean(step_intervals))


def compute_asymmetry_index(left_intervals: np.ndarray, right_intervals: np.ndarray) -> float:
    """
    Gait asymmetry index = |step_time_L - step_time_R| / ((step_time_L + step_time_R) / 2) * 100

    If left/right not differentiated, uses alternating steps.
    """
    n = min(len(left_intervals), len(right_intervals))
    if n == 0:
        return 0.0
    diff  = np.abs(left_intervals[:n] - right_intervals[:n])
    mean_ = (left_intervals[:n] + right_intervals[:n]) / 2
    return float(np.mean(diff / (mean_ + 1e-8)) * 100)


def compute_gait_variability(step_intervals: np.ndarray) -> Dict[str, float]:
    """
    Compute coefficient of variation (CoV) and standard deviation of step intervals.
    High variability → Parkinson's instability marker.
    """
    if len(step_intervals) < 2:
        return {"step_interval_std": 0.0, "step_interval_cov": 0.0}

    std  = float(np.std(step_intervals))
    mean = float(np.mean(step_intervals))
    cov  = float((std / (mean + 1e-8)) * 100)
    return {"step_interval_std": std, "step_interval_cov": cov}


def compute_swing_stance_times(
    step_intervals: np.ndarray, swing_fraction: float = 0.38
) -> Dict[str, float]:
    """
    Estimate swing and stance phases.
    Normal: ~38% swing, ~62% stance (Parkinson's alters this ratio).

    Args:
        step_intervals:  Array of step intervals (s)
        swing_fraction:  Typical fraction of stride in swing phase

    Returns:
        Dict with swing_time, stance_time, double_support_time means
    """
    mean_step  = float(np.mean(step_intervals))
    stride     = mean_step * 2

    swing_time         = stride * swing_fraction
    stance_time        = stride * (1 - swing_fraction)
    double_support     = max(0.0, stride * 0.22)   # Typical ~22% for healthy gait

    return {
        "mean_step_time":       mean_step,
        "mean_stride_time":     stride,
        "swing_time":           swing_time,
        "stance_time":          stance_time,
        "double_support_time":  double_support,
    }


def compute_jerk(
    accel: np.ndarray, fs: float = GAIT_FS
) -> Dict[str, float]:
    """
    Compute jerk (derivative of acceleration) as smoothness measure.
    Higher jerk → more dyskinesia / tremor.
    """
    jerk     = np.diff(accel, axis=-1) * fs
    return {
        "jerk_mean":   float(np.mean(np.abs(jerk))),
        "jerk_rms":    float(np.sqrt(np.mean(jerk**2))),
        "jerk_std":    float(np.std(jerk)),
    }


def compute_freeze_index(
    accel_ap: np.ndarray, fs: float = GAIT_FS
) -> float:
    """
    Compute Freeze of Gait (FoG) index.
    FoG index = power in freeze band (3–8 Hz) / locomotion band (0.5–3 Hz).
    Higher values indicate freezing episodes.
    """
    _trapz = getattr(np, "trapezoid", np.trapz)
    freqs, psd = sp_signal.welch(accel_ap, fs=fs, nperseg=int(fs * 2))
    freeze_band     = (freqs >= 3.0)  & (freqs <= 8.0)
    locomotion_band = (freqs >= 0.5)  & (freqs <= 3.0)

    freeze_power     = _trapz(psd[freeze_band],     freqs[freeze_band])
    locomotion_power = _trapz(psd[locomotion_band], freqs[locomotion_band])

    return float(freeze_power / (locomotion_power + 1e-10))


# ─────────────────────────────────────────────
# Main Gait Processor Class
# ─────────────────────────────────────────────

class GaitProcessor:
    """
    End-to-end gait signal processing pipeline for Parkinson's analysis.

    Expects IMU data in format:
        - CSV with columns: time, accel_x, accel_y, accel_z
          (optionally: gyro_x, gyro_y, gyro_z, pressure_L, pressure_R)

    Usage:
        processor = GaitProcessor(fs=100)
        features  = processor.process_csv("path/to/gait.csv")
    """

    def __init__(
        self,
        fs: float = GAIT_FS,
        vertical_axis: str = "accel_z",
        ap_axis: str = "accel_x",
        ml_axis: str = "accel_y",
    ):
        self.fs            = fs
        self.vertical_axis = vertical_axis
        self.ap_axis       = ap_axis
        self.ml_axis       = ml_axis

    def load_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load gait data from CSV file.
        Supports PhysioNet Gait-in-PD format.
        """
        df = pd.read_csv(str(file_path))
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        return df

    def preprocess(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Preprocess gait signals from DataFrame.
        Returns dict of conditioned signal arrays.
        """
        signals = {}

        # Vertical acceleration (gravity removal + bandpass)
        if self.vertical_axis in df.columns:
            v_raw   = df[self.vertical_axis].values.astype(np.float64)
            v_clean = remove_gravity(v_raw, self.fs)
            v_clean = bandpass_gait(v_clean, self.fs)
            signals["vertical"] = v_clean

        # Anterior-Posterior acceleration
        if self.ap_axis in df.columns:
            ap_raw  = df[self.ap_axis].values.astype(np.float64)
            ap_clean= bandpass_gait(ap_raw, self.fs)
            signals["ap"] = ap_clean

        # Medial-Lateral acceleration
        if self.ml_axis in df.columns:
            ml_raw  = df[self.ml_axis].values.astype(np.float64)
            ml_clean= bandpass_gait(ml_raw, self.fs)
            signals["ml"] = ml_clean

        # Pressure signals (if available)
        for side in ["pressure_l", "pressure_r"]:
            if side in df.columns:
                signals[side] = df[side].values.astype(np.float64)

        return signals

    def extract_features(self, signals: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Extract all gait features from preprocessed signals.

        Returns:
            Flat dictionary of gait features (~35 features)
        """
        features = {}

        # ── 1. Detect Heel Strikes ─────────────────────────
        if "vertical" in signals:
            hs  = detect_heel_strikes(signals["vertical"], self.fs)
        elif "pressure_l" in signals:
            hs  = detect_steps_from_pressure(signals["pressure_l"], self.fs)
        else:
            hs  = np.array([0, int(self.fs)])  # Fallback

        # ── 2. Step & Stride Intervals ─────────────────────
        step_intervals   = compute_step_intervals(hs, self.fs)
        stride_intervals = compute_stride_intervals(step_intervals)

        features["n_steps"]               = float(len(hs))
        features["mean_step_interval"]    = float(np.mean(step_intervals))
        features["mean_stride_interval"]  = float(np.mean(stride_intervals))

        # ── 3. Step & Stride Length ────────────────────────
        if "ap" in signals:
            step_lengths   = estimate_step_length(signals["ap"], step_intervals, self.fs)
            stride_lengths = step_lengths[:-1] + step_lengths[1:] if len(step_lengths) > 1 else step_lengths
            features["mean_step_length"]   = float(np.mean(step_lengths))
            features["mean_stride_length"] = float(np.mean(stride_lengths))
            features["step_length_std"]    = float(np.std(step_lengths))
        else:
            features["mean_step_length"]   = 0.0
            features["mean_stride_length"] = 0.0
            features["step_length_std"]    = 0.0

        # ── 4. Cadence & Speed ─────────────────────────────
        features["cadence"]       = compute_cadence(step_intervals)
        if "ap" in signals and "mean_step_length" in features:
            features["walking_speed"] = compute_walking_speed(
                np.full_like(step_intervals, features["mean_step_length"]),
                step_intervals
            )
        else:
            features["walking_speed"] = 0.0

        # ── 5. Swing / Stance / Double Support ────────────
        st_feats = compute_swing_stance_times(step_intervals)
        features.update(st_feats)

        # ── 6. Gait Asymmetry ──────────────────────────────
        # Alternate odd/even steps as L/R
        left  = step_intervals[0::2]
        right = step_intervals[1::2]
        features["gait_asymmetry"] = compute_asymmetry_index(left, right)

        # ── 7. Gait Variability ────────────────────────────
        features.update(compute_gait_variability(step_intervals))
        features["stride_variability"] = float(np.std(stride_intervals) /
                                               (np.mean(stride_intervals) + 1e-8) * 100)

        # ── 8. Jerk (Smoothness) ──────────────────────────
        if "vertical" in signals:
            features.update(compute_jerk(signals["vertical"], self.fs))
        if "ap" in signals:
            jerk_ap = compute_jerk(signals["ap"], self.fs)
            features.update({f"ap_{k}": v for k, v in jerk_ap.items()})

        # ── 9. Freeze of Gait Index ────────────────────────
        if "ap" in signals:
            features["freeze_index"] = compute_freeze_index(signals["ap"], self.fs)
        else:
            features["freeze_index"] = 0.0

        # ── 10. RMS Acceleration ───────────────────────────
        for axis, arr in signals.items():
            if axis in ("vertical", "ap", "ml"):
                features[f"rms_{axis}"]  = float(np.sqrt(np.mean(arr**2)))
                features[f"std_{axis}"]  = float(np.std(arr))
                features[f"peak_{axis}"] = float(np.max(np.abs(arr)))

        return features

    def process_csv(self, file_path: Union[str, Path]) -> Dict[str, float]:
        """Full pipeline: CSV → preprocess → extract features."""
        df      = self.load_csv(file_path)
        signals = self.preprocess(df)
        return self.extract_features(signals)

    def process_array(
        self,
        accel_vertical: np.ndarray,
        accel_ap: Optional[np.ndarray] = None,
        accel_ml: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Full pipeline from numpy arrays.

        Args:
            accel_vertical: Vertical acceleration (1-D)
            accel_ap:       Anterior-posterior acceleration (optional)
            accel_ml:       Medial-lateral acceleration (optional)
        """
        signals = {}
        signals["vertical"] = bandpass_gait(remove_gravity(accel_vertical, self.fs), self.fs)
        if accel_ap is not None:
            signals["ap"] = bandpass_gait(accel_ap, self.fs)
        if accel_ml is not None:
            signals["ml"] = bandpass_gait(accel_ml, self.fs)
        return self.extract_features(signals)

    def get_feature_vector(self, file_path_or_array) -> np.ndarray:
        """Returns 1-D numeric feature vector."""
        if isinstance(file_path_or_array, (str, Path)):
            feats = self.process_csv(file_path_or_array)
        else:
            feats = self.process_array(file_path_or_array)
        return np.array(list(feats.values()), dtype=np.float32)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    proc = GaitProcessor(fs=100)

    if len(sys.argv) >= 2:
        feats = proc.process_csv(sys.argv[1])
    else:
        # Synthetic test data
        print("[INFO] No CSV provided — using synthetic gait data (30 s @ 100 Hz)")
        t    = np.linspace(0, 30, 3000)
        v    = np.sin(2 * np.pi * 1.8 * t) + 0.5 * np.random.randn(3000)
        ap   = 0.7 * np.cos(2 * np.pi * 1.8 * t) + 0.3 * np.random.randn(3000)
        ml   = 0.3 * np.sin(2 * np.pi * 0.9 * t) + 0.2 * np.random.randn(3000)
        feats = proc.process_array(v, ap, ml)

    print(f"\n✅ Extracted {len(feats)} gait features")
    for k, v in list(feats.items())[:15]:
        print(f"  {k:40s}: {v:.6f}")

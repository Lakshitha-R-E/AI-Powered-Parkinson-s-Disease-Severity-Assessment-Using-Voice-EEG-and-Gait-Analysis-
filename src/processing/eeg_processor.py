"""
============================================================
EEG Signal Processor
AI-Powered Parkinson's Disease Severity Assessment
============================================================
Performs:
  - Bandpass filtering (1–40 Hz)
  - Notch filtering (50/60 Hz power line noise)
  - Artifact removal (amplitude thresholding)
  - ICA (Independent Component Analysis) artifact rejection
  - Channel selection
  - Normalization (z-score per channel)

Extracts:
  - Delta (1–4 Hz) band power
  - Theta (4–8 Hz) band power
  - Alpha (8–13 Hz) band power
  - Beta  (13–30 Hz) band power
  - Gamma (30–40 Hz) band power
  - Spectral Entropy
  - Coherence (inter-channel)
  - Functional Connectivity (PLV)
============================================================
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal as sp_signal
from scipy.signal import butter, filtfilt, welch

warnings.filterwarnings("ignore")

try:
    import mne
    from mne.preprocessing import ICA
    from mne_connectivity import spectral_connectivity_epochs
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("[WARN] MNE not installed — EEG processing will be limited.")

try:
    import antropy
    ANTROPY_AVAILABLE = True
except ImportError:
    ANTROPY_AVAILABLE = False


# ─────────────────────────────────────────────
# Frequency Band Definitions
# ─────────────────────────────────────────────
BANDS = {
    "delta": (1.0,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0,  13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

DEFAULT_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]


# ─────────────────────────────────────────────
# Low-Level DSP Functions
# ─────────────────────────────────────────────

def bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float,
    fs: float, order: int = 4
) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter.

    Args:
        data:    (n_channels, n_samples) or (n_samples,)
        lowcut:  Low frequency cutoff (Hz)
        highcut: High frequency cutoff (Hz)
        fs:      Sampling rate (Hz)
        order:   Filter order

    Returns:
        Filtered array of same shape as input
    """
    nyq = 0.5 * fs
    low  = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.9999)
    b, a = butter(order, [low, high], btype="band")
    if data.ndim == 1:
        return filtfilt(b, a, data)
    return np.array([filtfilt(b, a, ch) for ch in data])


def notch_filter(
    data: np.ndarray, fs: float,
    freqs: List[float] = [50.0, 60.0], Q: float = 30.0
) -> np.ndarray:
    """
    Apply notch filter(s) to remove power line noise.
    """
    filtered = data.copy()
    for freq in freqs:
        b, a = sp_signal.iirnotch(freq, Q, fs)
        if filtered.ndim == 1:
            filtered = filtfilt(b, a, filtered)
        else:
            filtered = np.array([filtfilt(b, a, ch) for ch in filtered])
    return filtered


def remove_artifacts_amplitude(
    data: np.ndarray, threshold_uv: float = 150.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove time segments where any channel exceeds amplitude threshold.

    Args:
        data:         (n_channels, n_samples)
        threshold_uv: Amplitude threshold in µV

    Returns:
        (cleaned_data, bad_sample_mask)
    """
    max_amp    = np.max(np.abs(data), axis=0)
    bad_mask   = max_amp > threshold_uv
    good_mask  = ~bad_mask
    return data[:, good_mask], bad_mask


def normalize_channels(data: np.ndarray) -> np.ndarray:
    """
    Z-score normalize each EEG channel independently.

    Args:
        data: (n_channels, n_samples)

    Returns:
        Normalized array of same shape
    """
    mean = np.mean(data, axis=1, keepdims=True)
    std  = np.std(data,  axis=1, keepdims=True) + 1e-8
    return (data - mean) / std


# ─────────────────────────────────────────────
# Band Power Extraction
# ─────────────────────────────────────────────

_trapz = getattr(np, "trapezoid", np.trapz)

def compute_band_power(
    data: np.ndarray, fs: float,
    band: Tuple[float, float],
    window: str = "hann",
    relative: bool = True,
) -> np.ndarray:
    """
    Compute power spectral density for a frequency band using Welch method.

    Args:
        data:     (n_channels, n_samples) or (n_samples,)
        fs:       Sampling rate
        band:     (low_freq, high_freq)
        window:   Welch window type
        relative: If True, return relative power (band / total)

    Returns:
        Power per channel — shape (n_channels,) or scalar
    """
    nperseg = min(int(fs * 2), data.shape[-1])
    if data.ndim == 1:
        data = data[np.newaxis, :]

    powers = []
    for ch in data:
        freqs, psd = welch(ch, fs=fs, window=window, nperseg=nperseg)
        freq_mask  = (freqs >= band[0]) & (freqs <= band[1])
        band_pow   = _trapz(psd[freq_mask], freqs[freq_mask])
        if relative:
            total_pow = _trapz(psd, freqs) + 1e-10
            band_pow /= total_pow
        powers.append(band_pow)

    return np.array(powers)


def compute_all_band_powers(
    data: np.ndarray, fs: float, relative: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute power for all 5 frequency bands.

    Returns:
        Dict mapping band name → array of shape (n_channels,)
    """
    return {
        band_name: compute_band_power(data, fs, band_range, relative=relative)
        for band_name, band_range in BANDS.items()
    }


# ─────────────────────────────────────────────
# Spectral Entropy
# ─────────────────────────────────────────────

def compute_spectral_entropy(data: np.ndarray, fs: float) -> np.ndarray:
    """
    Compute spectral entropy for each channel.
    Spectral entropy = -Σ p_i * log2(p_i), normalized by log2(N).

    Args:
        data: (n_channels, n_samples)

    Returns:
        Spectral entropy per channel — shape (n_channels,)
    """
    if data.ndim == 1:
        data = data[np.newaxis, :]

    entropies = []
    for ch in data:
        if ANTROPY_AVAILABLE:
            entropies.append(antropy.spectral_entropy(ch, sf=fs, normalize=True))
        else:
            # Manual implementation
            freqs, psd = welch(ch, fs=fs)
            psd_norm   = psd / (np.sum(psd) + 1e-10)
            psd_norm   = psd_norm[psd_norm > 0]
            se         = -np.sum(psd_norm * np.log2(psd_norm))
            se        /= np.log2(len(psd_norm))
            entropies.append(se)

    return np.array(entropies)


# ─────────────────────────────────────────────
# Coherence & Functional Connectivity
# ─────────────────────────────────────────────

def compute_coherence(
    data: np.ndarray, fs: float,
    ch1_idx: int = 0, ch2_idx: int = 1
) -> Dict[str, float]:
    """
    Compute magnitude-squared coherence between two channels.
    Returns mean coherence per frequency band.

    Args:
        data:    (n_channels, n_samples)
        fs:      Sampling rate
        ch1_idx: Index of first channel
        ch2_idx: Index of second channel

    Returns:
        Dict with mean coherence per band
    """
    nperseg  = min(int(fs * 2), data.shape[-1])
    freqs, cxy = sp_signal.coherence(
        data[ch1_idx], data[ch2_idx], fs=fs, nperseg=nperseg
    )

    result = {}
    for band_name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs <= high)
        result[f"coherence_{band_name}"] = float(np.mean(cxy[mask]))
    return result


def compute_plv_matrix(data: np.ndarray, fs: float, band: Tuple[float, float]) -> np.ndarray:
    """
    Compute Phase Locking Value (PLV) connectivity matrix.

    Args:
        data: (n_channels, n_samples)
        band: (low_freq, high_freq)

    Returns:
        PLV matrix of shape (n_channels, n_channels)
    """
    # Filter to band
    filtered = bandpass_filter(data, band[0], band[1], fs)
    n_ch     = filtered.shape[0]

    # Compute analytic signal (Hilbert transform)
    from scipy.signal import hilbert
    phases = np.angle(hilbert(filtered, axis=1))

    plv_mat = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            phase_diff = phases[i] - phases[j]
            plv        = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_mat[i, j] = plv
            plv_mat[j, i] = plv

    return plv_mat


def compute_functional_connectivity_features(
    data: np.ndarray, fs: float
) -> Dict[str, float]:
    """
    Summarize PLV-based functional connectivity across all bands.
    Returns mean and std PLV per band.
    """
    features = {}
    for band_name, band_range in BANDS.items():
        plv_mat  = compute_plv_matrix(data, fs, band_range)
        # Upper triangle (excluding diagonal)
        upper    = plv_mat[np.triu_indices_from(plv_mat, k=1)]
        features[f"plv_{band_name}_mean"] = float(np.mean(upper))
        features[f"plv_{band_name}_std"]  = float(np.std(upper))
        features[f"plv_{band_name}_max"]  = float(np.max(upper))
    return features


# ─────────────────────────────────────────────
# ICA Artifact Rejection (MNE)
# ─────────────────────────────────────────────

def apply_ica(
    raw: "mne.io.RawArray",
    n_components: int = 15,
    max_iter: int = 800,
    random_state: int = 42,
) -> "mne.io.RawArray":
    """
    Apply ICA to reject artifact components (eye blinks, muscle).
    Uses automatic EOG correlation for component selection.

    Args:
        raw:          MNE Raw object
        n_components: Number of ICA components
        max_iter:     Maximum ICA iterations

    Returns:
        ICA-cleaned MNE Raw object
    """
    if not MNE_AVAILABLE:
        return raw

    ica = ICA(
        n_components=n_components,
        max_iter=max_iter,
        random_state=random_state,
        method="fastica",
    )

    # Fit ICA
    ica.fit(raw, picks="eeg")

    # Automatically find EOG artifacts
    try:
        eog_indices, _ = ica.find_bads_eog(raw, threshold=3.0)
        ica.exclude = eog_indices
    except Exception:
        pass  # No EOG channel available

    # Automatically find muscle artifacts
    try:
        muscle_indices, _ = ica.find_bads_muscle(raw)
        ica.exclude += muscle_indices
    except Exception:
        pass

    ica.apply(raw)
    return raw


# ─────────────────────────────────────────────
# Main EEG Processor Class
# ─────────────────────────────────────────────

class EEGProcessor:
    """
    End-to-end EEG processing pipeline for Parkinson's analysis.

    Usage:
        processor = EEGProcessor(fs=256)

        # From EDF file
        features = processor.process_edf("path/to/eeg.edf")

        # From numpy array
        data = np.random.randn(19, 10000)  # (channels, samples)
        features = processor.process_array(data)
    """

    def __init__(
        self,
        fs: float = 256.0,
        bandpass_low: float = 1.0,
        bandpass_high: float = 40.0,
        artifact_threshold_uv: float = 150.0,
        apply_ica_flag: bool = True,
        reference: str = "average",
        channel_names: Optional[List[str]] = None,
    ):
        self.fs                    = fs
        self.bandpass_low          = bandpass_low
        self.bandpass_high         = bandpass_high
        self.artifact_threshold_uv = artifact_threshold_uv
        self.apply_ica_flag        = apply_ica_flag
        self.reference             = reference
        self.channel_names         = channel_names or DEFAULT_CHANNELS

    def load_edf(self, file_path: Union[str, Path]) -> "mne.io.Raw":
        """
        Load an EDF/EDF+ file using MNE.
        """
        if not MNE_AVAILABLE:
            raise ImportError("MNE is required to load EDF files: pip install mne")

        raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose=False)
        return raw

    def preprocess_raw(self, raw: "mne.io.Raw") -> np.ndarray:
        """
        Preprocess MNE Raw object:
        - Pick EEG channels
        - Set montage
        - Apply reference
        - Bandpass filter
        - Notch filter (50 & 60 Hz)
        - ICA (optional)
        - Return numpy (n_channels, n_samples)
        """
        if not MNE_AVAILABLE:
            raise ImportError("MNE is required for preprocessing.")

        # Pick EEG channels
        raw.pick("eeg")

        # Set average reference
        if self.reference == "average":
            raw.set_eeg_reference("average", projection=False, verbose=False)

        # Bandpass filter
        raw.filter(self.bandpass_low, self.bandpass_high,
                   method="iir", iir_params={"order": 4, "ftype": "butter"},
                   verbose=False)

        # Notch filter
        raw.notch_filter([50, 60], verbose=False)

        # ICA
        if self.apply_ica_flag and raw.n_times > 1000:
            try:
                raw = apply_ica(raw)
            except Exception as e:
                print(f"[WARN] ICA failed: {e}")

        return raw.get_data() * 1e6  # Convert V → µV

    def preprocess_array(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess a raw numpy array (n_channels, n_samples).
        """
        # Bandpass
        data = bandpass_filter(data, self.bandpass_low, self.bandpass_high, self.fs)
        # Notch
        data = notch_filter(data, self.fs)
        # Artifact removal
        data, _ = remove_artifacts_amplitude(data, self.artifact_threshold_uv)
        # Normalize
        data = normalize_channels(data)
        return data

    def extract_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract all EEG features from preprocessed array (n_channels, n_samples).

        Returns:
            Flat feature dictionary with ~120 entries
        """
        features = {}
        n_ch = data.shape[0]

        # ── 1. Band Powers (per channel) ──────────────────
        band_powers = compute_all_band_powers(data, self.fs, relative=True)
        for band_name, powers in band_powers.items():
            for ch_i, pwr in enumerate(powers):
                features[f"{band_name}_ch{ch_i:02d}"] = float(pwr)
            # Summary stats across channels
            features[f"{band_name}_mean"] = float(np.mean(powers))
            features[f"{band_name}_std"]  = float(np.std(powers))
            features[f"{band_name}_max"]  = float(np.max(powers))

        # ── 2. Band Power Ratios ───────────────────────────
        delta = band_powers["delta"] + 1e-10
        theta = band_powers["theta"] + 1e-10
        alpha = band_powers["alpha"] + 1e-10
        beta  = band_powers["beta"]  + 1e-10

        features["theta_alpha_ratio"] = float(np.mean(theta / alpha))
        features["delta_beta_ratio"]  = float(np.mean(delta / beta))
        features["theta_beta_ratio"]  = float(np.mean(theta / beta))
        features["alpha_beta_ratio"]  = float(np.mean(alpha / beta))

        # ── 3. Spectral Entropy ────────────────────────────
        entropies = compute_spectral_entropy(data, self.fs)
        features["spectral_entropy_mean"] = float(np.mean(entropies))
        features["spectral_entropy_std"]  = float(np.std(entropies))
        for ch_i, se in enumerate(entropies):
            features[f"spectral_entropy_ch{ch_i:02d}"] = float(se)

        # ── 4. Coherence (first pair of channels) ─────────
        if n_ch >= 2:
            coh_feats = compute_coherence(data, self.fs, ch1_idx=0, ch2_idx=1)
            features.update(coh_feats)

        # ── 5. Functional Connectivity (PLV) ──────────────
        plv_feats = compute_functional_connectivity_features(data, self.fs)
        features.update(plv_feats)

        # ── 6. Statistical Features per Channel ───────────
        features["eeg_mean"]        = float(np.mean(data))
        features["eeg_std"]         = float(np.std(data))
        features["eeg_kurtosis"]    = float(self._kurtosis(data))
        features["eeg_skewness"]    = float(self._skewness(data))
        features["eeg_peak_to_peak"]= float(np.max(data) - np.min(data))

        return features

    @staticmethod
    def _kurtosis(data: np.ndarray) -> float:
        from scipy.stats import kurtosis
        return float(kurtosis(data.flatten()))

    @staticmethod
    def _skewness(data: np.ndarray) -> float:
        from scipy.stats import skew
        return float(skew(data.flatten()))

    def process_edf(self, file_path: Union[str, Path]) -> Dict[str, float]:
        """
        Full pipeline: load EDF → preprocess → extract features.
        """
        raw  = self.load_edf(file_path)
        data = self.preprocess_raw(raw)
        return self.extract_features(data)

    def process_array(self, data: np.ndarray) -> Dict[str, float]:
        """
        Full pipeline: preprocess numpy array → extract features.
        """
        data = self.preprocess_array(data)
        return self.extract_features(data)

    def get_feature_vector(self, data_or_path: Union[np.ndarray, str, Path]) -> np.ndarray:
        """
        Returns 1-D feature vector suitable for model input.
        """
        if isinstance(data_or_path, np.ndarray):
            feats = self.process_array(data_or_path)
        else:
            feats = self.process_edf(data_or_path)

        return np.array(list(feats.values()), dtype=np.float32)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    proc = EEGProcessor(fs=256.0)

    if len(sys.argv) >= 2:
        fp = sys.argv[1]
        feats = proc.process_edf(fp)
    else:
        # Demo with synthetic data
        print("[INFO] No EDF file provided — using synthetic EEG data")
        synthetic = np.random.randn(19, 256 * 30)  # 19 channels, 30 seconds
        feats = proc.process_array(synthetic)

    print(f"\n✅ Extracted {len(feats)} EEG features")
    for k, v in list(feats.items())[:10]:
        print(f"  {k:45s}: {v:.6f}")

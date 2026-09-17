"""
============================================================
Feature Engineering Pipeline
AI-Powered Parkinson's Disease Severity Assessment
============================================================
Handles:
  - Missing Value Imputation (KNN + Median fallback)
  - Outlier Detection (IQR + Isolation Forest)
  - Feature Scaling (StandardScaler / RobustScaler)
  - PCA dimensionality reduction
  - Correlation Analysis & highly-correlated feature removal
  - Feature Fusion (concatenate voice + EEG + gait vectors)
============================================================
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
RANDOM_STATE   = 42
IQR_FACTOR     = 1.5       # Threshold multiplier for IQR outlier detection
ISO_FOREST_CONTAMINATION = 0.05  # Expected fraction of outliers


# ─────────────────────────────────────────────
# Missing Value Handling
# ─────────────────────────────────────────────

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "knn",
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """
    Impute missing values.

    Args:
        df:           Input DataFrame (numeric columns only)
        strategy:     'knn' | 'median' | 'mean' | 'zero'
        n_neighbors:  Number of neighbors for KNN imputation

    Returns:
        Imputed DataFrame
    """
    numeric_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric   = df.drop(columns=numeric_cols)

    n_missing = df[numeric_cols].isnull().sum().sum()
    if n_missing == 0:
        return df

    print(f"[INFO] Imputing {n_missing} missing values using strategy='{strategy}'")

    if strategy == "knn":
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    elif strategy == "median":
        imputer = SimpleImputer(strategy="median")
    elif strategy == "mean":
        imputer = SimpleImputer(strategy="mean")
    elif strategy == "zero":
        imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")

    imputed = imputer.fit_transform(df[numeric_cols])
    df_out  = pd.DataFrame(imputed, columns=numeric_cols, index=df.index)
    df_out  = pd.concat([df_out, non_numeric.reset_index(drop=True)], axis=1)
    return df_out


# ─────────────────────────────────────────────
# Outlier Detection & Removal
# ─────────────────────────────────────────────

def detect_outliers_iqr(
    df: pd.DataFrame,
    factor: float = IQR_FACTOR,
) -> pd.Series:
    """
    Detect outlier rows using IQR method.
    A sample is an outlier if ANY feature falls outside [Q1-k*IQR, Q3+k*IQR].

    Returns:
        Boolean Series (True = outlier)
    """
    numeric = df.select_dtypes(include=[np.number])
    Q1      = numeric.quantile(0.25)
    Q3      = numeric.quantile(0.75)
    IQR     = Q3 - Q1

    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR

    outlier_mask = ((numeric < lower) | (numeric > upper)).any(axis=1)
    n_out = outlier_mask.sum()
    print(f"[INFO] IQR detected {n_out} outlier rows ({n_out/len(df)*100:.1f}%)")
    return outlier_mask


def detect_outliers_isolation_forest(
    df: pd.DataFrame,
    contamination: float = ISO_FOREST_CONTAMINATION,
) -> pd.Series:
    """
    Detect outliers using Isolation Forest.

    Returns:
        Boolean Series (True = outlier)
    """
    from sklearn.ensemble import IsolationForest

    numeric = df.select_dtypes(include=[np.number]).fillna(0)
    iso     = IsolationForest(
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_estimators=100,
    )
    preds       = iso.fit_predict(numeric)
    outlier_mask= pd.Series(preds == -1, index=df.index)
    n_out = outlier_mask.sum()
    print(f"[INFO] IsolationForest detected {n_out} outliers ({n_out/len(df)*100:.1f}%)")
    return outlier_mask


def remove_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    cap_instead_of_remove: bool = True,
) -> pd.DataFrame:
    """
    Remove or cap outliers.

    Args:
        df:                   Input DataFrame
        method:               'iqr' | 'isolation_forest' | 'both'
        cap_instead_of_remove: If True, cap values at bounds instead of removing rows

    Returns:
        Cleaned DataFrame
    """
    if method == "iqr":
        mask = detect_outliers_iqr(df)
    elif method == "isolation_forest":
        mask = detect_outliers_isolation_forest(df)
    elif method == "both":
        m1   = detect_outliers_iqr(df)
        m2   = detect_outliers_isolation_forest(df)
        mask = m1 & m2
    else:
        raise ValueError(f"Unknown outlier method: {method}")

    if cap_instead_of_remove:
        numeric = df.select_dtypes(include=[np.number])
        Q1      = numeric.quantile(0.25)
        Q3      = numeric.quantile(0.75)
        IQR     = Q3 - Q1
        lower   = Q1 - IQR_FACTOR * IQR
        upper   = Q3 + IQR_FACTOR * IQR
        df_capped = df.copy()
        for col in numeric.columns:
            df_capped[col] = df_capped[col].clip(lower=lower[col], upper=upper[col])
        return df_capped
    else:
        return df[~mask].reset_index(drop=True)


# ─────────────────────────────────────────────
# Feature Scaling
# ─────────────────────────────────────────────

class FeatureScaler:
    """
    Wraps StandardScaler / RobustScaler with fit/transform/save/load.
    """

    def __init__(self, method: str = "robust"):
        """
        Args:
            method: 'standard' | 'robust'
        """
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        self.method = method
        self.feature_names: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        numeric = df.select_dtypes(include=[np.number])
        self.feature_names = numeric.columns.tolist()
        self.scaler.fit(numeric)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        df_out[self.feature_names] = self.scaler.transform(df[self.feature_names])
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def save(self, path: Union[str, Path]) -> None:
        joblib.dump(self.scaler, str(path))

    @classmethod
    def load(cls, path: Union[str, Path], method: str = "robust") -> "FeatureScaler":
        obj = cls(method)
        obj.scaler = joblib.load(str(path))
        return obj


# ─────────────────────────────────────────────
# Correlation Analysis
# ─────────────────────────────────────────────

def compute_correlation_matrix(
    df: pd.DataFrame, method: str = "pearson"
) -> pd.DataFrame:
    """Compute feature correlation matrix."""
    return df.select_dtypes(include=[np.number]).corr(method=method)


def remove_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
    method: str = "pearson",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features with pairwise correlation > threshold.
    Keeps the first feature in each correlated pair.

    Returns:
        (filtered_df, list_of_removed_columns)
    """
    corr_matrix = compute_correlation_matrix(df, method)
    upper       = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [col for col in upper.columns if any(upper[col].abs() > threshold)]
    print(f"[INFO] Removing {len(to_drop)} highly correlated features (threshold={threshold})")
    return df.drop(columns=to_drop), to_drop


# ─────────────────────────────────────────────
# PCA
# ─────────────────────────────────────────────

class PCAReducer:
    """
    PCA dimensionality reduction with variance explained tracking.
    """

    def __init__(
        self,
        n_components: Optional[Union[int, float]] = 0.95,
        random_state: int = RANDOM_STATE,
    ):
        """
        Args:
            n_components: int (exact components) or float (variance explained ratio)
        """
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> "PCAReducer":
        self.pca.fit(X)
        print(f"[INFO] PCA: {self.pca.n_components_} components explain "
              f"{self.pca.explained_variance_ratio_.sum()*100:.1f}% variance")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def get_loadings(self, feature_names: List[str]) -> pd.DataFrame:
        """Return PCA loadings as DataFrame."""
        return pd.DataFrame(
            self.pca.components_.T,
            index=feature_names,
            columns=[f"PC{i+1}" for i in range(self.pca.n_components_)],
        )

    def save(self, path: Union[str, Path]) -> None:
        joblib.dump(self.pca, str(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PCAReducer":
        obj = cls()
        obj.pca = joblib.load(str(path))
        return obj


# ─────────────────────────────────────────────
# Multimodal Feature Fusion
# ─────────────────────────────────────────────

def fuse_feature_vectors(
    voice_vec: Optional[np.ndarray] = None,
    eeg_vec:   Optional[np.ndarray] = None,
    gait_vec:  Optional[np.ndarray] = None,
    fill_missing: bool = True,
    expected_voice_dim: int = 200,
    expected_eeg_dim:   int = 150,
    expected_gait_dim:  int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate multimodal feature vectors.
    Handles missing modalities by zero-padding.

    Args:
        voice_vec: 1-D voice feature vector (or None if missing)
        eeg_vec:   1-D EEG feature vector (or None if missing)
        gait_vec:  1-D gait feature vector (or None if missing)
        fill_missing: If True, replace missing with zero vectors

    Returns:
        (fused_vector, modality_mask)
        modality_mask: [voice_present, eeg_present, gait_present]
    """
    parts = []
    mask  = []

    for vec, name, expected_dim in [
        (voice_vec, "voice", expected_voice_dim),
        (eeg_vec,   "eeg",   expected_eeg_dim),
        (gait_vec,  "gait",  expected_gait_dim),
    ]:
        if vec is not None:
            parts.append(vec.astype(np.float32).flatten())
            mask.append(1.0)
        elif fill_missing:
            parts.append(np.zeros(expected_dim, dtype=np.float32))
            mask.append(0.0)
            print(f"[WARN] {name} modality missing — filled with zeros")
        else:
            mask.append(0.0)

    if not parts:
        raise ValueError("At least one modality must be provided")

    return np.concatenate(parts), np.array(mask, dtype=np.float32)


# ─────────────────────────────────────────────
# Full Pipeline Class
# ─────────────────────────────────────────────

class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline combining all steps.

    Usage:
        pipeline = FeatureEngineeringPipeline()
        df_ready = pipeline.fit_transform(df_raw)
        df_new   = pipeline.transform(df_new_data)
    """

    def __init__(
        self,
        impute_strategy:   str   = "knn",
        scaling_method:    str   = "robust",
        outlier_method:    str   = "iqr",
        remove_outliers_:  bool  = True,
        corr_threshold:    float = 0.95,
        apply_pca:         bool  = False,
        pca_variance:      float = 0.95,
        label_cols:        Optional[List[str]] = None,
    ):
        self.impute_strategy  = impute_strategy
        self.scaling_method   = scaling_method
        self.outlier_method   = outlier_method
        self.remove_outliers_ = remove_outliers_
        self.corr_threshold   = corr_threshold
        self.apply_pca        = apply_pca
        self.pca_variance     = pca_variance
        self.label_cols       = label_cols or ["updrs_total", "severity_class"]

        self.scaler         = FeatureScaler(scaling_method)
        self.pca_reducer    = PCAReducer(pca_variance) if apply_pca else None
        self.dropped_cols   = []
        self.feature_cols   = []
        self.is_fitted      = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit pipeline and transform training data."""
        # Separate labels
        label_df = df[[c for c in self.label_cols if c in df.columns]]
        feat_df  = df.drop(columns=[c for c in self.label_cols if c in df.columns])

        # 1. Impute
        feat_df = handle_missing_values(feat_df, self.impute_strategy)

        # 2. Outlier removal
        if self.remove_outliers_:
            feat_df = remove_outliers(feat_df, self.outlier_method, cap_instead_of_remove=True)

        # 3. Remove correlated
        feat_df, self.dropped_cols = remove_highly_correlated_features(
            feat_df, self.corr_threshold
        )

        # 4. Scale
        feat_df = self.scaler.fit_transform(feat_df)

        # 5. PCA
        if self.apply_pca and self.pca_reducer:
            numeric   = feat_df.select_dtypes(include=[np.number])
            pca_arr   = self.pca_reducer.fit_transform(numeric.values)
            feat_df   = pd.DataFrame(
                pca_arr,
                columns=[f"PC{i+1}" for i in range(pca_arr.shape[1])],
                index=feat_df.index,
            )

        self.feature_cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
        self.is_fitted    = True

        return pd.concat([feat_df.reset_index(drop=True), label_df.reset_index(drop=True)], axis=1)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fit before transform. Call fit_transform first.")

        label_df = df[[c for c in self.label_cols if c in df.columns]]
        feat_df  = df.drop(columns=[c for c in self.label_cols if c in df.columns])

        feat_df = handle_missing_values(feat_df, self.impute_strategy)
        feat_df = feat_df.drop(columns=[c for c in self.dropped_cols if c in feat_df.columns])
        feat_df = self.scaler.transform(feat_df)

        if self.apply_pca and self.pca_reducer:
            numeric = feat_df.select_dtypes(include=[np.number])
            pca_arr = self.pca_reducer.transform(numeric.values)
            feat_df = pd.DataFrame(
                pca_arr,
                columns=[f"PC{i+1}" for i in range(pca_arr.shape[1])],
                index=feat_df.index,
            )

        return pd.concat([feat_df.reset_index(drop=True), label_df.reset_index(drop=True)], axis=1)

    def save(self, save_dir: Union[str, Path]) -> None:
        """Save scaler and PCA reducer to disk."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.scaler.save(save_dir / "scaler.pkl")
        if self.pca_reducer:
            self.pca_reducer.save(save_dir / "pca.pkl")
        import json
        meta = {
            "dropped_cols":   self.dropped_cols,
            "feature_cols":   self.feature_cols,
            "impute_strategy": self.impute_strategy,
            "scaling_method":  self.scaling_method,
            "corr_threshold":  self.corr_threshold,
        }
        with open(save_dir / "pipeline_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[INFO] Pipeline saved to {save_dir}")

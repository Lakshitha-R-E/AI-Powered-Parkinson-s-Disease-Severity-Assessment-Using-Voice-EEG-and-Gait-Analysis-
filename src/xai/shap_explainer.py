"""
============================================================
SHAP Explainer
AI-Powered Parkinson's Disease Severity Assessment
============================================================
Generates SHAP explanations for:
  - Voice feature importance
  - EEG feature importance
  - Gait feature importance
  - Combined feature importance

Produces:
  - Beeswarm plots
  - Waterfall plots
  - Bar plots
  - Force plots
============================================================
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] SHAP not installed: pip install shap")


# ─────────────────────────────────────────────
# XGBoost SHAP Explainer (Feature-level)
# ─────────────────────────────────────────────

class VoiceSHAPExplainer:
    """
    SHAP explainer for voice features using XGBoost or linear model.
    Works on the extracted feature vector level (not embeddings).
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names
        self.explainer     = None
        self.model         = None
        self.shap_values   = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, task: str = "classification"):
        """
        Train a surrogate XGBoost model and create SHAP explainer.

        Args:
            X_train: Feature matrix (N, n_features)
            y_train: Labels (N,)
            task:    'classification' | 'regression'
        """
        if not SHAP_AVAILABLE:
            raise ImportError("Install SHAP: pip install shap")

        from xgboost import XGBClassifier, XGBRegressor

        if task == "classification":
            self.model = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=42,
            )
        else:
            self.model = XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42,
            )

        self.model.fit(X_train, y_train)
        self.explainer = shap.TreeExplainer(self.model)
        print("[SHAP] Explainer fitted successfully")

    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for samples X.

        Returns:
            shap_values: array of shape (N, n_features) or list of (N, n_features) per class
        """
        if self.explainer is None:
            raise RuntimeError("Call fit() first")
        self.shap_values = self.explainer.shap_values(X)
        return self.shap_values

    def plot_beeswarm(
        self,
        X: np.ndarray,
        max_display: int = 20,
        class_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Generate SHAP beeswarm (summary) plot."""
        shap_vals = self.explain(X)

        if isinstance(shap_vals, list):
            sv = shap_vals[class_idx]
        else:
            sv = shap_vals

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            sv, X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
            plot_type="dot",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_waterfall(
        self,
        X: np.ndarray,
        sample_idx: int = 0,
        class_idx:  int = 0,
        save_path:  Optional[str] = None,
    ) -> plt.Figure:
        """Generate SHAP waterfall plot for a single sample."""
        if self.explainer is None:
            raise RuntimeError("Call fit() first")

        expl = self.explainer(X)

        if isinstance(expl.values, list):
            vals      = expl.values[class_idx]
            base_vals = expl.base_values[:, class_idx]
        else:
            vals      = expl.values
            base_vals = expl.base_values

        shap_exp = shap.Explanation(
            values=vals[sample_idx],
            base_values=base_vals[sample_idx],
            data=X[sample_idx],
            feature_names=self.feature_names,
        )

        fig = plt.figure(figsize=(10, 8))
        shap.plots.waterfall(shap_exp, show=False, max_display=15)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_bar(
        self,
        X: np.ndarray,
        max_display: int = 15,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Generate SHAP global feature importance bar plot."""
        shap_vals = self.explain(X)

        if isinstance(shap_vals, list):
            # Average absolute SHAP across classes
            sv = np.mean([np.abs(s) for s in shap_vals], axis=0)
        else:
            sv = np.abs(shap_vals)

        mean_shap = np.mean(sv, axis=0)
        indices   = np.argsort(mean_shap)[::-1][:max_display]

        names  = [self.feature_names[i] if self.feature_names else f"f{i}" for i in indices]
        values = mean_shap[indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors  = plt.cm.RdBu(np.linspace(0.2, 0.8, len(names)))
        bars    = ax.barh(names[::-1], values[::-1], color=colors)
        ax.set_xlabel("Mean |SHAP value|", fontsize=12)
        ax.set_title("Feature Importance (SHAP)", fontsize=14, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def get_top_features(self, X: np.ndarray, n: int = 10) -> List[Tuple[str, float]]:
        """Return top N features by mean absolute SHAP value."""
        shap_vals = self.explain(X)

        if isinstance(shap_vals, list):
            sv = np.mean([np.abs(s) for s in shap_vals], axis=0)
        else:
            sv = np.abs(shap_vals)

        mean_shap = np.mean(sv, axis=0)
        indices   = np.argsort(mean_shap)[::-1][:n]

        return [
            (self.feature_names[i] if self.feature_names else f"feature_{i}", float(mean_shap[i]))
            for i in indices
        ]


# ─────────────────────────────────────────────
# Neural Network SHAP (Deep SHAP / Gradient)
# ─────────────────────────────────────────────

class DeepModelSHAPExplainer:
    """
    SHAP explainer for PyTorch neural network models.
    Uses GradientExplainer for differentiable attribution.
    """

    def __init__(self, model, background_data: "torch.Tensor"):
        """
        Args:
            model:           PyTorch model (in eval mode)
            background_data: Background reference tensor (B, input_dim)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("Install SHAP: pip install shap")

        import torch
        model.eval()
        self.model      = model
        self.explainer  = shap.GradientExplainer(model, background_data)
        self.shap_values = None

    def explain(self, X: "torch.Tensor", nsamples: int = 50) -> np.ndarray:
        """
        Compute SHAP values for input tensor X.

        Returns:
            shap_values: (N, input_dim) or list per output
        """
        self.shap_values = self.explainer.shap_values(X, nsamples=nsamples)
        return self.shap_values


# ─────────────────────────────────────────────
# Multimodal SHAP Dashboard
# ─────────────────────────────────────────────

class MultimodalSHAPDashboard:
    """
    Generates a combined SHAP dashboard for all three modalities.
    """

    def __init__(
        self,
        voice_feature_names: Optional[List[str]] = None,
        eeg_feature_names:   Optional[List[str]] = None,
        gait_feature_names:  Optional[List[str]] = None,
    ):
        self.voice_explainer = VoiceSHAPExplainer(voice_feature_names)
        self.eeg_explainer   = VoiceSHAPExplainer(eeg_feature_names)
        self.gait_explainer  = VoiceSHAPExplainer(gait_feature_names)

    def fit_all(
        self,
        voice_X: np.ndarray, y_cls: np.ndarray,
        eeg_X:   Optional[np.ndarray] = None,
        gait_X:  Optional[np.ndarray] = None,
    ):
        """Fit SHAP explainers for all available modalities."""
        self.voice_explainer.fit(voice_X, y_cls, task="classification")
        if eeg_X is not None:
            self.eeg_explainer.fit(eeg_X, y_cls, task="classification")
        if gait_X is not None:
            self.gait_explainer.fit(gait_X, y_cls, task="classification")

    def generate_dashboard(
        self,
        voice_X: np.ndarray,
        eeg_X:   Optional[np.ndarray] = None,
        gait_X:  Optional[np.ndarray] = None,
        save_dir: Optional[str] = None,
    ) -> Dict[str, plt.Figure]:
        """
        Generate full SHAP dashboard with importance plots for each modality.

        Returns:
            Dict mapping plot name → matplotlib Figure
        """
        figures = {}
        save_dir = Path(save_dir) if save_dir else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        # Voice
        fig = self.voice_explainer.plot_bar(
            voice_X, max_display=15,
            save_path=str(save_dir / "shap_voice.png") if save_dir else None,
        )
        figures["voice_importance"] = fig

        # EEG
        if eeg_X is not None:
            fig = self.eeg_explainer.plot_bar(
                eeg_X, max_display=15,
                save_path=str(save_dir / "shap_eeg.png") if save_dir else None,
            )
            figures["eeg_importance"] = fig

        # Gait
        if gait_X is not None:
            fig = self.gait_explainer.plot_bar(
                gait_X, max_display=15,
                save_path=str(save_dir / "shap_gait.png") if save_dir else None,
            )
            figures["gait_importance"] = fig

        # Combined summary figure
        fig = self._plot_combined_summary(voice_X, eeg_X, gait_X, save_dir)
        figures["combined"] = fig

        return figures

    def _plot_combined_summary(
        self,
        voice_X: np.ndarray,
        eeg_X:   Optional[np.ndarray],
        gait_X:  Optional[np.ndarray],
        save_dir: Optional[Path],
    ) -> plt.Figure:
        """Generate a 3-panel figure showing top features per modality."""
        n_panels = 1 + (eeg_X is not None) + (gait_X is not None)
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
        if n_panels == 1:
            axes = [axes]

        panel_idx = 0

        def _plot_panel(explainer, X, title, ax):
            top = explainer.get_top_features(X, n=10)
            names  = [t[0][:20] for t in top]  # truncate long names
            values = [t[1] for t in top]
            colors = plt.cm.RdBu(np.linspace(0.2, 0.8, len(names)))
            ax.barh(names[::-1], values[::-1], color=colors)
            ax.set_title(title, fontweight="bold", fontsize=11)
            ax.set_xlabel("Mean |SHAP|")
            ax.spines[["top", "right"]].set_visible(False)

        _plot_panel(self.voice_explainer, voice_X, "Voice Features", axes[panel_idx])
        panel_idx += 1

        if eeg_X is not None:
            _plot_panel(self.eeg_explainer, eeg_X, "EEG Features", axes[panel_idx])
            panel_idx += 1

        if gait_X is not None:
            _plot_panel(self.gait_explainer, gait_X, "Gait Features", axes[panel_idx])

        fig.suptitle("SHAP Feature Importance — Multimodal Parkinson's Analysis",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_dir:
            fig.savefig(str(save_dir / "shap_combined.png"), dpi=150, bbox_inches="tight")
        return fig

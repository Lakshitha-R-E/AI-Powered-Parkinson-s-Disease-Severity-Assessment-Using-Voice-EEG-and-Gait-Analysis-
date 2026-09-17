"""
============================================================
Feature Selection Module
AI-Powered Parkinson's Disease Severity Assessment
============================================================
Provides methods for feature selection:
  - Variance Thresholding
  - Mutual Information (Regression & Classification)
  - Random Forest / XGBoost Feature Importance
  - Recursive Feature Elimination (RFE)
  - Lasso / L1 SelectFromModel
============================================================
"""

import warnings
from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    mutual_info_classif,
    mutual_info_regression,
    RFE,
    SelectFromModel,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV

warnings.filterwarnings("ignore")

class FeatureSelector:
    """
    Automated Feature Selection for Multimodal Parkinson's Signals.
    """

    def __init__(self, task: str = "classification"):
        """
        Args:
            task: 'classification' or 'regression'
        """
        self.task = task
        self.selected_features: List[str] = []

    def remove_low_variance(self, df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Remove features with variance lower than threshold."""
        selector = VarianceThreshold(threshold=threshold)
        numeric = df.select_dtypes(include=[np.number])
        selector.fit(numeric)
        
        kept_cols = numeric.columns[selector.get_support()].tolist()
        print(f"[FeatureSelector] Low Variance Filter: Retained {len(kept_cols)}/{len(numeric.columns)} features.")
        return df[kept_cols]

    def select_mutual_info(self, X: pd.DataFrame, y: np.ndarray, k: int = 50) -> pd.DataFrame:
        """Select top k features based on Mutual Information."""
        score_fn = mutual_info_classif if self.task == "classification" else mutual_info_regression
        selector = SelectKBest(score_func=score_fn, k=min(k, X.shape[1]))
        X_new = selector.fit_transform(X, y)
        selected = X.columns[selector.get_support()].tolist()
        print(f"[FeatureSelector] Mutual Information: Selected top {len(selected)} features.")
        return X[selected]

    def select_tree_importance(self, X: pd.DataFrame, y: np.ndarray, top_n: int = 50) -> pd.DataFrame:
        """Select features based on Tree-based importance (RandomForest)."""
        if self.task == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            
        model.fit(X, y)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        selected = X.columns[indices].tolist()
        print(f"[FeatureSelector] Tree Importance: Selected top {len(selected)} features.")
        return X[selected]

    def select_rfe(self, X: pd.DataFrame, y: np.ndarray, n_features_to_select: int = 30) -> pd.DataFrame:
        """Recursive Feature Elimination (RFE)."""
        if self.task == "classification":
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)

        rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=0.1)
        rfe.fit(X, y)
        selected = X.columns[rfe.get_support()].tolist()
        print(f"[FeatureSelector] RFE: Selected {len(selected)} features.")
        return X[selected]

    def select_lasso(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """L1/Lasso-based feature selection for regression."""
        if self.task != "regression":
            print("[WARN] Lasso feature selection is intended for regression tasks.")
            return X
            
        lasso = LassoCV(cv=5, random_state=42).fit(X, y)
        model = SelectFromModel(lasso, prefit=True)
        selected = X.columns[model.get_support()].tolist()
        print(f"[FeatureSelector] Lasso: Selected {len(selected)} features.")
        return X[selected]

"""
EcoTox-AI: Feature Engineering Module
Generates polynomial features, interaction terms, and performs
feature selection using mutual information and correlation analysis.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_regression, SelectKBest
import logging

from src.data_loader import FEATURE_NAMES

logger = logging.getLogger(__name__)


def add_polynomial_features(X: np.ndarray, degree: int = 2, 
                            include_bias: bool = False,
                            interaction_only: bool = False) -> tuple:
    """
    Generate polynomial and interaction features.
    
    For 6 input features with degree=2:
    - Original 6 features
    - 6 squared terms (x1², x2², ...)
    - 15 interaction terms (x1*x2, x1*x3, ...)
    - Total: 27 features (without bias)
    
    Args:
        X: Feature matrix (n_samples, n_features)
        degree: Polynomial degree (default 2)
        include_bias: Whether to include bias column
        interaction_only: If True, only interaction terms (no x²)
        
    Returns:
        (X_poly, poly_transformer, feature_names_poly)
    """
    poly = PolynomialFeatures(
        degree=degree, 
        include_bias=include_bias,
        interaction_only=interaction_only
    )
    X_poly = poly.fit_transform(X)
    
    # Generate meaningful feature names
    feature_names_poly = poly.get_feature_names_out(FEATURE_NAMES)
    
    logger.info(
        f"Polynomial features (degree={degree}): "
        f"{X.shape[1]} → {X_poly.shape[1]} features"
    )
    return X_poly, poly, feature_names_poly


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix for features and target.
    
    Args:
        df: DataFrame with feature and target columns
        
    Returns:
        Correlation matrix as DataFrame
    """
    corr_matrix = df.corr()
    logger.info("Correlation matrix computed.")
    return corr_matrix


def compute_mutual_information(X: np.ndarray, y: np.ndarray, 
                               n_neighbors: int = 5) -> np.ndarray:
    """
    Compute mutual information between each feature and the target.
    Higher MI = more informative feature.
    
    Args:
        X: Feature matrix
        y: Target vector
        n_neighbors: Number of neighbors for MI estimation
        
    Returns:
        Array of MI scores for each feature
    """
    mi_scores = mutual_info_regression(
        X, y, n_neighbors=n_neighbors, random_state=42
    )
    
    logger.info("Mutual Information scores:")
    for name, score in zip(FEATURE_NAMES, mi_scores):
        logger.info(f"  {name}: {score:.4f}")
    
    return mi_scores


def select_top_features(X: np.ndarray, y: np.ndarray, 
                        k: int = 'all') -> tuple:
    """
    Select top-k features based on mutual information.
    
    Args:
        X: Feature matrix
        y: Target vector
        k: Number of top features to select (or 'all')
        
    Returns:
        (X_selected, selector, selected_indices)
    """
    selector = SelectKBest(
        score_func=mutual_info_regression, 
        k=k
    )
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    
    selected_names = [FEATURE_NAMES[i] for i in selected_indices]
    logger.info(f"Selected {len(selected_names)} features: {selected_names}")
    
    return X_selected, selector, selected_indices


def add_custom_interaction_features(X: np.ndarray) -> tuple:
    """
    Add domain-specific interaction features known to be relevant
    in ecotoxicity prediction.
    
    Key interactions for aquatic toxicity:
    - CIC0 * MLOGP: Connectivity × lipophilicity (bioavailability)
    - GATS1i * MLOGP: Autocorrelation × lipophilicity
    - NdsCH * NdssC: Unsaturated carbon descriptor interactions
    
    Args:
        X: Feature matrix with columns in FEATURE_NAMES order
        
    Returns:
        (X_augmented, new_feature_names)
    """
    # Column indices: CIC0=0, SM1_Dz_Z=1, GATS1i=2, NdsCH=3, NdssC=4, MLOGP=5
    interactions = {
        'CIC0_x_MLOGP': X[:, 0] * X[:, 5],
        'GATS1i_x_MLOGP': X[:, 2] * X[:, 5],
        'NdsCH_x_NdssC': X[:, 3] * X[:, 4],
        'CIC0_x_GATS1i': X[:, 0] * X[:, 2],
        'MLOGP_squared': X[:, 5] ** 2,
        'CIC0_squared': X[:, 0] ** 2,
    }
    
    new_features = np.column_stack(list(interactions.values()))
    X_augmented = np.hstack([X, new_features])
    new_feature_names = FEATURE_NAMES + list(interactions.keys())
    
    logger.info(
        f"Added {len(interactions)} custom interaction features. "
        f"Total features: {X_augmented.shape[1]}"
    )
    return X_augmented, new_feature_names


def engineer_features(X_train: np.ndarray, X_test: np.ndarray, 
                      y_train: np.ndarray,
                      use_polynomial: bool = False,
                      use_interactions: bool = True,
                      poly_degree: int = 2) -> dict:
    """
    Full feature engineering pipeline.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target (for MI computation)
        use_polynomial: Whether to add polynomial features
        use_interactions: Whether to add custom interaction features
        poly_degree: Degree for polynomial features
        
    Returns:
        Dictionary with engineered features and metadata
    """
    logger.info("Starting feature engineering...")
    
    feature_names = list(FEATURE_NAMES)
    
    if use_interactions:
        X_train, feature_names = add_custom_interaction_features(X_train)
        X_test_aug = np.hstack([
            X_test,
            np.column_stack([
                X_test[:, 0] * X_test[:, 5],  # CIC0_x_MLOGP
                X_test[:, 2] * X_test[:, 5],  # GATS1i_x_MLOGP
                X_test[:, 3] * X_test[:, 4],  # NdsCH_x_NdssC
                X_test[:, 0] * X_test[:, 2],  # CIC0_x_GATS1i
                X_test[:, 5] ** 2,             # MLOGP_squared
                X_test[:, 0] ** 2,             # CIC0_squared
            ])
        ])
        X_test = X_test_aug
    
    if use_polynomial:
        X_train, poly, feature_names = add_polynomial_features(X_train, degree=poly_degree)
        X_test = poly.transform(X_test)
        feature_names = list(feature_names)
    
    # Compute mutual information for feature importance
    mi_scores = compute_mutual_information(X_train, y_train)
    
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'feature_names': feature_names,
        'mi_scores': mi_scores,
    }
    
    logger.info(f"Feature engineering complete. Final feature count: {X_train.shape[1]}")
    return result

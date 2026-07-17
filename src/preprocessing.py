"""
EcotoxAI: Preprocessing Module
Handles feature scaling, outlier removal, train/test splitting,
and cross-validation setup.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging
import joblib
import os

from src.data_loader import FEATURE_NAMES, TARGET_NAME, PROJECT_ROOT

logger = logging.getLogger(__name__)

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
MODELS_DIR = os.path.join(OUTPUTS_DIR, 'models')


def remove_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from the dataset using IQR or Z-score method.
    
    Args:
        df: Input DataFrame
        method: 'iqr' (Interquartile Range) or 'zscore'
        threshold: IQR multiplier (default 1.5) or Z-score threshold (default 3.0)
        
    Returns:
        DataFrame with outliers removed
    """
    initial_count = len(df)
    
    if method == 'iqr':
        mask = pd.Series([True] * len(df), index=df.index)
        for col in FEATURE_NAMES + [TARGET_NAME]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            mask &= (df[col] >= lower) & (df[col] <= upper)
        df = df[mask].reset_index(drop=True)
        
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[FEATURE_NAMES + [TARGET_NAME]]))
        mask = (z_scores < threshold).all(axis=1)
        df = df[mask].reset_index(drop=True)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'.")
    
    removed = initial_count - len(df)
    logger.info(f"Outlier removal ({method}): removed {removed} rows, {len(df)} remain.")
    return df


def split_features_target(df: pd.DataFrame) -> tuple:
    """
    Split DataFrame into features (X) and target (y).
    
    Returns:
        (X, y) tuple of numpy arrays
    """
    X = df[FEATURE_NAMES].values
    y = df[TARGET_NAME].values
    return X, y


def create_train_test_split(X: np.ndarray, y: np.ndarray, 
                            test_size: float = 0.2, 
                            random_state: int = 42) -> tuple:
    """
    Create train/test split with fixed random seed for reproducibility.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Fraction for test set (default 0.2 = 80/20 split)
        random_state: Random seed for reproducibility
        
    Returns:
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(
        f"Train/Test split: {len(X_train)} training / {len(X_test)} test samples "
        f"({(1-test_size)*100:.0f}/{test_size*100:.0f} split)"
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train: np.ndarray, X_test: np.ndarray, 
                   method: str = 'standard') -> tuple:
    """
    Scale features using the specified method. Fit on train, transform both.
    
    Args:
        X_train: Training features
        X_test: Test features
        method: 'standard' (z-score), 'minmax' (0-1), or 'robust' (median/IQR)
        
    Returns:
        (X_train_scaled, X_test_scaled, scaler)
    """
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler(),
    }
    
    if method not in scalers:
        raise ValueError(f"Unknown scaling method: {method}. Use: {list(scalers.keys())}")
    
    scaler = scalers[method]
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Feature scaling applied: {method}")
    return X_train_scaled, X_test_scaled, scaler


def get_cv_splitter(n_splits: int = 10, shuffle: bool = True, 
                    random_state: int = 42) -> KFold:
    """
    Create a K-Fold cross-validation splitter.
    
    Args:
        n_splits: Number of folds (default 10)
        shuffle: Whether to shuffle before splitting
        random_state: Random seed
        
    Returns:
        KFold splitter object
    """
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def save_scaler(scaler, name: str = 'scaler'):
    """Save a fitted scaler to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f'{name}.joblib')
    joblib.dump(scaler, path)
    logger.info(f"Scaler saved to {path}")


def load_scaler(name: str = 'scaler'):
    """Load a saved scaler from disk."""
    path = os.path.join(MODELS_DIR, f'{name}.joblib')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved scaler found at {path}")
    return joblib.load(path)


def preprocess_pipeline(df: pd.DataFrame, 
                        remove_outliers_flag: bool = True,
                        outlier_method: str = 'iqr',
                        scale_method: str = 'standard',
                        test_size: float = 0.2,
                        random_state: int = 42) -> dict:
    """
    Full preprocessing pipeline: outlier removal → split → scale.
    
    Args:
        df: Raw validated DataFrame
        remove_outliers_flag: Whether to remove outliers
        outlier_method: Method for outlier removal
        scale_method: Method for feature scaling
        test_size: Test set fraction
        random_state: Random seed
        
    Returns:
        Dictionary with all preprocessed data and objects:
        {
            'X_train', 'X_test', 'y_train', 'y_test',
            'X_train_scaled', 'X_test_scaled',
            'scaler', 'cv_splitter',
            'df_clean': cleaned DataFrame
        }
    """
    logger.info("Starting preprocessing pipeline...")
    
    # Step 1: Remove outliers
    if remove_outliers_flag:
        df_clean = remove_outliers(df, method=outlier_method)
    else:
        df_clean = df.copy()
    
    # Step 2: Split features and target
    X, y = split_features_target(df_clean)
    
    # Step 3: Train/test split
    X_train, X_test, y_train, y_test = create_train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Step 4: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, method=scale_method
    )
    
    # Step 5: Save scaler for later use
    save_scaler(scaler)
    
    # Step 6: Create CV splitter
    cv = get_cv_splitter()
    
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler,
        'cv_splitter': cv,
        'df_clean': df_clean,
    }
    
    logger.info("Preprocessing pipeline complete.")
    return result

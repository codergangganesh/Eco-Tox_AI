"""
EcoTox-AI: Data Loader Module
Loads and validates the UCI QSAR Fish Toxicity dataset.
Supports both local CSV and automatic download via ucimlrepo.
"""

import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Column names for the UCI QSAR datasets
FISH_FEATURE_NAMES = ['CIC0', 'SM1_Dz_Z', 'GATS1i', 'NdsCH', 'NdssC', 'MLOGP']
DAPHNIA_FEATURE_NAMES = ['TPSA_Tot', 'SAacc', 'H_050', 'MLOGP', 'RDCHI', 'GATS1p', 'nN', 'C_040']
FEATURE_NAMES = FISH_FEATURE_NAMES  # For backward compatibility
TARGET_NAME = 'LC50'
ALL_COLUMNS = FEATURE_NAMES + [TARGET_NAME]

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, 'qsar_fish_toxicity.csv')


def get_feature_names(dataset_type: str = 'fish') -> list:
    """Return the list of feature names for the specified dataset type."""
    if dataset_type == 'daphnia':
        return DAPHNIA_FEATURE_NAMES
    return FISH_FEATURE_NAMES


def load_from_csv(filepath: str = None, dataset_type: str = 'fish') -> pd.DataFrame:
    """
    Load the dataset from a local CSV file.
    
    Args:
        filepath: Path to the CSV file. If None, uses default path for the dataset_type.
        dataset_type: 'fish' or 'daphnia'
        
    Returns:
        pd.DataFrame with named columns
    """
    if filepath is None:
        if dataset_type == 'daphnia':
            filepath = os.path.join(DATA_DIR, 'qsar_aquatic_toxicity.csv')
        else:
            filepath = DEFAULT_DATA_PATH
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            f"Please run train.py to download it automatically."
        )
    
    logger.info(f"Loading {dataset_type} dataset from {filepath}")
    
    features = get_feature_names(dataset_type)
    all_cols = features + [TARGET_NAME]
    
    # UCI datasets use semicolons as separators and have no header
    df = pd.read_csv(filepath, sep=';', header=None, names=all_cols)
    
    logger.info(f"Loaded {len(df)} records with {len(features)} features")
    return df


def load_from_ucimlrepo(dataset_type: str = 'fish') -> pd.DataFrame:
    """
    Download the dataset directly from UCI ML Repository using the ucimlrepo package.
    Also saves a local copy to data/ for future use.
    
    Args:
        dataset_type: 'fish' or 'daphnia'
        ~
    Returns:
        pd.DataFrame with named columns
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise ImportError(
            "ucimlrepo package not installed. Run: pip install ucimlrepo"
        )
    
    if dataset_type == 'daphnia':
        logger.info("Downloading QSAR Aquatic Toxicity dataset (Daphnia) from UCI ML Repository...")
        dataset = fetch_ucirepo(id=505)
        save_path = os.path.join(DATA_DIR, 'qsar_aquatic_toxicity.csv')
    else:
        logger.info("Downloading QSAR Fish Toxicity dataset from UCI ML Repository...")
        dataset = fetch_ucirepo(id=504)
        save_path = DEFAULT_DATA_PATH
    
    X = dataset.data.features
    y = dataset.data.targets
    
    # Combine features and target into a single DataFrame
    df = pd.concat([X, y], axis=1)
    
    features = get_feature_names(dataset_type)
    all_cols = features + [TARGET_NAME]
    df.columns = all_cols
    
    # Save locally for future use
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(save_path, sep=';', header=False, index=False)
    logger.info(f"Dataset saved locally to {save_path}")
    
    logger.info(f"Downloaded {len(df)} records with {len(features)} features")
    return df


def load_data(filepath: str = None, auto_download: bool = True, dataset_type: str = 'fish') -> pd.DataFrame:
    """
    Smart loader: tries local CSV first, falls back to UCI download.
    
    Args:
        filepath: Path to local CSV. If None, uses default path.
        auto_download: If True and local file missing, download from UCI.
        dataset_type: 'fish' or 'daphnia'
        
    Returns:
        Validated pd.DataFrame
    """
    try:
        df = load_from_csv(filepath, dataset_type=dataset_type)
    except FileNotFoundError:
        if auto_download:
            logger.info("Local file not found. Attempting automatic download...")
            df = load_from_ucimlrepo(dataset_type=dataset_type)
        else:
            raise
    
    # Validate the loaded data
    df = validate_data(df, dataset_type=dataset_type)
    return df


def validate_data(df: pd.DataFrame, dataset_type: str = 'fish') -> pd.DataFrame:
    """
    Validate dataset integrity: check for missing values, correct types,
    and reasonable value ranges.
    
    Args:
        df: Raw DataFrame
        dataset_type: 'fish' or 'daphnia'
        
    Returns:
        Validated DataFrame (rows with issues may be dropped)
    """
    logger.info(f"Validating {dataset_type} dataset...")
    initial_count = len(df)
    
    features = get_feature_names(dataset_type)
    all_cols = features + [TARGET_NAME]
    
    # Check column count
    if len(df.columns) != len(all_cols):
        raise ValueError(
            f"Expected {len(all_cols)} columns, got {len(df.columns)}. "
            f"Ensure the dataset is the correct QSAR dataset for {dataset_type}."
        )
    
    # Ensure column names
    df.columns = all_cols
    
    # Convert all columns to numeric, coercing errors
    for col in all_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with any missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.warning(f"Found {missing_count} missing values. Dropping affected rows.")
        df = df.dropna().reset_index(drop=True)
    
    # Validate LC50 target range (should be positive in -log(mol/L) scale)
    invalid_target = df[df[TARGET_NAME] < 0]
    if len(invalid_target) > 0:
        logger.warning(f"Found {len(invalid_target)} records with negative LC50. Dropping.")
        df = df[df[TARGET_NAME] >= 0].reset_index(drop=True)
    
    final_count = len(df)
    dropped = initial_count - final_count
    if dropped > 0:
        logger.info(f"Validation complete: dropped {dropped} invalid rows. {final_count} records remain.")
    else:
        logger.info(f"Validation complete: all {final_count} records are valid.")
    
    return df


def get_data_summary(df: pd.DataFrame, dataset_type: str = 'fish') -> dict:
    """
    Generate a comprehensive summary of the dataset.
    
    Args:
        df: Validated DataFrame
        dataset_type: 'fish' or 'daphnia'
        
    Returns:
        Dictionary containing summary statistics
    """
    features = get_feature_names(dataset_type)
    summary = {
        'total_records': len(df),
        'num_features': len(features),
        'feature_names': features,
        'target_name': TARGET_NAME,
        'target_stats': {
            'mean': float(df[TARGET_NAME].mean()),
            'std': float(df[TARGET_NAME].std()),
            'min': float(df[TARGET_NAME].min()),
            'max': float(df[TARGET_NAME].max()),
            'median': float(df[TARGET_NAME].median()),
        },
        'feature_stats': {}
    }
    
    for feat in features:
        summary['feature_stats'][feat] = {
            'mean': float(df[feat].mean()),
            'std': float(df[feat].std()),
            'min': float(df[feat].min()),
            'max': float(df[feat].max()),
        }
    
    return summary


def print_data_summary(df: pd.DataFrame, dataset_type: str = 'fish'):
    """Print a formatted summary of the dataset to console."""
    summary = get_data_summary(df, dataset_type=dataset_type)
    features = get_feature_names(dataset_type)
    
    print("\n" + "=" * 60)
    print(f"  EcotoxAI — {dataset_type.upper()} Dataset Summary")
    print("=" * 60)
    print(f"  Total records:  {summary['total_records']}")
    print(f"  Features:       {summary['num_features']}")
    print(f"  Target:         {summary['target_name']} [-LOG(mol/L)]")
    print("-" * 60)
    print(f"  LC50 Range:     {summary['target_stats']['min']:.3f} to {summary['target_stats']['max']:.3f}")
    print(f"  LC50 Mean:      {summary['target_stats']['mean']:.3f} ± {summary['target_stats']['std']:.3f}")
    print(f"  LC50 Median:    {summary['target_stats']['median']:.3f}")
    print("-" * 60)
    print("  Feature Statistics:")
    print(f"  {'Feature':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for feat in features:
        stats = summary['feature_stats'][feat]
        print(f"  {feat:<12} {stats['mean']:>8.3f} {stats['std']:>8.3f} {stats['min']:>8.3f} {stats['max']:>8.3f}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    # Quick test: load and summarize both
    df_fish = load_data(dataset_type='fish')
    print_data_summary(df_fish, dataset_type='fish')
    
    df_daphnia = load_data(dataset_type='daphnia')
    print_data_summary(df_daphnia, dataset_type='daphnia')

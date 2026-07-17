"""
EcoTox-AI: Data Loader Module
Loads and validates the UCI QSAR Fish Toxicity dataset.
Supports both local CSV and automatic download via ucimlrepo.
"""

import os
import glob
import json
import zipfile
import tempfile
import urllib.request
from urllib.parse import urlparse
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Column names for the UCI QSAR Fish Toxicity dataset
FEATURE_NAMES = ['CIC0', 'SM1_Dz_Z', 'GATS1i', 'NdsCH', 'NdssC', 'MLOGP']
TARGET_NAME = 'LC50'
ALL_COLUMNS = FEATURE_NAMES + [TARGET_NAME]

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, 'qsar_fish_toxicity.csv')
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, 'external')
REMOTE_DATASET_LINKS_PATH = os.path.join(DATA_DIR, 'dataset_links.json')


def load_from_csv(filepath: str = None) -> pd.DataFrame:
    """
    Load the QSAR Fish Toxicity dataset from a local CSV file.
    
    The UCI dataset uses semicolons as delimiters and has no header row.
    
    Args:
        filepath: Path to the CSV file. Defaults to data/qsar_fish_toxicity.csv
        
    Returns:
        pd.DataFrame with named columns
    """
    if filepath is None:
        filepath = DEFAULT_DATA_PATH
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Download from https://archive.ics.uci.edu/dataset/504/qsar+fish+toxicity "
            "or use load_from_ucimlrepo() for automatic download."
        )
    
    logger.info(f"Loading dataset from {filepath}")
    
    # UCI dataset uses semicolons as separators and has no header
    df = pd.read_csv(filepath, sep=';', header=None, names=ALL_COLUMNS)
    
    logger.info(f"Loaded {len(df)} records with {len(FEATURE_NAMES)} features")
    return df


def load_external_same_schema_data(directory: str = EXTERNAL_DATA_DIR) -> pd.DataFrame:
    """
    Load optional external training files that already match this model's schema.

    This intentionally accepts only the six established descriptors plus LC50 so
    the current model, scaler, prediction API, and web workflow remain unchanged.
    Put CSV files in data/external/ with either headers matching ALL_COLUMNS or
    no header in the same semicolon-delimited order as qsar_fish_toxicity.csv.
    """
    if not os.path.isdir(directory):
        return pd.DataFrame(columns=ALL_COLUMNS)

    frames = []
    for path in sorted(glob.glob(os.path.join(directory, '*.csv'))):
        logger.info(f"Loading optional external training data from {path}")
        sample = pd.read_csv(path, nrows=1, sep=None, engine='python')
        has_named_columns = set(ALL_COLUMNS).issubset(sample.columns)

        if has_named_columns:
            df = pd.read_csv(path, sep=None, engine='python')
            df = df[ALL_COLUMNS]
        else:
            df = pd.read_csv(path, sep=';', header=None, names=ALL_COLUMNS)

        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=ALL_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(combined)} optional external records")
    return combined


def _read_same_schema_csv(path: str) -> pd.DataFrame:
    """Read a CSV that either has project headers or follows the UCI no-header order."""
    sample = pd.read_csv(path, nrows=1, sep=None, engine='python')
    has_named_columns = set(ALL_COLUMNS).issubset(sample.columns)

    if has_named_columns:
        df = pd.read_csv(path, sep=None, engine='python')
        return df[ALL_COLUMNS]

    return pd.read_csv(path, sep=';', header=None, names=ALL_COLUMNS)


def _download_file(url: str, destination: str):
    """Download a URL to a local destination using only the Python standard library."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'EcoTox-AI/1.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        with open(destination, 'wb') as out_file:
            out_file.write(response.read())


def _extract_csv_from_zip(zip_path: str, output_path: str):
    """Extract the first CSV from a zip archive into output_path."""
    with zipfile.ZipFile(zip_path) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith('.csv')]
        if not csv_members:
            raise ValueError(f"No CSV file found in remote archive: {zip_path}")

        with archive.open(csv_members[0]) as source, open(output_path, 'wb') as target:
            target.write(source.read())


def sync_remote_dataset_links(config_path: str = REMOTE_DATASET_LINKS_PATH,
                              output_dir: str = EXTERNAL_DATA_DIR) -> list:
    """
    Download enabled remote CSV/ZIP dataset links into data/external/.

    Only links declared as same_schema are accepted for training. This keeps the
    model's feature contract unchanged while allowing training data to come from
    URLs instead of manual uploads.
    """
    if not os.path.exists(config_path):
        return []

    with open(config_path, 'r', encoding='utf-8') as f:
        sources = json.load(f)

    downloaded = []
    os.makedirs(output_dir, exist_ok=True)

    for source in sources:
        if not source.get('enabled', False):
            continue
        if not source.get('same_schema', False):
            logger.warning(
                "Skipping remote dataset '%s': it is not marked as same-schema.",
                source.get('name', 'unnamed')
            )
            continue

        name = source.get('name', 'remote_dataset')
        url = source['url']
        output_path = os.path.join(output_dir, f"{name}.csv")

        if os.path.exists(output_path) and not source.get('refresh', False):
            logger.info(f"Using cached remote dataset: {output_path}")
            downloaded.append(output_path)
            continue

        logger.info(f"Downloading remote training dataset '{name}' from {url}")
        parsed_path = urlparse(url).path.lower()

        try:
            if parsed_path.endswith('.zip'):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    zip_path = os.path.join(tmp_dir, f"{name}.zip")
                    _download_file(url, zip_path)
                    _extract_csv_from_zip(zip_path, output_path)
            elif parsed_path.endswith('.csv'):
                _download_file(url, output_path)
            else:
                logger.warning(
                    "Skipping remote dataset '%s': only direct CSV or ZIP links are supported.",
                    name
                )
                continue

            # Fail early if the downloaded file does not match the training schema.
            _read_same_schema_csv(output_path)
            downloaded.append(output_path)
            logger.info(f"Remote dataset cached for training: {output_path}")
        except Exception as exc:
            logger.warning(f"Could not use remote dataset '{name}': {exc}")

    return downloaded


def load_from_ucimlrepo() -> pd.DataFrame:
    """
    Download the dataset directly from UCI ML Repository using the ucimlrepo package.
    Also saves a local copy to data/ for future use.
    
    Returns:
        pd.DataFrame with named columns
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise ImportError(
            "ucimlrepo package not installed. Run: pip install ucimlrepo"
        )
    
    logger.info("Downloading QSAR Fish Toxicity dataset from UCI ML Repository...")
    dataset = fetch_ucirepo(id=504)
    
    X = dataset.data.features
    y = dataset.data.targets
    
    # Combine features and target into a single DataFrame
    df = pd.concat([X, y], axis=1)
    df.columns = ALL_COLUMNS
    
    # Save locally for future use
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DEFAULT_DATA_PATH, sep=';', header=False, index=False)
    logger.info(f"Dataset saved locally to {DEFAULT_DATA_PATH}")
    
    logger.info(f"Downloaded {len(df)} records with {len(FEATURE_NAMES)} features")
    return df


def load_data(filepath: str = None, auto_download: bool = True,
              include_external: bool = True, sync_remote_links: bool = True) -> pd.DataFrame:
    """
    Smart loader: tries local CSV first, falls back to UCI download.
    
    Args:
        filepath: Path to local CSV. If None, uses default path.
        auto_download: If True and local file missing, download from UCI.
        include_external: If True, append compatible files from data/external/.
        sync_remote_links: If True, download configured compatible URL sources first.
        
    Returns:
        Validated pd.DataFrame
    """
    try:
        df = load_from_csv(filepath)
    except FileNotFoundError:
        if auto_download:
            logger.info("Local file not found. Attempting automatic download...")
            df = load_from_ucimlrepo()
        else:
            raise
    
    if include_external:
        if sync_remote_links:
            sync_remote_dataset_links()

        external_df = load_external_same_schema_data()
        if not external_df.empty:
            before = len(df)
            df = pd.concat([df, external_df], ignore_index=True)
            df = df.drop_duplicates().reset_index(drop=True)
            logger.info(
                "Combined base dataset with optional external data: "
                f"{before} base + {len(external_df)} external -> {len(df)} unique records"
            )

    # Validate the loaded data
    df = validate_data(df)
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate dataset integrity: check for missing values, correct types,
    and reasonable value ranges.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Validated DataFrame (rows with issues may be dropped)
    """
    logger.info("Validating dataset...")
    initial_count = len(df)
    
    # Check column count
    if len(df.columns) != len(ALL_COLUMNS):
        raise ValueError(
            f"Expected {len(ALL_COLUMNS)} columns, got {len(df.columns)}. "
            "Ensure the dataset is the UCI QSAR Fish Toxicity dataset."
        )
    
    # Ensure column names
    df.columns = ALL_COLUMNS
    
    # Convert all columns to numeric, coercing errors
    for col in ALL_COLUMNS:
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


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a comprehensive summary of the dataset.
    
    Args:
        df: Validated DataFrame
        
    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        'total_records': len(df),
        'num_features': len(FEATURE_NAMES),
        'feature_names': FEATURE_NAMES,
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
    
    for feat in FEATURE_NAMES:
        summary['feature_stats'][feat] = {
            'mean': float(df[feat].mean()),
            'std': float(df[feat].std()),
            'min': float(df[feat].min()),
            'max': float(df[feat].max()),
        }
    
    return summary


def print_data_summary(df: pd.DataFrame):
    """Print a formatted summary of the dataset to console."""
    summary = get_data_summary(df)
    
    print("\n" + "=" * 60)
    print("  EcoTox-AI — Dataset Summary")
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
    for feat in FEATURE_NAMES:
        stats = summary['feature_stats'][feat]
        print(f"  {feat:<12} {stats['mean']:>8.3f} {stats['std']:>8.3f} {stats['min']:>8.3f} {stats['max']:>8.3f}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    # Quick test: load and summarize
    df = load_data()
    print_data_summary(df)

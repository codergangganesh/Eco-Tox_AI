"""
EcoTox-AI: Master Training Script
=================================
Single entry point that orchestrates the full training pipeline:
  1. Load & validate data
  2. Preprocess (outliers, scaling, splitting)
  3. Train all 9 models (3 linear + 3 tree + SVR + KNN + Deep MLP)
  4. Build ensemble from top 3
  5. Evaluate all models on test set
  6. Run explainability analysis
  7. Generate comparison plots & reports
  8. Save the best model + ensemble

Usage:
    python train.py
"""

import os
import sys
import time
import json
import joblib
import logging
import warnings
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_data, print_data_summary, FEATURE_NAMES
from src.preprocessing import preprocess_pipeline
from src.algorithms.regression.linear_regression_models import train_all_linear_models
from src.algorithms.regression.tree_based_regression_models import train_all_tree_models
from src.algorithms.regression.support_vector_regression import train_svr
from src.algorithms.regression.knn_regression import train_knn
from src.algorithms.regression.neural_network_regression import train_neural_network, save_nn_model
from src.algorithms.regression.ensemble_regression import build_ensembles
from src.evaluation import full_evaluation
from src.explainability import run_explainability

MODELS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'models')


def save_best_model(all_results: list, model_info: dict):
    """Save the best performing model and metadata."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Find best sklearn model
    sklearn_results = [r for r in all_results if not r.get('is_keras', False)]
    best = max(sklearn_results, key=lambda r: r['cv_score'])
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'best_model.joblib')
    joblib.dump(best['model'], model_path)
    logger.info(f"Best model ({best['name']}) saved to {model_path}")
    
    # Save model info
    info = {
        'best_model_name': best['name'],
        'best_cv_r2': float(best['cv_score']),
        'best_params': {k: str(v) for k, v in best.get('best_params', {}).items()},
        'feature_names': FEATURE_NAMES,
        **model_info,
    }
    info_path = os.path.join(MODELS_DIR, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    logger.info(f"Model info saved to {info_path}")
    
    # Save all sklearn models
    for result in sklearn_results:
        safe_name = result['name'].replace(' ', '_').replace('(', '').replace(')', '').lower()
        path = os.path.join(MODELS_DIR, f'{safe_name}.joblib')
        joblib.dump(result['model'], path)
    logger.info(f"All {len(sklearn_results)} sklearn models saved.")


def main():
    """Run the full EcoTox-AI training pipeline."""
    
    start_time = time.time()
    
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + "  🧪 EcoTox-AI: Aquatic Toxicity Prediction Platform  ".center(58) + "║")
    print("║" + "  Training Pipeline v1.0  ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # ────────────────────────────────────────────────────────
    # STEP 1: Load Data
    # ────────────────────────────────────────────────────────
    logger.info("STEP 1/8: Loading Dataset...")
    df = load_data(auto_download=True)
    print_data_summary(df)
    
    # ────────────────────────────────────────────────────────
    # STEP 2: Preprocess
    # ────────────────────────────────────────────────────────
    logger.info("STEP 2/8: Preprocessing Data...")
    data = preprocess_pipeline(
        df, 
        remove_outliers_flag=True,
        outlier_method='iqr',
        scale_method='standard',
        test_size=0.2,
        random_state=42
    )
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    cv = data['cv_splitter']
    
    logger.info(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # ────────────────────────────────────────────────────────
    # STEP 3: Train Linear Models
    # ────────────────────────────────────────────────────────
    logger.info("STEP 3/8: Training Linear Models...")
    linear_results = train_all_linear_models(X_train_scaled, y_train, cv)
    
    # ────────────────────────────────────────────────────────
    # STEP 4: Train Tree-Based Models
    # ────────────────────────────────────────────────────────
    logger.info("STEP 4/8: Training Tree-Based Models...")
    tree_results = train_all_tree_models(X_train_scaled, y_train, cv)
    
    # ────────────────────────────────────────────────────────
    # STEP 5: Train SVR & KNN
    # ────────────────────────────────────────────────────────
    logger.info("STEP 5/8: Training SVR & KNN...")
    svr_result = train_svr(X_train_scaled, y_train, cv)
    knn_result = train_knn(X_train_scaled, y_train, cv)
    
    # ────────────────────────────────────────────────────────
    # STEP 6: Train Neural Network
    # ────────────────────────────────────────────────────────
    logger.info("STEP 6/8: Training Deep Neural Network...")
    nn_result = train_neural_network(
        X_train_scaled, y_train,
        epochs=500, batch_size=32, patience=30
    )
    
    # Save NN model separately
    os.makedirs(MODELS_DIR, exist_ok=True)
    nn_model_path = os.path.join(MODELS_DIR, 'deep_mlp.joblib')
    save_nn_model(nn_result['model'], nn_model_path)
    
    # ────────────────────────────────────────────────────────
    # Collect all individual results
    # ────────────────────────────────────────────────────────
    all_individual_results = (
        linear_results + 
        tree_results + 
        [svr_result, knn_result, nn_result]
    )
    
    # ────────────────────────────────────────────────────────
    # STEP 7: Build Ensembles
    # ────────────────────────────────────────────────────────
    logger.info("STEP 7/8: Building Ensemble Models...")
    
    # For ensembles, we need to decide which data to use.
    # Tree models use unscaled data, linear/SVR/KNN use scaled.
    # Use scaled data for ensembles since the top models typically include
    # a mix — but only for sklearn models.
    # Actually, let's use unscaled data and only include tree + linear models
    # in the ensemble to avoid mixing scaled/unscaled requirements.
    
    # We'll train ensembles on scaled data since most models benefit from it
    ensemble_results = build_ensembles(
        all_individual_results, 
        X_train_scaled, y_train, cv, top_n=3
    )
    
    all_results = all_individual_results + ensemble_results
    
    # ────────────────────────────────────────────────────────
    # STEP 8: Evaluate & Generate Reports
    # ────────────────────────────────────────────────────────
    logger.info("STEP 8/8: Evaluating All Models...")
    
    comparison_df = full_evaluation(
        all_results, 
        X_test_scaled, y_test,  # Use scaled for consistent comparison
        X_train_scaled, y_train, 
        cv,
        X_test_scaled=X_test_scaled,
        X_train_scaled=X_train_scaled
    )
    
    # ────────────────────────────────────────────────────────
    # Explainability Analysis
    # ────────────────────────────────────────────────────────
    logger.info("Running Explainability Analysis...")
    try:
        # For tree-based models, they were trained on scaled data
        run_explainability(
            tree_results, X_train_scaled, X_test_scaled, y_test,
            feature_names=FEATURE_NAMES
        )
    except Exception as e:
        logger.warning(f"Explainability analysis encountered issues: {e}")
    
    # ────────────────────────────────────────────────────────
    # Save Best Model
    # ────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    model_info = {
        'training_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'n_features': int(X_train.shape[1]),
        'total_models_trained': len(all_results),
        'training_time_seconds': round(elapsed, 1),
    }
    
    save_best_model(all_results, model_info)
    
    # ────────────────────────────────────────────────────────
    # Final Summary
    # ────────────────────────────────────────────────────────
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + "  ✅ Training Complete!  ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print(f"  Total training time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"  Models trained: {len(all_results)}")
    print(f"  Best model: {comparison_df.iloc[0]['Model']} (R² = {comparison_df.iloc[0]['R²']:.4f})")
    print(f"  Results saved to: outputs/")
    print(f"  Models saved to: outputs/models/")
    print(f"  Plots saved to: outputs/plots/")
    print()
    print("  Next steps:")
    print("    1. Review plots in outputs/plots/")
    print("    2. Run the web app: python web/app.py")
    print("    3. Make predictions: python -m src.predict")
    print()
    
    return comparison_df


if __name__ == '__main__':
    main()

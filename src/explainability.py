"""
EcoTox-AI: Explainability Module
SHAP values, feature importance rankings, and permutation importance
to explain which molecular descriptors drive toxicity predictions.
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

from src.data_loader import FEATURE_NAMES, PROJECT_ROOT

logger = logging.getLogger(__name__)

PLOTS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'plots')


def compute_shap_values(model, X_train, X_test, model_name: str = '',
                        feature_names: list = None):
    """
    Compute SHAP values for a tree-based model.
    
    SHAP (SHapley Additive exPlanations) assigns each feature a contribution
    value for each prediction. Positive SHAP = pushes prediction toward
    higher toxicity; Negative SHAP = pushes toward lower toxicity.
    
    Args:
        model: Trained tree-based model (RF, GBR, XGBoost)
        X_train: Training data (for background/reference)
        X_test: Test data to explain
        model_name: Name for logging
        feature_names: Feature names for plots
        
    Returns:
        SHAP values array
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Run: pip install shap")
        return None
    
    if feature_names is None:
        feature_names = FEATURE_NAMES
    
    logger.info(f"Computing SHAP values for {model_name}...")
    
    try:
        # Use TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        logger.info(f"  SHAP values computed: shape {shap_values.shape}")
        return shap_values
    except Exception as e:
        logger.warning(f"  TreeExplainer failed for {model_name}: {e}")
        try:
            # Fallback to KernelExplainer
            background = shap.sample(X_train, 100)
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_test[:50])  # Subset for speed
            return shap_values
        except Exception as e2:
            logger.warning(f"  KernelExplainer also failed: {e2}")
            return None


def plot_shap_summary(shap_values, X_test, model_name: str = '',
                      feature_names: list = None):
    """
    Create SHAP summary plot (beeswarm plot).
    
    Shows:
    - Which features matter most (vertical order)
    - How feature values affect predictions (color = feature value)
    - Distribution of impacts (spread of dots)
    """
    try:
        import shap
    except ImportError:
        return
    
    if feature_names is None:
        feature_names = FEATURE_NAMES
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use only the features that match
    n_features = min(shap_values.shape[1], len(feature_names))
    shap.summary_plot(
        shap_values[:, :n_features],
        X_test[:, :n_features] if len(X_test.shape) > 1 else X_test,
        feature_names=feature_names[:n_features],
        show=False
    )
    
    plt.title(f'SHAP Feature Importance — {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    path = os.path.join(PLOTS_DIR, f'shap_summary_{safe_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"SHAP summary plot saved to {path}")


def plot_shap_bar(shap_values, feature_names: list = None, model_name: str = ''):
    """
    Create SHAP bar plot showing mean absolute SHAP values.
    Simpler than beeswarm — shows overall feature importance ranking.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    n_features = min(shap_values.shape[1], len(feature_names))
    mean_shap = np.abs(shap_values[:, :n_features]).mean(axis=0)
    
    # Sort by importance
    sorted_idx = np.argsort(mean_shap)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        mean_shap[sorted_idx],
        color=plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_idx)))
    )
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title(f'Feature Importance (SHAP) — {model_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    path = os.path.join(PLOTS_DIR, f'shap_bar_{safe_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"SHAP bar plot saved to {path}")


def plot_feature_importance_comparison(results_with_importances: list,
                                       feature_names: list = None):
    """
    Compare built-in feature importances across tree-based models.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    n_features = len(feature_names)
    x = np.arange(n_features)
    width = 0.8 / len(results_with_importances)
    
    for i, result in enumerate(results_with_importances):
        importances = result.get('feature_importances')
        if importances is None:
            continue
        imp = importances[:n_features]  # Use only base features
        offset = (i - len(results_with_importances) / 2) * width + width / 2
        ax.bar(x + offset, imp, width, label=result['name'], alpha=0.8)
    
    ax.set_xlabel('Feature')
    ax.set_ylabel('Importance')
    ax.set_title('Feature Importance Comparison (Tree-Based Models)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'feature_importance_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Feature importance comparison saved to {path}")


def compute_permutation_importance(model, X_test, y_test, 
                                    feature_names: list = None,
                                    model_name: str = '',
                                    n_repeats: int = 10):
    """
    Compute permutation importance (model-agnostic).
    
    Shuffles each feature and measures how much the prediction accuracy drops.
    Larger drop = more important feature.
    """
    from sklearn.inspection import permutation_importance
    
    if feature_names is None:
        feature_names = FEATURE_NAMES
    
    logger.info(f"Computing permutation importance for {model_name}...")
    
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
        scoring='r2'
    )
    
    # Log results
    for name, imp_mean, imp_std in zip(
        feature_names[:X_test.shape[1]], 
        result.importances_mean[:len(feature_names)], 
        result.importances_std[:len(feature_names)]
    ):
        logger.info(f"  {name}: {imp_mean:.4f} ± {imp_std:.4f}")
    
    return result


def run_explainability(all_results: list, X_train, X_test, y_test,
                       feature_names: list = None):
    """
    Run the full explainability pipeline:
    1. SHAP values for tree-based models
    2. Built-in feature importance comparison
    3. Permutation importance for best model
    """
    logger.info("=" * 60)
    logger.info("Running Explainability Analysis")
    logger.info("=" * 60)
    
    if feature_names is None:
        feature_names = FEATURE_NAMES
    
    # Get tree-based models
    tree_models = [r for r in all_results if 'feature_importances' in r]
    
    # Plot built-in feature importance comparison
    if tree_models:
        plot_feature_importance_comparison(tree_models, feature_names)
    
    # SHAP for tree-based models
    for result in tree_models:
        try:
            shap_values = compute_shap_values(
                result['model'], X_train, X_test, 
                result['name'], feature_names
            )
            if shap_values is not None:
                plot_shap_summary(shap_values, X_test, result['name'], feature_names)
                plot_shap_bar(shap_values, feature_names, result['name'])
        except Exception as e:
            logger.warning(f"SHAP analysis failed for {result['name']}: {e}")
    
    # Permutation importance for the best model
    best_result = max(all_results, key=lambda r: r['cv_score'])
    if not best_result.get('is_keras', False):
        try:
            compute_permutation_importance(
                best_result['model'], X_test, y_test,
                feature_names, best_result['name']
            )
        except Exception as e:
            logger.warning(f"Permutation importance failed: {e}")
    
    logger.info("Explainability analysis complete.")

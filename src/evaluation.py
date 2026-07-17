"""
EcoTox-AI: Evaluation Module
Computes comprehensive metrics, generates comparison charts,
predicted vs actual plots, residual plots, and learning curves.
"""

import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, 
    mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score, learning_curve
import logging

from src.data_loader import PROJECT_ROOT

logger = logging.getLogger(__name__)

PLOTS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'plots')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'results')

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def compute_metrics(y_true, y_pred, model_name: str = '') -> dict:
    """
    Compute comprehensive regression metrics.
    
    Returns dict with R², RMSE, MAE, MAPE.
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE — handle near-zero values
    mask = np.abs(y_true) > 0.01
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
    else:
        mape = float('nan')
    
    metrics = {
        'Model': model_name,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE (%)': mape,
    }
    
    logger.info(
        f"  {model_name}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.1f}%"
    )
    return metrics


def evaluate_model(model, X_test, y_test, model_name: str = '',
                   X_train=None, y_train=None, cv=None,
                   is_keras: bool = False) -> dict:
    """
    Full evaluation of a single model on the test set + cross-validation.
    """
    # Get predictions
    if is_keras:
        y_pred = model.predict(X_test, verbose=0).flatten()
    else:
        y_pred = model.predict(X_test)
    
    # Compute metrics on test set
    metrics = compute_metrics(y_test, y_pred, model_name)
    
    # Cross-validation (only for sklearn models)
    if cv is not None and X_train is not None and y_train is not None and not is_keras:
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring='r2', n_jobs=-1
        )
        metrics['CV R² (mean)'] = cv_scores.mean()
        metrics['CV R² (std)'] = cv_scores.std()
        logger.info(f"  {model_name} CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    metrics['y_pred'] = y_pred
    return metrics


def evaluate_all_models(all_results: list, X_test, y_test,
                        X_train=None, y_train=None, cv=None,
                        X_test_scaled=None, X_train_scaled=None) -> list:
    """
    Evaluate all trained models and return metrics.
    
    Args:
        all_results: List of result dicts from model training
        X_test: Unscaled test features (for tree/ensemble models)
        y_test: Test targets
        X_train: Unscaled train features
        y_train: Train targets
        cv: Cross-validation splitter
        X_test_scaled: Scaled test features (for NN/SVR/KNN)
        X_train_scaled: Scaled train features
        
    Returns:
        List of metrics dicts for each model
    """
    logger.info("=" * 60)
    logger.info("Evaluating All Models on Test Set")
    logger.info("=" * 60)
    
    all_metrics = []
    
    for result in all_results:
        is_keras = result.get('is_keras', False)
        name = result['name']
        model = result['model']
        
        # Use scaled data for models that need it
        needs_scaling = is_keras or name in ['SVR (RBF)', 'KNN Regressor', 'Deep MLP (Neural Network)']
        
        if needs_scaling and X_test_scaled is not None:
            x_test_use = X_test_scaled
            x_train_use = X_train_scaled
        else:
            x_test_use = X_test
            x_train_use = X_train
        
        metrics = evaluate_model(
            model, x_test_use, y_test, name,
            X_train=x_train_use, y_train=y_train,
            cv=cv, is_keras=is_keras
        )
        all_metrics.append(metrics)
    
    return all_metrics


def create_comparison_table(all_metrics: list) -> pd.DataFrame:
    """Create a DataFrame comparison table of all model metrics."""
    table_data = []
    for m in all_metrics:
        row = {k: v for k, v in m.items() if k != 'y_pred'}
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    df = df.sort_values('R²', ascending=False).reset_index(drop=True)
    df.index += 1  # 1-indexed ranking
    df.index.name = 'Rank'
    
    return df


def save_comparison_report(comparison_df: pd.DataFrame):
    """Save the comparison table as CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    comparison_df.to_csv(path)
    logger.info(f"Model comparison report saved to {path}")


def plot_model_comparison(comparison_df: pd.DataFrame):
    """
    Create a bar chart comparing all models across key metrics.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('EcotoxAI — Model Comparison', fontsize=16, fontweight='bold')
    
    models = comparison_df['Model'].values
    colors = sns.color_palette('husl', len(models))
    
    # R² Score
    ax = axes[0]
    bars = ax.barh(models, comparison_df['R²'].values, color=colors)
    ax.set_xlabel('R² Score')
    ax.set_title('R² Score (higher = better)')
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, comparison_df['R²'].values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    
    # RMSE
    ax = axes[1]
    bars = ax.barh(models, comparison_df['RMSE'].values, color=colors)
    ax.set_xlabel('RMSE')
    ax.set_title('RMSE (lower = better)')
    for bar, val in zip(bars, comparison_df['RMSE'].values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    
    # MAE
    ax = axes[2]
    bars = ax.barh(models, comparison_df['MAE'].values, color=colors)
    ax.set_xlabel('MAE')
    ax.set_title('MAE (lower = better)')
    for bar, val in zip(bars, comparison_df['MAE'].values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'model_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Model comparison chart saved to {path}")


def plot_predicted_vs_actual(all_metrics: list, y_test):
    """
    Create predicted vs actual scatter plots for all models.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    n_models = len(all_metrics)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    fig.suptitle('EcotoxAI — Predicted vs Actual LC50', fontsize=16, fontweight='bold')
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, metrics in enumerate(all_metrics):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        
        y_pred = metrics['y_pred']
        r2 = metrics['R²']
        
        ax.scatter(y_test, y_pred, alpha=0.5, s=20, edgecolors='none')
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        
        ax.set_xlabel('Actual LC50')
        ax.set_ylabel('Predicted LC50')
        ax.set_title(f"{metrics['Model']}\nR² = {r2:.4f}")
        ax.legend(fontsize=8)
    
    # Hide empty subplots
    for idx in range(n_models, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'predicted_vs_actual.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Predicted vs actual plots saved to {path}")


def plot_residuals(all_metrics: list, y_test):
    """
    Create residual distribution plots for all models.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    n_models = len(all_metrics)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    fig.suptitle('EcotoxAI — Residual Distributions', fontsize=16, fontweight='bold')
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, metrics in enumerate(all_metrics):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        
        residuals = y_test - metrics['y_pred']
        
        ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Residual (Actual - Predicted)')
        ax.set_ylabel('Frequency')
        ax.set_title(f"{metrics['Model']}\nMean Residual: {residuals.mean():.4f}")
    
    # Hide empty subplots
    for idx in range(n_models, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'residuals.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Residual plots saved to {path}")


def plot_learning_curves(model, X_train, y_train, model_name: str, cv=None):
    """
    Plot learning curves showing train vs validation error.
    Helps diagnose overfitting vs underfitting.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        cv=cv if cv is not None else 5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2',
        n_jobs=-1,
        random_state=42
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    ax.plot(train_sizes, val_mean, 'o-', color='orange', label='Validation score')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('R² Score')
    ax.set_title(f'Learning Curve — {model_name}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    path = os.path.join(PLOTS_DIR, f'learning_curve_{safe_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Learning curve for {model_name} saved to {path}")


def print_leaderboard(comparison_df: pd.DataFrame):
    """Print a formatted leaderboard to console."""
    print("\n" + "=" * 80)
    print("  EcotoxAI — MODEL LEADERBOARD")
    print("=" * 80)
    print(f"  {'Rank':<5} {'Model':<25} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}")
    print(f"  {'-'*5} {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    for rank, (_, row) in enumerate(comparison_df.iterrows(), 1):
        medal = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else '  '
        print(
            f"  {medal}{rank:<3} {row['Model']:<25} "
            f"{row['R²']:>8.4f} {row['RMSE']:>8.4f} "
            f"{row['MAE']:>8.4f} {row['MAPE (%)']:>7.1f}%"
        )
    
    print("=" * 80)
    print(f"  🏆 Best Model: {comparison_df.iloc[0]['Model']} (R² = {comparison_df.iloc[0]['R²']:.4f})")
    print("=" * 80 + "\n")


def full_evaluation(all_results: list, X_test, y_test,
                    X_train, y_train, cv=None,
                    X_test_scaled=None, X_train_scaled=None) -> pd.DataFrame:
    """
    Run the complete evaluation pipeline:
    1. Evaluate all models on test set
    2. Create comparison table
    3. Generate all plots
    4. Save reports
    5. Print leaderboard
    
    Returns:
        Comparison DataFrame
    """
    # Evaluate all models
    all_metrics = evaluate_all_models(
        all_results, X_test, y_test,
        X_train, y_train, cv,
        X_test_scaled, X_train_scaled
    )
    
    # Create comparison table
    comparison_df = create_comparison_table(all_metrics)
    
    # Save report
    save_comparison_report(comparison_df)
    
    # Generate plots
    plot_model_comparison(comparison_df)
    plot_predicted_vs_actual(all_metrics, y_test)
    plot_residuals(all_metrics, y_test)
    
    # Learning curves for top 3 sklearn models
    sklearn_results = [r for r in all_results if not r.get('is_keras', False)]
    top_3 = sorted(sklearn_results, key=lambda r: r['cv_score'], reverse=True)[:3]
    for result in top_3:
        name = result['name']
        needs_scaling = name in ['SVR (RBF)', 'KNN Regressor']
        x_train_use = X_train_scaled if needs_scaling and X_train_scaled is not None else X_train
        try:
            plot_learning_curves(result['model'], x_train_use, y_train, name, cv)
        except Exception as e:
            logger.warning(f"Could not generate learning curve for {name}: {e}")
    
    # Print leaderboard
    print_leaderboard(comparison_df)
    
    return comparison_df

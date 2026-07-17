"""
EcoTox-AI: Ensemble Models
Stacking Ensemble and Voting Ensemble built from the top-performing
individual models.
"""

import numpy as np
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)


def select_top_models(all_results: list, top_n: int = 3) -> list:
    """
    Select the top N performing models based on CV R² score.
    
    Args:
        all_results: List of result dicts from all trained models
        top_n: Number of top models to select
        
    Returns:
        List of top N result dicts, sorted by CV R² (descending)
    """
    # Filter out Keras models (they can't be used in sklearn ensembles directly)
    sklearn_results = [r for r in all_results if not r.get('is_keras', False)]
    
    # Sort by CV score descending
    sorted_results = sorted(sklearn_results, key=lambda r: r['cv_score'], reverse=True)
    top = sorted_results[:top_n]
    
    logger.info(f"Top {top_n} models selected for ensemble:")
    for i, r in enumerate(top, 1):
        logger.info(f"  {i}. {r['name']} — CV R²: {r['cv_score']:.4f}")
    
    return top


def build_stacking_ensemble(top_results: list, X_train=None, y_train=None, cv=None):
    """
    Build a Stacking Ensemble from top models.
    
    Stacking uses the predictions of base models as features for
    a meta-learner (Ridge regression). This captures complementary
    strengths of different algorithms.
    
    Args:
        top_results: List of top model result dicts
        X_train: Training features
        y_train: Training targets
        cv: Cross-validation splitter
        
    Returns:
        Result dictionary with stacking ensemble
    """
    logger.info("Building Stacking Ensemble...")
    
    # Create list of (name, estimator) tuples for base learners
    estimators = []
    for result in top_results:
        name = result['name'].replace(' ', '_').replace('(', '').replace(')', '')
        estimators.append((name, result['model']))
    
    # Ridge as meta-learner (simple, regularized, avoids overfitting)
    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1,
        passthrough=False  # Only use base model predictions as meta-features
    )
    
    # Fit the stacking ensemble
    stacking.fit(X_train, y_train)
    
    # Evaluate with cross-validation
    cv_scores = cross_val_score(
        stacking, X_train, y_train,
        cv=cv if cv is not None else 10,
        scoring='r2',
        n_jobs=-1
    )
    
    mean_cv = cv_scores.mean()
    std_cv = cv_scores.std()
    
    logger.info(f"Stacking Ensemble — CV R²: {mean_cv:.4f} ± {std_cv:.4f}")
    
    return {
        'model': stacking,
        'name': 'Stacking Ensemble',
        'best_params': {
            'base_learners': [r['name'] for r in top_results],
            'meta_learner': 'Ridge(alpha=1.0)',
        },
        'cv_score': mean_cv,
        'cv_std': std_cv,
        'cv_scores': cv_scores,
    }


def build_voting_ensemble(top_results: list, X_train=None, y_train=None, cv=None):
    """
    Build a Voting Ensemble from top models.
    
    Voting simply averages the predictions of all base models.
    Simple but often surprisingly effective as a consensus approach.
    
    Args:
        top_results: List of top model result dicts
        X_train: Training features
        y_train: Training targets
        cv: Cross-validation splitter
        
    Returns:
        Result dictionary with voting ensemble
    """
    logger.info("Building Voting Ensemble...")
    
    # Create list of (name, estimator) tuples
    estimators = []
    for result in top_results:
        name = result['name'].replace(' ', '_').replace('(', '').replace(')', '')
        estimators.append((name, result['model']))
    
    voting = VotingRegressor(
        estimators=estimators,
        n_jobs=-1
    )
    
    # Fit the voting ensemble
    voting.fit(X_train, y_train)
    
    # Evaluate with cross-validation
    cv_scores = cross_val_score(
        voting, X_train, y_train,
        cv=cv if cv is not None else 10,
        scoring='r2',
        n_jobs=-1
    )
    
    mean_cv = cv_scores.mean()
    std_cv = cv_scores.std()
    
    logger.info(f"Voting Ensemble — CV R²: {mean_cv:.4f} ± {std_cv:.4f}")
    
    return {
        'model': voting,
        'name': 'Voting Ensemble',
        'best_params': {
            'base_learners': [r['name'] for r in top_results],
            'method': 'average',
        },
        'cv_score': mean_cv,
        'cv_std': std_cv,
        'cv_scores': cv_scores,
    }


def build_ensembles(all_results: list, X_train, y_train, cv=None, top_n: int = 3):
    """
    Build both Stacking and Voting ensembles from the top models.
    
    Args:
        all_results: List of all individual model results
        X_train: Training features
        y_train: Training targets
        cv: Cross-validation splitter
        top_n: Number of top models to use
        
    Returns:
        List of ensemble result dicts
    """
    logger.info("=" * 50)
    logger.info("Building Ensemble Models...")
    logger.info("=" * 50)
    
    # Select top models
    top_results = select_top_models(all_results, top_n=top_n)
    
    ensemble_results = []
    
    # Build stacking ensemble
    stacking_result = build_stacking_ensemble(top_results, X_train, y_train, cv)
    ensemble_results.append(stacking_result)
    
    # Build voting ensemble
    voting_result = build_voting_ensemble(top_results, X_train, y_train, cv)
    ensemble_results.append(voting_result)
    
    logger.info("All ensemble models built successfully.")
    return ensemble_results

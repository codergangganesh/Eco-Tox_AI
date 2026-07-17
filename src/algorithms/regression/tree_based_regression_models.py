"""
EcotoxAI: Tree-Based Models
Random Forest, Gradient Boosting, and XGBoost with hyperparameter tuning.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
import logging

logger = logging.getLogger(__name__)


def train_random_forest(X_train, y_train, cv=None):
    """
    Train Random Forest Regressor with RandomizedSearchCV.
    
    Random Forest builds multiple decision trees on random subsets
    of data and features, then averages predictions. Robust against
    overfitting and handles non-linear relationships well.
    """
    param_distributions = {
        'n_estimators': [100, 200, 300, 500, 700],
        'max_depth': [None, 5, 10, 15, 20, 30],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.5, 0.7, 1.0],
    }
    
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        model, param_distributions,
        n_iter=80,
        cv=cv if cv is not None else 10,
        scoring='r2',
        n_jobs=-1,
        verbose=0,
        random_state=42,
        return_train_score=True
    )
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    logger.info(
        f"Random Forest — Best params: {search.best_params_}, "
        f"CV R²: {search.best_score_:.4f}"
    )
    
    # Log feature importances
    importances = best_model.feature_importances_
    logger.info(f"  Top feature importances: {np.sort(importances)[::-1][:5]}")
    
    return {
        'model': best_model,
        'name': 'Random Forest',
        'best_params': search.best_params_,
        'cv_score': search.best_score_,
        'grid_search': search,
        'feature_importances': importances,
    }


def train_gradient_boosting(X_train, y_train, cv=None):
    """
    Train Gradient Boosting Regressor with RandomizedSearchCV.
    
    Builds trees sequentially, each correcting errors from the previous.
    More prone to overfitting than RF but potentially higher accuracy.
    """
    param_distributions = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    
    model = GradientBoostingRegressor(random_state=42)
    search = RandomizedSearchCV(
        model, param_distributions,
        n_iter=80,
        cv=cv if cv is not None else 10,
        scoring='r2',
        n_jobs=-1,
        verbose=0,
        random_state=42,
        return_train_score=True
    )
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    logger.info(
        f"Gradient Boosting — Best params: {search.best_params_}, "
        f"CV R²: {search.best_score_:.4f}"
    )
    
    return {
        'model': best_model,
        'name': 'Gradient Boosting',
        'best_params': search.best_params_,
        'cv_score': search.best_score_,
        'grid_search': search,
        'feature_importances': best_model.feature_importances_,
    }


def train_xgboost(X_train, y_train, cv=None):
    """
    Train XGBoost Regressor with RandomizedSearchCV.
    
    XGBoost is an optimized gradient boosting implementation with
    built-in L1/L2 regularization, making it the state-of-the-art
    for tabular data prediction tasks.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")
    
    param_distributions = {
        'n_estimators': [100, 200, 300, 500, 700],
        'max_depth': [3, 4, 5, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 1.0, 10.0],
        'reg_lambda': [0.1, 0.5, 1.0, 5.0, 10.0],
        'min_child_weight': [1, 3, 5, 7],
    }
    
    model = XGBRegressor(
        random_state=42, 
        n_jobs=-1, 
        verbosity=0,
        objective='reg:squarederror'
    )
    search = RandomizedSearchCV(
        model, param_distributions,
        n_iter=100,
        cv=cv if cv is not None else 10,
        scoring='r2',
        n_jobs=-1,
        verbose=0,
        random_state=42,
        return_train_score=True
    )
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    logger.info(
        f"XGBoost — Best params: {search.best_params_}, "
        f"CV R²: {search.best_score_:.4f}"
    )
    
    return {
        'model': best_model,
        'name': 'XGBoost',
        'best_params': search.best_params_,
        'cv_score': search.best_score_,
        'grid_search': search,
        'feature_importances': best_model.feature_importances_,
    }


def train_all_tree_models(X_train, y_train, cv=None):
    """
    Train all tree-based models and return results.
    
    Returns:
        List of result dictionaries for each model
    """
    logger.info("=" * 50)
    logger.info("Training Tree-Based Models...")
    logger.info("=" * 50)
    
    results = []
    results.append(train_random_forest(X_train, y_train, cv))
    results.append(train_gradient_boosting(X_train, y_train, cv))
    results.append(train_xgboost(X_train, y_train, cv))
    
    logger.info("All tree-based models trained successfully.")
    return results

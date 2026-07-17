"""
EcoTox-AI: Linear Models
Ridge Regression, Lasso Regression, and ElasticNet with hyperparameter tuning.
"""

import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
import logging

logger = logging.getLogger(__name__)


def train_ridge(X_train, y_train, cv=None):
    """
    Train Ridge Regression with GridSearchCV for optimal alpha.
    
    Ridge uses L2 regularization to prevent overfitting while
    keeping all features in the model.
    """
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
    }
    
    model = Ridge()
    grid_search = GridSearchCV(
        model, param_grid, 
        cv=cv if cv is not None else 10,
        scoring='r2',
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    logger.info(
        f"Ridge Regression — Best alpha: {grid_search.best_params_['alpha']}, "
        f"CV R²: {grid_search.best_score_:.4f}"
    )
    
    return {
        'model': best_model,
        'name': 'Ridge Regression',
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'grid_search': grid_search,
    }


def train_lasso(X_train, y_train, cv=None):
    """
    Train Lasso Regression with GridSearchCV.
    
    Lasso uses L1 regularization which can shrink some coefficients
    to exactly zero, effectively performing feature selection.
    """
    param_grid = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    }
    
    model = Lasso(max_iter=10000)
    grid_search = GridSearchCV(
        model, param_grid,
        cv=cv if cv is not None else 10,
        scoring='r2',
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    logger.info(
        f"Lasso Regression — Best alpha: {grid_search.best_params_['alpha']}, "
        f"CV R²: {grid_search.best_score_:.4f}"
    )
    
    # Report zeroed-out features
    zero_coefs = np.sum(best_model.coef_ == 0)
    if zero_coefs > 0:
        logger.info(f"  Lasso eliminated {zero_coefs} features (coefficients = 0)")
    
    return {
        'model': best_model,
        'name': 'Lasso Regression',
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'grid_search': grid_search,
    }


def train_elasticnet(X_train, y_train, cv=None):
    """
    Train ElasticNet with GridSearchCV.
    
    ElasticNet combines L1 and L2 regularization, controlled by:
    - alpha: overall regularization strength
    - l1_ratio: balance between L1 (1.0) and L2 (0.0)
    """
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
    }
    
    model = ElasticNet(max_iter=10000)
    grid_search = GridSearchCV(
        model, param_grid,
        cv=cv if cv is not None else 10,
        scoring='r2',
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    logger.info(
        f"ElasticNet — Best params: {grid_search.best_params_}, "
        f"CV R²: {grid_search.best_score_:.4f}"
    )
    
    return {
        'model': best_model,
        'name': 'ElasticNet',
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'grid_search': grid_search,
    }


def train_all_linear_models(X_train, y_train, cv=None):
    """
    Train all linear models and return results.
    
    Returns:
        List of result dictionaries for each model
    """
    logger.info("=" * 50)
    logger.info("Training Linear Models...")
    logger.info("=" * 50)
    
    results = []
    results.append(train_ridge(X_train, y_train, cv))
    results.append(train_lasso(X_train, y_train, cv))
    results.append(train_elasticnet(X_train, y_train, cv))
    
    logger.info("All linear models trained successfully.")
    return results

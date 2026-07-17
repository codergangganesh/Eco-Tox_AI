"""
EcotoxAI: Support Vector Regression Model
SVR with RBF kernel and hyperparameter tuning.
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import logging

logger = logging.getLogger(__name__)


def train_svr(X_train, y_train, cv=None):
    """
    Train Support Vector Regression with RBF kernel using GridSearchCV.
    
    SVR with RBF kernel maps data into a higher-dimensional space
    to find non-linear relationships. Particularly effective for
    small-to-medium datasets like our 908-chemical dataset.
    
    Key hyperparameters:
    - C: Regularization (higher = less regularization, tighter fit)
    - gamma: RBF kernel width (higher = tighter decision boundary)
    - epsilon: Margin of tolerance (points within epsilon are not penalized)
    """
    param_grid = {
        'C': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 0.5],
        'epsilon': [0.01, 0.05, 0.1, 0.2, 0.5],
    }
    
    model = SVR(kernel='rbf')
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
        f"SVR (RBF) — Best params: {grid_search.best_params_}, "
        f"CV R²: {grid_search.best_score_:.4f}"
    )
    
    # Log support vector count
    n_sv = best_model.n_support_
    logger.info(f"  Number of support vectors: {sum(n_sv)}")
    
    return {
        'model': best_model,
        'name': 'SVR (RBF)',
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'grid_search': grid_search,
    }
